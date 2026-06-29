#!/usr/bin/env python3
import argparse
import time
import threading
import collections
import zmq
import numpy as np
import sys
import os
import copy

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stretch4_emulated_rgbd.api import get_emulated_rgbd_stream
from stretch4_emulated_rgbd.shared_utils import get_arg_parser, ExtrinsicsCalibration
from stretch4_emulated_rgbd import rgbd_networking as gn
import stretch4_body.robot.robot_client as rc

class RobotStatePoller:
    def __init__(self, robot):
        self.robot = robot
        self.history_buffer = []
        self.lock = threading.Lock()
        self.running = True
        self.last_ts = None
        self.state_counter = 0
        
        self.thread = threading.Thread(target=self._poll_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def _poll_loop(self):
        while self.running:
            self.robot.pull_status()
            st = self.robot.status
            
            # The pimu timestamp is a good pulse check for new lower-level samples
            pimu_st = st.get('pimu', {})
            curr_ts = pimu_st.get('timestamp', 0.0)
            
            if self.last_ts is not None and curr_ts == self.last_ts:
                # No new data update from lower level hardware
                time.sleep(0.002)
                continue
                
            self.last_ts = curr_ts
            
            eoa = st.get('end_of_arm', {})
            gripper_st = eoa.get('stretch_gripper', {})
            base = st.get('base', {})
            lift_st = st.get('lift', {})
            arm_st = st.get('arm', {})
            
            wrist_yaw = eoa.get('wrist_yaw', {})
            wrist_pitch = eoa.get('wrist_pitch', {})
            wrist_roll = eoa.get('wrist_roll', {})
            
            data = {
                'gripper': {
                    'pos_pct': gripper_st.get('pos_pct', 0.0), 
                    'effort': gripper_st.get('effort', 0.0)
                },
                'lift': {
                    'height': lift_st.get('pos', 0.0)
                },
                'arm': {
                    'extension': arm_st.get('pos', 0.0)
                },
                'wrist_yaw': {
                    'angle': wrist_yaw.get('pos', 0.0), 
                    'effort': wrist_yaw.get('effort', 0.0)
                },
                'wrist_pitch': {
                    'angle': wrist_pitch.get('pos', 0.0), 
                    'effort': wrist_pitch.get('effort', 0.0)
                },
                'wrist_roll': {
                    'angle': wrist_roll.get('pos', 0.0), 
                    'effort': wrist_roll.get('effort', 0.0)
                },
                'base_odometry': {
                    'x': base.get('x', 0.0), 
                    'y': base.get('y', 0.0), 
                    'theta': base.get('theta', 0.0)
                },
                'timestamp': time.time(),
                'monotonic_timestamp': time.monotonic(),
                'state_number': self.state_counter
            }
            
            with self.lock:
                self.history_buffer.append(data)
                self.state_counter += 1
                
            time.sleep(0.002) # Ensure we don't thrash CPU, ~500Hz max polling
            
    def stop(self):
        self.running = False
        self.thread.join()
        
    def get_and_clear_history(self):
        with self.lock:
            history = list(self.history_buffer)
            self.history_buffer.clear()
            return history

def main():
    parser = get_arg_parser("Publish Fast Emulated RGB-D stream and Joint States over PyZMQ.")
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when allowing a remote computer to receive images. Configure rgbd_networking.py first.')
    args = parser.parse_args()
    
    print("Starting Robot Client...")
    robot = rc.RobotClient()
    robot.startup()
    
    if not robot.is_homed():
        print("WARNING: Robot is not homed. Joint values may be incorrect.")
        
    poller = RobotStatePoller(robot)
    
    print(f"Initializing Fast Emulated RGB-D Streamer at {args.emulated_rgbd_fps} FPS...")
    
    calibration = None
    if args.opt_yaml:
        calibration = ExtrinsicsCalibration.load_from_yaml(args.opt_yaml)
        if calibration is None:
            return
            
    streamer = None
    try:
        streamer, generator = get_emulated_rgbd_stream(
            use_left=(args.camera in ["left", "left_right", "all"]),
            use_right=(args.camera in ["right", "left_right", "all"]),
            use_center=(args.camera in ["center", "all"]),
            use_left_right=(args.camera == "left_right"),
            use_left_right_center=(args.camera == "all"),
            use_left_lidar=(args.lidar in ["left", "both"]),
            use_right_lidar=(args.lidar in ["right", "both"]),
            emulated_rgbd_fps=args.emulated_rgbd_fps,
            camera_fps=args.camera_fps,
            resolution_height=args.resolution,
            compress=not args.disable_compression,
            oak_buffer_size=args.oak_buffer_size,
            calibration=calibration
        )
    except RuntimeError as e:
        print(f"Error initializing streamer: {e}")
        return
        
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    
    if args.remote:
        address = f"tcp://*:{gn.rgbd_and_joints_port}"
    else:
        address = f"tcp://127.0.0.1:{gn.rgbd_and_joints_port}"
        
    print(f"Binding ZMQ Publisher to {address}")
    socket.bind(address)
    gn.print_network_info()
    
    sliding_window = collections.deque(maxlen=500)
    robot_id = os.environ.get('HELLO_FLEET_ID', 'unknown')
    
    print("Streaming started. Press Ctrl+C to exit.")
    
    try:
        for frame in generator:
            if frame is None:
                continue
                
            cam_timestamp = frame.timestamp
            if cam_timestamp is None:
                cam_timestamp = time.monotonic()
                
            system_boot_epoch = time.time() - time.monotonic()
            sys_timestamp = system_boot_epoch + cam_timestamp
            
            new_history = poller.get_and_clear_history()
            sliding_window.extend(new_history)
            
            closest_joint_state = None
            min_diff = float('inf')
            
            for state in sliding_window:
                time_relative = state['monotonic_timestamp'] - cam_timestamp
                state['time_relative_to_image'] = time_relative
                if abs(time_relative) < min_diff:
                    min_diff = abs(time_relative)
                    closest_joint_state = copy.deepcopy(state)
                    
            if closest_joint_state is not None:
                closest_joint_state['time_relative_to_image'] = closest_joint_state['monotonic_timestamp'] - cam_timestamp
                
            for state in new_history:
                state['time_relative_to_image'] = state['monotonic_timestamp'] - cam_timestamp
                
            output_dict = frame.to_dict()
            output_dict['robot_id'] = robot_id
            output_dict['system_timestamp'] = sys_timestamp
            output_dict['joint_state_history'] = new_history
            output_dict['closest_joint_state'] = closest_joint_state
            
            # Send
            socket.send_pyobj(output_dict)
            
    except KeyboardInterrupt:
        print("\nStopping publisher...")
    finally:
        poller.stop()
        robot.stop()
        if streamer:
            streamer.stop()

if __name__ == "__main__":
    main()

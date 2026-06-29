#!/usr/bin/env python3
import argparse
import time
import zmq
import sys
import os

import rerun as rr
import rerun.blueprint as rrb

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stretch4_emulated_rgbd.shared_utils import RGBDFrame
from stretch4_emulated_rgbd.api import visualize_rgbd_frame, ValidityMaskManager
from stretch4_emulated_rgbd import rgbd_networking as gn

def main():
    parser = argparse.ArgumentParser(description="Receive Fast Emulated RGB-D stream and Joint States over PyZMQ.")
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when running the code on a remote computer. Configure rgbd_networking.py first.')
    parser.add_argument('--disable-rate-print', action='store_true', help='Disable printing of the receiving rate and dropped messages.')
    args = parser.parse_args()
    
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    
    if args.remote:
        address = f"tcp://{gn.robot_ip}:{gn.rgbd_and_joints_port}"
    else:
        address = f"tcp://127.0.0.1:{gn.rgbd_and_joints_port}"
        
    print(f"Connecting ZMQ Subscriber to {address}")
    socket.connect(address)
    
    print("Initializing ReRun...")
    rr.init("Stretch Emulated RGBD with Joint States", spawn=False)
    rr.spawn(memory_limit="2GiB")

    # We don't know the exact cameras until we receive frames, but we can set up a generic layout
    camera_views = [
        rrb.Spatial2DView(name="Left Camera", origin="Cameras/left"),
        rrb.Spatial2DView(name="Right Camera", origin="Cameras/right"),
        rrb.Spatial2DView(name="Center Camera", origin="Cameras/center")
    ]
    
    timeseries_views = [
        rrb.TimeSeriesView(name="Lift & Arm", origin="Telemetry/LiftArm"),
        rrb.TimeSeriesView(name="Wrist", origin="Telemetry/Wrist")
    ]

    view_layout = rrb.Horizontal(
        rrb.Spatial3DView(name="Base Frame", origin="/", contents=["+ Pointclouds/base_frame/**"]),
        rrb.Vertical(
            rrb.Horizontal(*camera_views),
            rrb.Horizontal(*timeseries_views)
        ),
        column_shares=[2, 3]
    )

    blueprint = rrb.Blueprint(
        view_layout,
        rrb.BlueprintPanel(expanded=False),
        rrb.TimePanel(play_state="following"),
    )
    rr.send_blueprint(blueprint)

    mask_manager = None

    print("Receiving stream... Press 'Ctrl+C' to exit.")
    
    last_print_time = time.time()
    frames_received = 0
    last_seq_num = None
    dropped_messages = 0
    
    try:
        while True:
            output_dict = socket.recv_pyobj()
            frame = RGBDFrame.from_dict(output_dict)
            
            if mask_manager is None:
                robot_id = output_dict.get('robot_id')
                if robot_id and robot_id != 'unknown':
                    fleet_path = os.environ.get("HELLO_FLEET_PATH", os.path.expanduser('~/stretch_user'))
                    masks_dir = os.path.join(fleet_path, robot_id, "calibration_cameras")
                    mask_manager = ValidityMaskManager(masks_dir=masks_dir)
                else:
                    mask_manager = ValidityMaskManager()
                    
            closest_joint_state = output_dict.get('closest_joint_state', None)
            
            frames_received += 1
            img_seq = frame.image_frame.frame_number
            if img_seq is not None:
                if last_seq_num is not None:
                    dropped = img_seq - last_seq_num - 1
                    if dropped > 0:
                        dropped_messages += dropped
                last_seq_num = img_seq
                
            # Log Joint States
            if closest_joint_state is not None:
                # Log to rerun telemetry
                rr.set_time("timestamp", timestamp=closest_joint_state['monotonic_timestamp'])
                
                rr.log("Telemetry/LiftArm/Lift", rr.Scalars(closest_joint_state['lift']['height']))
                rr.log("Telemetry/LiftArm/Arm", rr.Scalars(closest_joint_state['arm']['extension']))
                rr.log("Telemetry/LiftArm/Gripper", rr.Scalars(closest_joint_state['gripper']['pos_pct']))
                
                rr.log("Telemetry/Wrist/Yaw", rr.Scalars(closest_joint_state['wrist_yaw']['angle']))
                rr.log("Telemetry/Wrist/Pitch", rr.Scalars(closest_joint_state['wrist_pitch']['angle']))
                rr.log("Telemetry/Wrist/Roll", rr.Scalars(closest_joint_state['wrist_roll']['angle']))
              
            # Log RGB-D Frames
            c_name = frame.camera_type
            lidar_str = frame.lidars_used if frame.lidars_used else "no_lidar"
            
            vig_mask, depth_mask = mask_manager.get_masks(c_name, lidar_str, frame.image.shape)
            visualize_rgbd_frame(c_name, frame, vig_mask=vig_mask, depth_mask=depth_mask)
            
            # Print stats
            current_time = time.time()
            elapsed = current_time - last_print_time
            if elapsed >= 5.0:
                if not args.disable_rate_print:
                    hz = frames_received / elapsed
                    print(f"Rate: {hz:.2f} Hz | Estimated dropped messages in last {elapsed:.1f}s: {dropped_messages}")
                frames_received = 0
                dropped_messages = 0
                last_print_time = current_time
                
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopped receiving.")

if __name__ == "__main__":
    main()

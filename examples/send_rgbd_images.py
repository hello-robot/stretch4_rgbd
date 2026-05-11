#!/usr/bin/env python3
import argparse
import time
import zmq
import numpy as np
import sys
import os

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stretch4_emulated_rgbd.api import get_emulated_rgbd_stream
from stretch4_emulated_rgbd.shared_utils import get_arg_parser, ExtrinsicsCalibration

def main():
    parser = get_arg_parser("Publish Fast Emulated RGB-D stream over PyZMQ.")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ port to publish on (default: 5556)")
    args = parser.parse_args()
    
    print(f"Initializing Fast Emulated RGB-D Streamer at {args.emulated_rgbd_fps} FPS...")
    
    calibration = None
    if args.opt_yaml:
        calibration = ExtrinsicsCalibration.load_from_yaml(args.opt_yaml)
        if calibration is None:
            return
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
    
    address = f"tcp://*:{args.port}"
    print(f"Binding ZMQ Publisher to {address}")
    socket.bind(address)
    
    print("Streaming started. Press Ctrl+C to exit.")
    
    robot_id = os.environ.get('HELLO_FLEET_ID', 'unknown')
    
    try:
        for frame in generator:
            if frame is None:
                continue
                
            output_dict = frame.to_dict()
            output_dict['robot_id'] = robot_id
            
            # Send
            socket.send_pyobj(output_dict)
            
    except KeyboardInterrupt:
        print("\nStopping publisher...")
    finally:
        if streamer:
            streamer.stop()

if __name__ == "__main__":
    main()

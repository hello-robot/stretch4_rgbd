#!/usr/bin/env python3
import argparse
import time
import zmq
import numpy as np
import cv2
import sys
import os

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stretch4_emulated_rgbd.shared_utils import RGBDFrame, ValidityMaskManager

def apply_color_map(depth_image, max_depth=3.0):
    valid_mask = (depth_image > 0)
    depth_norm = np.clip(depth_image / max_depth, 0, 1)
    
    depth_colormap = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    depth_colormap[~valid_mask] = 0
    return depth_colormap

def main():
    parser = argparse.ArgumentParser(description="Receive Fast Emulated RGB-D stream over PyZMQ.")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address of the publisher (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ port to connect to (default: 5556)")
    args = parser.parse_args()
    
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    
    address = f"tcp://{args.ip}:{args.port}"
    print(f"Connecting ZMQ Subscriber to {address}")
    socket.connect(address)
    
    print("Receiving stream... Press 'q' or 'Esc' to exit.")
    
    last_print_time = time.time()
    frames_received = 0
    mask_manager = None
    
    try:
        while True:
            output_dict = socket.recv_pyobj()
            frame = RGBDFrame.from_dict(output_dict)
            frames_received += 1
            
            if mask_manager is None:
                robot_id = output_dict.get('robot_id')
                if robot_id and robot_id != 'unknown':
                    fleet_path = os.environ.get("HELLO_FLEET_PATH", os.path.expanduser('~/stretch_user'))
                    masks_dir = os.path.join(fleet_path, robot_id, "calibration_cameras")
                    mask_manager = ValidityMaskManager(masks_dir=masks_dir)
                else:
                    mask_manager = ValidityMaskManager()
            
            # Access the cleanly aligned, decompressed, and rotated image 
            # simply by calling frame.image (handled transparently by RGBDFrame)
            color_image = frame.image
            depth_image = frame.depth_image
            
            c_name = getattr(frame, 'camera_type', 'left')
            lidar_str = getattr(frame, 'lidars_used', 'no_lidar')
            if not lidar_str:
                lidar_str = "no_lidar"
                
            shape = color_image.shape if color_image is not None else (depth_image.shape if depth_image is not None else (0,0))
            if shape != (0,0):
                vig_mask, depth_mask = mask_manager.get_masks(c_name, lidar_str, shape)
                
                if color_image is not None and vig_mask is not None:
                    color_image[~vig_mask] = 0
                    
                if depth_image is not None and depth_mask is not None:
                    depth_image[~depth_mask] = 0
            
            # Show RGB
            if color_image is not None:
                cv2.namedWindow("RGB Stream", cv2.WINDOW_NORMAL)
                cv2.imshow("RGB Stream", color_image)
                
            # Show Depth
            if depth_image is not None:
                depth_vis = apply_color_map(depth_image)
                cv2.namedWindow("Depth Stream", cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Stream", depth_vis)
                
            key = cv2.waitKey(1)
            if key in (27, ord('q')):
                break
                
            # Print stats
            current_time = time.time()
            elapsed = current_time - last_print_time
            if elapsed >= 2.0:
                fps = frames_received / elapsed
                print(f"Receiving at {fps:.2f} FPS")
                frames_received = 0
                last_print_time = current_time
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("\nStopped receiving.")

if __name__ == "__main__":
    main()

#!/usr/env/bin python3
import os
import cv2
import numpy as np
import yaml
import time
from datetime import datetime
from pathlib import Path

import sys

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stretch4_emulated_rgbd.api import get_emulated_rgbd_stream

from stretch4_emulated_rgbd.shared_utils import get_arg_parser, NonBlockingInput, DenseDepthImage

def save_captured_data(capture_dir, seq_num, args, synced_frames, streamer):
    from stretch4_emulated_rgbd.shared_utils import CapturedSequence

    seq_dir = capture_dir / f"emulated_rgbd_{seq_num:08d}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    lidar_str = "no_lidar"
    if args.lidar == "both":
        lidar_str = "both_lidar"
    elif args.lidar == "left":
        lidar_str = "left_lidar"
    elif args.lidar == "right":
        lidar_str = "right_lidar"
        
    frames_to_save = []
    # If not using left_right or left_right_center, synced_frames is just a single RGBDFrame
    # We standardise it to a list of tuples: (camera_name, frame)
    if hasattr(synced_frames, "left"): # SyncedRGBDFrame
        if synced_frames.left is not None: frames_to_save.append(("left", synced_frames.left))
        if synced_frames.right is not None: frames_to_save.append(("right", synced_frames.right))
        if getattr(synced_frames, "center", None) is not None: frames_to_save.append(("center", synced_frames.center))
    else:
        # single frame
        if synced_frames is not None:
            camera_name = args.camera if args.camera in ["left", "right", "center"] else "center"
            frames_to_save.append((camera_name, synced_frames))
            
    for c_name, frame in frames_to_save:
        if frame is None: continue
        
        distinct_dir = seq_dir / f"{c_name}_camera_{lidar_str}_{timestamp_str}"
        
        lidar_left_pts = streamer.latest_lidar_pts.get("left")
        lidar_right_pts = streamer.latest_lidar_pts.get("right")
        
        calib = getattr(streamer, 'calibs', {}).get(c_name)
        T_base_to_cam_factory = getattr(streamer, 'T_base_to_cam_factory', {}).get(c_name)
        T_base_to_cam_optimized = getattr(streamer, 'T_base_to_cam_optimized', {})
        T_base_to_cam_opt_val = T_base_to_cam_optimized.get(c_name) if T_base_to_cam_optimized else None
        
        def _safe_float(val):
            if val is None: return None
            try:
                if hasattr(val, "size") and val.size == 0: return None
                return float(np.atleast_1d(val)[0])
            except:
                return float(val)

        metadata_extras = {
            "camera_model": getattr(calib, "camera_model", "fisheye" if (calib and calib.is_fisheye) else "pinhole") if calib else "pinhole",
            "omnidir_xi": _safe_float(getattr(calib, "omnidir_xi", 0.0)) if calib else 0.0,
            "is_fisheye": bool(np.atleast_1d(calib.is_fisheye)[0]) if calib and hasattr(calib.is_fisheye, "size") and calib.is_fisheye.size > 0 else bool(calib.is_fisheye) if calib else False,
            "T_base_to_cam_factory": T_base_to_cam_factory.tolist() if T_base_to_cam_factory is not None else None,
            "T_base_to_cam_optimized": T_base_to_cam_opt_val.tolist() if T_base_to_cam_opt_val is not None else None,
        }
        
        seq = CapturedSequence(frame, raw_lidar_left=lidar_left_pts, raw_lidar_right=lidar_right_pts)
        seq.save(distinct_dir, save_dense_depth=True, metadata_extras=metadata_extras)
            
    print(f"Captured sequence {seq_num} at {timestamp_str}")


def main():
    parser = get_arg_parser("Capture Emulated RGB-D Imagery to Disk.")
    parser.add_argument("--ignore_prior_optimizations", action="store_true", help="Capture using raw factory calibration.")
    args = parser.parse_args()

    use_left = args.camera in ["left", "left_right", "all"]
    use_right = args.camera in ["right", "left_right", "all"]
    use_center = args.camera in ["center", "all"]
    use_left_right = (args.camera == "left_right")
    use_left_right_center = (args.camera == "all")
    use_left_lidar = (args.lidar in ["left", "both"])
    use_right_lidar = (args.lidar in ["right", "both"])

    print("Initializing RGBD Streamer with Lidars...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_dir = Path(f"data/captured_emulated_rgbd_{timestamp}")
    capture_dir.mkdir(parents=True, exist_ok=True)
    
    capture_metadata = {
        "capture_fleet_id": os.environ.get("HELLO_FLEET_ID", "UNKNOWN")
    }
    with open(capture_dir / "capture_metadata.yaml", "w") as f:
        yaml.dump(capture_metadata, f, default_flow_style=False)
        
    print(f"Data will be saved to: {capture_dir}")
    print("Press SPACE in terminal to capture, Q to quit.")

    streamer, stream_generator = get_emulated_rgbd_stream(
        use_left=use_left,
        use_right=use_right,
        use_center=use_center,
        use_left_right=use_left_right,
        use_left_right_center=use_left_right_center,
        use_left_lidar=use_left_lidar,
        use_right_lidar=use_right_lidar,
        ignore_prior_optimizations=args.ignore_prior_optimizations
    )

    input_listener = NonBlockingInput()
    seq_num = 1
    
    try:
        for synced_frame in stream_generator:
            if synced_frame is None:
                continue
                
            char = input_listener.get_char()
            if char is not None:
                if char.lower() == 'q':
                    print("Quitting...")
                    break
                elif char == ' ':
                    save_captured_data(capture_dir, seq_num, args, synced_frame, streamer)
                    seq_num += 1

    except KeyboardInterrupt:
        print("Stopping... (Ctrl+C pressed)")
    finally:
        input_listener.cleanup()
        if streamer: streamer.stop()
        print("Capture session ended.")

if __name__ == "__main__":
    main()

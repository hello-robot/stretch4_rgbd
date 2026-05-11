#!/usr/env/bin python3
import os
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import yaml
from pathlib import Path
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stretch_body_ii.subsystem.cameras.emulated_rgbd import (
        stream_left_rgbd,
        stream_right_rgbd,
        stream_center_rgbd,
        stream_left_right_rgbd,
        stream_left_right_center_rgbd,
        EmulatedRGBDStreamer,
    )
    HAS_STRETCH_BODY = True
except ImportError:
    HAS_STRETCH_BODY = False

from stretch4_emulated_rgbd.shared_utils import get_arg_parser, reconstruct_rgbd_frame, LoopTimer
from stretch4_emulated_rgbd.api import (
    get_emulated_rgbd_stream, 
    ValidityMaskManager, 
    visualize_rgbd_frame, 
    ExtrinsicsCalibration
)



from stretch4_emulated_rgbd.shared_utils import get_arg_parser, reconstruct_rgbd_frame, LoopTimer, CapturedSequence

def replay_sequence(data_path, calibration: ExtrinsicsCalibration = None):
    print(f"Replaying from {data_path}")
    base_dir = Path(data_path)
    seq_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("emulated_rgbd_")])
    
    if not seq_dirs:
        print("No sequence directories found.")
        return
        
    current_replay_time = 0.0
    loaded_masks = {}
    for seq_dir in seq_dirs:
        distinct_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
        for d in distinct_dirs:
            meta_path = d / "metadata.yaml"
            if not meta_path.exists(): continue
            
            try:
                seq = CapturedSequence.load(d)
            except Exception as e:
                print(f"Failed to load sequence {d}: {e}")
                continue
                
            # Override time for sequential replay
            seq.frame.timestamp = current_replay_time
            
            if calibration is not None and seq.frame.T_base_to_cam is not None:
                seq.frame._T_base_to_cam = calibration.apply_to_camera_extrinsics(seq.frame.T_base_to_cam)
                
            c_name = seq.frame.camera_type
            lidar_str = seq.frame.lidars_used if seq.frame.lidars_used else "left_lidar"
                    
            # Apply masks in replay if they exist
            mask_key = f"{c_name}_{lidar_str}"
            if mask_key not in loaded_masks:
                from stretch4_emulated_rgbd.shared_utils import load_and_confirm_optimization_masks
                try:
                    vig_mask, depth_mask = load_and_confirm_optimization_masks(
                        data_path, c_name, lidar_str
                    )
                except Exception as e:
                    print(f"Skipping optimization masks for {c_name}: {e}")
                    vig_mask, depth_mask = None, None
                loaded_masks[mask_key] = (vig_mask, depth_mask)
            
            vig_mask, depth_mask = loaded_masks[mask_key]
            
            # Ensure masks match the image resolution
            h, w = seq.frame.image.shape[:2]
            if vig_mask is not None and vig_mask.shape != (h, w):
                vig_mask = cv2.resize(vig_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            if depth_mask is not None and depth_mask.shape != (h, w):
                depth_mask = cv2.resize(depth_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            
            visualize_rgbd_frame(c_name, seq.frame, vig_mask=vig_mask, depth_mask=depth_mask)
            
        current_replay_time += 1.0
        time.sleep(0.1)
        
    # Extend the timeline by 1 second after the final image
    rr.set_time("timestamp", timestamp=current_replay_time)
    rr.log("TimelineEnd", rr.TextDocument("Replay Complete"))

def main():
    parser = get_arg_parser("Visualize colored point clouds and RGBD streams from Stretch lidars and cameras in rerun.")
    parser.add_argument("--data_path", nargs='?', const='LATEST', default=None, help="Path to captured data directory to replay. If passed without a path, reloads the latest.")
    args = parser.parse_args()
    
    print("Initializing ReRun...")
    rr.init("Stretch Emulated RGBD", spawn=False)
    rr.spawn(memory_limit="2GiB")

    camera_views = []
    if args.camera in ["left", "left_right", "all"]:
        camera_views.append(rrb.Spatial2DView(name="Left Camera", origin="Cameras/left"))
    if args.camera in ["center", "all"]:
        camera_views.append(rrb.Spatial2DView(name="Center Camera", origin="Cameras/center"))
    if args.camera in ["right", "left_right", "all"]:
        camera_views.append(rrb.Spatial2DView(name="Right Camera", origin="Cameras/right"))

    view_layout = rrb.Horizontal(
        rrb.Spatial3DView(name="Base Frame", origin="/", contents=["+ Pointclouds/base_frame/**"]),
        rrb.Vertical(*camera_views) if len(camera_views) > 1 else camera_views[0],
        column_shares=[3, 1]
    )

    blueprint = rrb.Blueprint(
        view_layout,
        rrb.BlueprintPanel(expanded=True),
        rrb.TimePanel(play_state="following"),
    )
    rr.send_blueprint(blueprint)

    calibration = None
    if args.opt_yaml:
        calibration = ExtrinsicsCalibration.load_from_yaml(args.opt_yaml)
        if calibration is None:
            return

    # REPLAY MODE
    replay_dir = args.data_path
    if replay_dir == 'LATEST':
        data_dir = Path("data")
        if data_dir.exists():
            captures = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("captured_emulated_rgbd_")])
            if captures:
                replay_dir = str(captures[-1])
            else:
                replay_dir = None
                print("No captured data found to replay.")
                
    if replay_dir is not None and replay_dir != 'LATEST' and Path(replay_dir).exists():
        replay_sequence(replay_dir, calibration=calibration)
        print("Replay finished.")
        return
        
    # LIVE STREAM MODE
    show_fps = args.show_fps
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
        print(e)
        return

    if streamer is None:
        return
        
    mask_manager = ValidityMaskManager()

    print("Streaming started. Ctrl+C to exit.")
    loop_timer = LoopTimer()
    loop_timer.start_of_iteration()
    def print_loop_timer():
        if not show_fps: return
        loop_timer.end_of_iteration()
        loop_timer.pretty_print(minimum=True)
        loop_timer.start_of_iteration()
        
    try:
        for frame_data in generator:
            print_loop_timer()
            if frame_data is None: return
            
            lidar_str = "no_lidar"
            if args.lidar == "both":
                lidar_str = "both_lidar"
            elif args.lidar == "left":
                lidar_str = "left_lidar"
            elif args.lidar == "right":
                lidar_str = "right_lidar"
                
            # Helper to render and apply masks
            def _render_with_masks(c_name, frame):
                vig_mask, depth_mask = mask_manager.get_masks(c_name, lidar_str, frame.image.shape)
                visualize_rgbd_frame(c_name, frame, vig_mask=vig_mask, depth_mask=depth_mask)

            if hasattr(frame_data, "left") or hasattr(frame_data, "right") or hasattr(frame_data, "center"):
                if hasattr(frame_data, "left") and frame_data.left: _render_with_masks("left", frame_data.left)
                if hasattr(frame_data, "right") and frame_data.right: _render_with_masks("right", frame_data.right)
                if hasattr(frame_data, "center") and frame_data.center: _render_with_masks("center", frame_data.center)
            else:
                _render_with_masks(frame_data.camera_type, frame_data)

    except KeyboardInterrupt:
        print("Stopping... (Force quitting due to background threads)")
        os._exit(0)
    finally:
        streamer.stop()

if __name__ == "__main__":
    main()

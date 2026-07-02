import sys
import os
import cv2
import numpy as np

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stretch4_emulated_rgbd.api import (
    get_emulated_rgbd_stream,
    ValidityMaskManager,
    DenseDepthImage,
    create_point_cloud_from_depth
)
from stretch4_emulated_rgbd.shared_utils import get_arg_parser
from stretch4_emulated_rgbd import emulated_rgbd_config as config

def main():
    parser = get_arg_parser("Stretch 4 Emulated RGB-D API Capabilities Demonstration")
    args = parser.parse_args()

    use_left = args.camera in ["left", "left_right", "all"]
    use_right = args.camera in ["right", "left_right", "all"]
    use_center = args.camera in ["center", "all"]
    use_left_right = (args.camera == "left_right")
    use_left_right_center = (args.camera == "all")
    use_left_lidar = (args.lidar in ["left", "both"])
    use_right_lidar = (args.lidar in ["right", "both"])

    print("=" * 60)
    print(" Stretch 4 Emulated RGB-D API Capabilities Demonstration")
    print("=" * 60)

    # 1. Initialize the RGB-D Stream
    print("\n[1] Initializing Stream...")
    streamer, generator = get_emulated_rgbd_stream(
        use_left=use_left,
        use_right=use_right,
        use_center=use_center,
        use_left_right=use_left_right,
        use_left_right_center=use_left_right_center,
        use_left_lidar=use_left_lidar,
        use_right_lidar=use_right_lidar,
        emulated_rgbd_fps=args.emulated_rgbd_fps,
        camera_fps=args.camera_fps,
        resolution_height=args.resolution,
        compress=not args.disable_compression,
        oak_buffer_size=args.oak_buffer_size,
        merge_lidars=args.merge_lidars
    )


    # 2. Initialize the Validity Mask Manager
    print("\n[2] Initializing Validity Mask Manager...")
    mask_manager = ValidityMaskManager()

    try:
        print("\n[3] Streaming frames for Visualization...")
        print("    -> OpenCV visualization active. Press 'q' or ESC to switch to Rerun.")
        
        cv2.namedWindow("Masked RGB", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Dense Depth", cv2.WINDOW_NORMAL)
        
        rerun_mode = False
        rerun_initialized = False

        # The generator yields synchronized frames indefinitely
        for frame_data in generator:
            if frame_data is None:
                continue
                
            # Handle both single frames and multi-frame objects
            frames = []
            if hasattr(frame_data, "left") or hasattr(frame_data, "right") or hasattr(frame_data, "center"):
                if getattr(frame_data, "left", None): frames.append(frame_data.left)
                if getattr(frame_data, "right", None): frames.append(frame_data.right)
                if getattr(frame_data, "center", None): frames.append(frame_data.center)
            else:
                frames.append(frame_data)

            # For visualization stacking
            all_rgb_masked = []
            all_depth_vis = []

            for frame in frames:
                # 4. Access Lazy Properties
                rgb_image = frame.image
                depth_image = frame.depth_image
                
                # 5. Access Calibration Data
                cam_matrix = frame.camera_matrix
                dist_coeffs = frame.distortion_coefficients
                T_base_to_cam = frame.T_base_to_cam
                
                # 6. Apply Validity Masks
                c_name = frame.camera_type
                lidar_str = frame.lidars_used if frame.lidars_used else "no_lidar"
                vig_mask, lidar_mask = mask_manager.get_masks(c_name, lidar_str, rgb_image.shape)
                
                dense_depth_validity_mask = vig_mask & lidar_mask
                
                # Apply the vignetting mask to remove invalid fisheye edges from the RGB image
                masked_rgb = rgb_image.copy()
                masked_rgb[~vig_mask] = 0
                
                # 7. Generate Dense Depth Map
                dense_processor = DenseDepthImage(rgb_image, depth_image, apply_validity_mask=False)
                dense_depth = dense_processor.compute_dense_depth()
                
                # Before creating a point cloud, apply the eroded combined mask to drop unstable boundary pixels
                dense_depth[~dense_depth_validity_mask] = 0
                
                # 8. Create Colored Point Cloud
                pts_cam, colors = create_point_cloud_from_depth(
                    dense_depth, masked_rgb, cam_matrix, dist_coeffs
                )
                
                # Transform points to the robot's base coordinate frame for correct upright 3D viewing
                pts_cam_homog = np.hstack((pts_cam, np.ones((pts_cam.shape[0], 1))))
                T_cam_to_base = np.linalg.inv(T_base_to_cam)
                pts_base = (T_cam_to_base @ pts_cam_homog.T).T[:, :3]

                # 9. Visualization
                if not rerun_mode:
                    # 9a. OpenCV Visualization
                    max_depth = 5.0
                    depth_vis = np.clip(dense_depth, 0, max_depth) / max_depth
                    depth_vis = (depth_vis * 255).astype(np.uint8)
                    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    depth_colormap[dense_depth == 0] = [0, 0, 0]
                    
                    all_rgb_masked.append(masked_rgb)
                    all_depth_vis.append(depth_colormap)
                else:
                    # 9b. Rerun Visualization
                    if not rerun_initialized:
                        try:
                            import rerun as rr
                            import rerun.blueprint as rrb
                            print("    -> Spawning Rerun viewer...")
                            rr.init("api_example_visualization", spawn=True)
                            
                            blueprint = rrb.Blueprint(
                                rrb.Horizontal(
                                    rrb.Spatial3DView(name="Sparse Point Cloud", origin="sparse_view"),
                                    rrb.Spatial3DView(name="Dense Point Cloud", origin="dense_view"),
                                    rrb.Spatial2DView(name="Layered RGB-D", origin="camera"),
                                ),
                                rrb.BlueprintPanel(expanded=False),
                                rrb.SelectionPanel(expanded=True),
                                rrb.TimePanel(expanded=False, play_state="following"),
                            )
                            rr.send_blueprint(blueprint)
                            rerun_initialized = True
                        except ImportError:
                            print("    -> Rerun is not installed. Exiting.")
                            break
                    
                    # Prefix paths with camera name for multi-camera support in Rerun
                    prefix = f"camera/{c_name}/"
                    rr.log(prefix + "rgb", rr.Image(masked_rgb[:, :, ::-1])) 
                    rr.log(prefix + "dense_depth", rr.DepthImage(dense_depth, meter=1.0, depth_range=[0.0, config.RERUN_COLOR_MAX_DEPTH_M]))
                    rr.log(prefix + "sparse_depth", rr.DepthImage(depth_image, meter=1.0, depth_range=[0.0, config.RERUN_COLOR_MAX_DEPTH_M]))
                    
                    rr.log(
                        f"sparse_view/{c_name}/point_cloud", 
                        rr.Points3D(frame.point_cloud_base, colors=frame.point_colors, radii=[0.01]) 
                    )
                    rr.log(
                        f"dense_view/{c_name}/point_cloud", 
                        rr.Points3D(pts_base, colors=colors[:, ::-1], radii=[0.01]) 
                    )

            if not rerun_mode and all_rgb_masked:
                # Tile images if there are multiple
                stacked_rgb = np.hstack(all_rgb_masked)
                stacked_depth = np.hstack(all_depth_vis)
                cv2.imshow("Masked RGB", stacked_rgb)
                cv2.imshow("Dense Depth", stacked_depth)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    rerun_mode = True
                    cv2.destroyAllWindows()
                    print("\n[9b] Switching to Rerun Visualization (press Ctrl+C to exit)...")

        
    finally:
        print("\nStopping streamer...")
        streamer.stop()

if __name__ == "__main__":
    main()

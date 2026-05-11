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
from stretch4_emulated_rgbd import emulated_rgbd_config as config

def main():
    print("=" * 60)
    print(" Stretch 4 Emulated RGB-D API Capabilities Demonstration")
    print("=" * 60)

    # 1. Initialize the RGB-D Stream
    # We use the left fisheye camera and the left LiDAR.
    print("\n[1] Initializing Stream...")
    streamer, generator = get_emulated_rgbd_stream(
        use_left=True, 
        use_left_lidar=True,
        emulated_rgbd_fps=10.0,
        camera_fps=10.0
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
        for frame in generator:
            if frame is None:
                continue
                
            # 4. Access Lazy Properties
            rgb_image = frame.image
            depth_image = frame.depth_image
            
            # 5. Access Calibration Data
            cam_matrix = frame.camera_matrix
            dist_coeffs = frame.distortion_coefficients
            T_base_to_cam = frame.T_base_to_cam
            
            # 6. Apply Validity Masks
            vig_mask, lidar_mask = mask_manager.get_masks("left", "left_lidar", rgb_image.shape)
            
            if False:
                # Combine the masks and ERODE them slightly. 
                # Eroding shrinks the mask inward by a few pixels. This elegantly drops the extreme 
                # boundary pixels of the fisheye lens BEFORE unprojection, preventing the tan(theta) explosion.
                combined_mask = vig_mask & lidar_mask
                erosion_kernel = np.ones((5, 5), np.uint8)
                dense_depth_validity_mask = cv2.erode(combined_mask.astype(np.uint8), erosion_kernel, iterations=12).astype(bool)
            else: 
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
                # Normalize depth for visualization (cap at 5 meters for better contrast)
                max_depth = 5.0
                depth_vis = np.clip(dense_depth, 0, max_depth) / max_depth
                depth_vis = (depth_vis * 255).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                # Set invalid depth (0) to black
                depth_colormap[dense_depth == 0] = [0, 0, 0]
                
                cv2.imshow("Masked RGB", masked_rgb)
                cv2.imshow("Dense Depth", depth_colormap)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # 27 is ESC
                    rerun_mode = True
                    cv2.destroyAllWindows()
                    print("\n[9b] Switching to Rerun Visualization (press Ctrl+C to exit)...")
            else:
                # 9b. Rerun Visualization
                if not rerun_initialized:
                    try:
                        import rerun as rr
                        import rerun.blueprint as rrb
                        print("    -> Spawning Rerun viewer...")
                        rr.init("api_example_visualization", spawn=True)
                        
                        # Explicitly define the layout blueprint to force stacking of the 2D images
                        # under the 'camera' origin, while keeping the 3D views separate.
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
                        
                # Log the images (OpenCV uses BGR, Rerun expects RGB)
                rr.log("camera/rgb", rr.Image(masked_rgb[:, :, ::-1])) 
                rr.log("camera/dense_depth", rr.DepthImage(dense_depth, meter=1.0, depth_range=[0.0, config.RERUN_COLOR_MAX_DEPTH_M]))
                
                # Add the sparse depth image overlay as requested
                rr.log("camera/sparse_depth", rr.DepthImage(depth_image, meter=1.0, depth_range=[0.0, config.RERUN_COLOR_MAX_DEPTH_M]))
                
                # Log the perfectly aligned SPARSE point cloud directly from the API.
                # Logged to a separate root path ("sparse_view") to force a side-by-side window in Rerun
                rr.log(
                    "sparse_view/point_cloud", 
                    rr.Points3D(frame.point_cloud_base, colors=frame.point_colors, radii=[0.01]) 
                )
                
                # Log the perfectly aligned DENSE point cloud generated from the depth map.
                # Logged to a separate root path ("dense_view") to force a side-by-side window in Rerun
                rr.log(
                    "dense_view/point_cloud", 
                    rr.Points3D(pts_base, colors=colors[:, ::-1], radii=[0.01]) 
                )
        
    finally:
        print("\nStopping streamer...")
        streamer.stop()

if __name__ == "__main__":
    main()

import os
import yaml
import numpy as np
from stretch4_emulated_rgbd.shared_utils import ImageFrame, RGBDFrame, project_points, apply_shadow_filter, ExtrinsicsCalibration
from stretch4_emulated_rgbd import emulated_rgbd_config as config

from stretch4_emulated_rgbd.head_camera import HeadCamera
from stretch4_emulated_rgbd.lidar_poller import LidarPoller

class FastEmulatedRGBDStreamer:
    """
    A low-latency, temporally synchronized emulated RGB-D streamer.
    
    FRAME RATE & PERFORMANCE OPTIMIZATIONS:
    Achieving a consistent 10Hz output required completely bypassing the heavier `stretch4_body` 
    pipeline which introduces overhead through multiple message passing and synchronization layers.
    
    1. Direct Hardware Access: This class pulls raw, temporally-stamped frames directly from the 
       non-blocking queues of `HeadCamera` and `LidarPoller`.
    2. Zero-Copy Distorted Projection: Instead of running an expensive dense image unwarp 
       (`cv2.undistort`) on the 10Hz RGB stream to map it to a rectilinear depth image, this pipeline 
       does the inverse: it projects the 3D LiDAR points directly into the raw, distorted fisheye 
       image space. This turns a heavy O(N_pixels) operation into a lightweight O(N_lidar_points) operation.
    3. Tight Generator Loop: The data is yielded iteratively via a generator, avoiding memory 
       bloat and garbage collection pauses.
    4. Software Over-Sampling & LiDAR-Driven Architecture: The loop is strictly driven by the 10Hz 
       mechanical sweep of the LiDAR. When a sweep finishes, the streamer fetches the mathematically 
       optimal frame from the camera's high-frequency (e.g. 30Hz) history buffer. This drops phase 
       misalignment error to ~16ms.
    """
    def __init__(self, emulated_rgbd_fps=10, camera_fps=30, resolution_height=800, compress=True, oak_buffer_size=1, calibration: ExtrinsicsCalibration = None, ignore_prior_optimizations=False):
        self.emulated_rgbd_fps = emulated_rgbd_fps
        self.fleet_path = os.environ.get("HELLO_FLEET_PATH", "")
        self.fleet_id = os.environ.get("HELLO_FLEET_ID", "")
        
        if not self.fleet_path or not self.fleet_id:
            raise RuntimeError("HELLO_FLEET_PATH or HELLO_FLEET_ID environment variables are missing.")

        # Load LiDAR calibration
        from stretch4_body.subsystem.cameras.calibrate_extrinsics_lidars import DualLidarCalibration
        self.lidar_calib = DualLidarCalibration()
        self.T_lidar_to_base_left = self.lidar_calib.get_lidar_to_base_transform(is_right_lidar=False)

        # Load camera extrinsics
        camera_extrinsics_path = os.path.join(
            self.fleet_path, self.fleet_id, "calibration_cameras", "camera_extrinsics.yaml"
        )
        self.camera_extrinsics = {}
        if os.path.exists(camera_extrinsics_path):
            with open(camera_extrinsics_path, "r") as f:
                self.camera_extrinsics = yaml.safe_load(f) or {}

        # The head cameras are calibrated relative to the center camera
        self.T_left_to_center = np.array(self.camera_extrinsics.get("left_to_center", np.eye(4)))
        
        # Determine the center camera's position relative to the base using the right LiDAR (this is the factory convention)
        self.T_base_to_center = np.eye(4)
        key = "transform_right_lidar_to_head_center"
        try:
            T_l_to_c = np.array(self.camera_extrinsics[key]["data"])
        except KeyError as e:
            print(f"Key {key} not found in camera_extrinsics.yaml")
            print(f"Please run REx_camera_calibrate.")
            raise e
        T_base_to_right_lidar = self.lidar_calib.get_lidar_to_base_transform(is_right_lidar=True)
        self.T_base_to_center = T_l_to_c @ np.linalg.inv(T_base_to_right_lidar)

        self.T_base_to_cam = {
            "left": np.linalg.inv(self.T_left_to_center) @ self.T_base_to_center,
        }
        
        # Save a copy of the factory baseline calibration
        self.T_base_to_cam_factory = {
            "left": self.T_base_to_cam["left"].copy()
        }
        
        self.T_base_to_cam_optimized = None

        # Apply provided or default optimized calibration
        if calibration is None and not ignore_prior_optimizations:
            default_calib_path = os.path.join(self.fleet_path, self.fleet_id, "calibration_cameras", "emulated_rgbd_extrinsics_left_camera_left_lidar.yaml")
            if os.path.exists(default_calib_path):
                print(f"Loading automatic optimized Emulated RGB-D calibration from {default_calib_path}")
                calibration = ExtrinsicsCalibration.load_from_yaml(default_calib_path)
                
        if calibration is not None:
            self.T_base_to_cam["left"] = calibration.apply_to_camera_extrinsics(self.T_base_to_cam["left"])
            self.T_base_to_cam_optimized = {
                "left": self.T_base_to_cam["left"].copy()
            }



        # Create calibs property equivalent to EmulatedRGBDStreamer for easy drop-in compatibility
        self.calibs = {"left": self}
        self.latest_lidar_pts = {}

        # Initialize the hardware wrappers
        self.camera = HeadCamera(fps=camera_fps, resolution_height=resolution_height, compress=compress, oak_buffer_size=oak_buffer_size)
        self.lidar_poller = LidarPoller()
        
        self.camera.start()
        self.camera_matrix, self.distortion_coefficients = self.camera.get_intrinsics()
        
        # Fisheye camera model is default for left/right head cameras
        self.is_fisheye = True

    def stream_left_rgbd(self):
        """Generator that yields low-latency, synchronized left RGBD frames."""
        try:
            target_interval = 1.0 / self.emulated_rgbd_fps
            last_yield_time = 0
            last_mid_ts = None
            
            while True:
                # LiDAR drives the loop
                mid_ts, end_ts, lidar_frame = self.lidar_poller.wait_for_next_frame(last_mid_ts)
                if mid_ts is None:
                    continue
                last_mid_ts = mid_ts
                
                if lidar_frame is not None:
                    self.latest_lidar_pts["left"] = lidar_frame.points

                
                # Enforce output frame rate restraint (e.g. drop frames to hit 5Hz)
                if mid_ts - last_yield_time < target_interval - 0.01: # 10ms tolerance
                    continue
                last_yield_time = mid_ts

                # Fetch the optimal over-sampled RGB frame
                rgb_frame = self.camera.get_closest_frame(mid_ts)
                if rgb_frame is None or rgb_frame[0] is None:
                    continue
                    
                rgb_img, rgb_timestamp, rgb_seq, img_data = rgb_frame
                
                # Setup the base return struct
                image_frame = ImageFrame(image_raw=img_data if img_data is not None else rgb_img, image=rgb_img, timestamp=rgb_timestamp, frame_number=rgb_seq)
                
                if lidar_frame is None or self.camera_matrix is None:
                    yield RGBDFrame(
                        timestamp=rgb_timestamp,
                        image_frame=image_frame,
                        camera_type="left",
                        point_cloud=np.zeros((0, 3)),
                        point_cloud_base=np.zeros((0, 3)),
                        point_colors=np.zeros((0, 3)),
                        depth_image=np.zeros(rgb_img.shape[:2], dtype=np.float32),
                        robot_id=self.fleet_id,
                        timestamp_image=rgb_timestamp,
                        timestamp_lidar_left=lidar_frame.timestamp if lidar_frame else None,
                        timestamp_lidar_right=None,
                        lidars_used="left_lidar"
                    )
                    continue

                # Process point cloud from Left LiDAR (LiDAR Frame -> Base Frame)
                ones = np.ones((len(lidar_frame.points), 1))
                pts_base = (self.T_lidar_to_base_left @ np.hstack([lidar_frame.points, ones]).T).T[:, :3]

                # Transform to Camera Frame
                T_base_to_left_cam = self.T_base_to_cam["left"]
                pts_cam_all = (T_base_to_left_cam @ np.hstack([pts_base, ones]).T).T[:, :3]

                # Filter points behind camera
                valid_idx = pts_cam_all[:, 2] > 0
                pts_cam_valid = pts_cam_all[valid_idx]
                pts_base_valid = pts_base[valid_idx]

                depth_img = np.zeros(rgb_img.shape[:2], dtype=np.float32)

                if len(pts_cam_valid) > 0:
                    rvec, tvec = np.zeros(3), np.zeros(3)
                    img_pts = project_points(
                        pts_cam_valid, rvec, tvec, self.camera_matrix, self.distortion_coefficients, camera_model="fisheye"
                    ).reshape(-1, 2)

                    h, w = rgb_img.shape[:2]
                    img_pts_int = np.round(img_pts).astype(int)
                    u = img_pts_int[:, 0]
                    v = img_pts_int[:, 1]

                    valid_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)

                    u_valid = u[valid_uv]
                    v_valid = v[valid_uv]

                    if len(v_valid) > 0:
                        z_vals = pts_cam_valid[valid_uv, 2]
                        # Z-buffer sorting
                        sort_idx = np.argsort(z_vals)[::-1]
                        v_sorted = v_valid[sort_idx]
                        u_sorted = u_valid[sort_idx]
                        z_sorted = z_vals[sort_idx]
                        
                        # Track which points survive the Z-buffer and shadow filter
                        index_img = np.full(rgb_img.shape[:2], -1, dtype=int)
                        orig_idx_sorted = np.arange(len(z_vals))[sort_idx]
                        
                        depth_img[v_sorted, u_sorted] = z_sorted
                        index_img[v_sorted, u_sorted] = orig_idx_sorted
                        
                        if config.ENABLE_SHADOW_FILTER:
                            depth_img = apply_shadow_filter(
                                depth_img,
                                window_size=config.SHADOW_FILTER_WINDOW_SIZE,
                                depth_threshold=config.SHADOW_FILTER_DEPTH_THRESHOLD_M
                            )
                            
                        valid_mask = depth_img > 0
                        surviving_indices = index_img[valid_mask]
                        
                        pts_cam = pts_cam_valid[valid_uv][surviving_indices]
                        pts_world = pts_base_valid[valid_uv][surviving_indices]
                        
                        v_filtered, u_filtered = np.where(valid_mask)
                        colors_bgr = rgb_img[v_filtered, u_filtered]
                        cols = colors_bgr[:, ::-1]  # BGR to RGB

                else:
                    pts_cam = np.zeros((0, 3))
                    pts_world = np.zeros((0, 3))
                    cols = np.zeros((0, 3))

                yield RGBDFrame(
                    timestamp=rgb_timestamp,
                    image_frame=image_frame,
                    camera_type="left",
                    point_cloud=pts_cam,
                    point_cloud_base=pts_world,
                    point_colors=cols,
                    depth_image=depth_img,
                    camera_matrix=self.camera_matrix,
                    distortion_coefficients=self.distortion_coefficients,
                    T_base_to_cam=self.T_base_to_cam["left"],
                    T_lidar_to_base_left=self.T_lidar_to_base_left,
                    T_lidar_to_base_right=None,
                    robot_id=self.fleet_id,
                    timestamp_image=rgb_timestamp,
                    timestamp_lidar_left=lidar_frame.timestamp if lidar_frame else None,
                    timestamp_lidar_right=None,
                    lidars_used="left_lidar"
                )
        except GeneratorExit:
            pass
        except Exception as e:
            print(f"Streamer encountered an error: {e}")

    def stop(self):
        """Stops the camera and LiDAR poller."""
        self.camera.stop()
        self.lidar_poller.stop()

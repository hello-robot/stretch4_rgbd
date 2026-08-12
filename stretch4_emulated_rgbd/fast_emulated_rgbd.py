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
    def __init__(self, camera="left", lidar="left", emulated_rgbd_fps=10, camera_fps=30, resolution_height=800, compress=True, oak_buffer_size=1, calibration: ExtrinsicsCalibration = None, ignore_prior_optimizations=False, merge_lidars=False):
        if isinstance(camera, str):
            self.camera_names = [camera]
        else:
            self.camera_names = camera

        if isinstance(lidar, str):
            self.lidar_names = [lidar]
        else:
            self.lidar_names = lidar

        self.merge_lidars = merge_lidars


        self.emulated_rgbd_fps = emulated_rgbd_fps
        self.fleet_path = os.environ.get("HELLO_FLEET_PATH", "")
        self.fleet_id = os.environ.get("HELLO_FLEET_ID", "")
        
        if not self.fleet_path or not self.fleet_id:
            raise RuntimeError("HELLO_FLEET_PATH or HELLO_FLEET_ID environment variables are missing.")

        # Load LiDAR calibration
        from stretch4_body.subsystem.cameras.calibrate_extrinsics_lidars import DualLidarCalibration
        self.lidar_calib = DualLidarCalibration()
        self.T_lidar_to_base_left = self.lidar_calib.get_lidar_to_base_transform(is_right_lidar=False)
        self.T_lidar_to_base_right = self.lidar_calib.get_lidar_to_base_transform(is_right_lidar=True)

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
        self.T_right_to_center = np.array(self.camera_extrinsics.get("right_to_center", np.eye(4)))
        
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
            "right": np.linalg.inv(self.T_right_to_center) @ self.T_base_to_center,
        }
        
        # Save a copy of the factory baseline calibration
        self.T_base_to_cam_factory = {
            "left": self.T_base_to_cam["left"].copy(),
            "right": self.T_base_to_cam["right"].copy(),
        }
        
        self.T_base_to_cam_optimized = {}

        # Apply provided or default optimized calibration for each camera
        for c_name in self.camera_names:
            cam_calib = calibration
            if cam_calib is None and not ignore_prior_optimizations:
                # Determine which lidar was likely used for this camera's optimization
                # Default to the same side lidar if available, otherwise 'both'
                l_name = c_name if c_name in self.lidar_names else (self.lidar_names[0] if self.lidar_names else "left")
                if len(self.lidar_names) > 1:
                    l_name = "both"

                default_calib_path = os.path.join(
                    self.fleet_path,
                    self.fleet_id,
                    "calibration_cameras",
                    f"emulated_rgbd_extrinsics_{c_name}_camera_{l_name}_lidar.yaml"
                )
                if os.path.exists(default_calib_path):
                    print(f"Loading automatic optimized Emulated RGB-D calibration for {c_name} camera from {default_calib_path}")
                    cam_calib = ExtrinsicsCalibration.load_from_yaml(default_calib_path)
                    
            if cam_calib is not None:
                self.T_base_to_cam[c_name] = cam_calib.apply_to_camera_extrinsics(self.T_base_to_cam_factory[c_name])
                self.T_base_to_cam_optimized[c_name] = self.T_base_to_cam[c_name].copy()

        # Create calibs property equivalent to EmulatedRGBDStreamer for easy drop-in compatibility
        self.calibs = {name: self for name in self.camera_names}
        self.latest_lidar_pts = {}

        # Initialize the hardware wrappers
        self.camera = HeadCamera(camera_name=self.camera_names, fps=camera_fps, resolution_height=resolution_height, compress=compress, oak_buffer_size=oak_buffer_size)
        
        self.lidar_pollers = {}
        for l_name in self.lidar_names:
            self.lidar_pollers[l_name] = LidarPoller(lidar_name=l_name)
        
        self.camera.start()
        
        self.camera_matrices = {}
        self.distortion_coefficients = {}
        for c_name in self.camera_names:
            M, D = self.camera.get_intrinsics(c_name)
            self.camera_matrices[c_name] = M
            self.distortion_coefficients[c_name] = D
        
        # Fisheye camera model is default for left/right head cameras
        self.is_fisheye = True

        # For backward compatibility properties
        self.camera_matrix = self.camera_matrices.get(self.camera_names[0])
        self.distortion_coeffs = self.distortion_coefficients.get(self.camera_names[0])
        self.camera_name = self.camera_names[0]
        self.lidar_name = self.lidar_names[0] if self.lidar_names else "none"


    def stream_left_rgbd(self):
        """Legacy wrapper that yields left RGBD frames (uses stream_rgbd)."""
        return self.stream_rgbd()

    def stream_rgbd(self):
        """Generator that yields low-latency, synchronized RGBD frames."""
        from stretch4_emulated_rgbd.shared_utils import MultiRGBDFrame, merge_lidar_points
        try:
            target_interval = 1.0 / self.emulated_rgbd_fps
            last_yield_time = 0
            last_mid_ts = None
            
            # Use the first lidar as the master timing source
            master_lidar_name = self.lidar_names[0] if self.lidar_pollers else None
            master_poller = self.lidar_pollers.get(master_lidar_name)

            while True:
                if master_poller:
                    # LiDAR drives the loop
                    mid_ts, end_ts, master_lidar_frame = master_poller.wait_for_next_frame(last_mid_ts)
                    if mid_ts is None:
                        continue
                    last_mid_ts = mid_ts
                else:
                    # Fallback if no lidar is available (rare)
                    time.sleep(target_interval)
                    mid_ts = time.monotonic()
                
                # Enforce output frame rate restraint
                if mid_ts - last_yield_time < target_interval - 0.01: # 10ms tolerance
                    continue
                last_yield_time = mid_ts

                # Synchronize all pollers and cameras to this mid_ts
                synced_lidar_frames = {}
                for l_name, poller in self.lidar_pollers.items():
                    if l_name == master_lidar_name:
                        synced_lidar_frames[l_name] = master_lidar_frame
                    else:
                        synced_lidar_frames[l_name] = poller.get_closest_frame(mid_ts)
                    
                    if synced_lidar_frames[l_name]:
                        self.latest_lidar_pts[l_name] = synced_lidar_frames[l_name].points

                # Pre-calculate point clouds based on requested merging behavior
                from stretch4_emulated_rgbd.shared_utils import merge_lidar_points
                
                l_frame = synced_lidar_frames.get("left")
                r_frame = synced_lidar_frames.get("right")
                
                # Global merge for all cameras if requested
                merged_pts_base_global = None
                if self.merge_lidars:
                    merged_pts_base_global = merge_lidar_points(
                        l_frame.points if l_frame else None,
                        r_frame.points if r_frame else None,
                        self.T_lidar_to_base_left,
                        self.T_lidar_to_base_right
                    )

                # Process RGB-D for each camera
                rgbd_frames = {}
                for c_name in self.camera_names:
                    # Decide which points to use for this specific camera
                    if self.merge_lidars:
                        pts_to_use = merged_pts_base_global
                        active_lidars = list(synced_lidar_frames.keys())
                        if len(active_lidars) == 2:
                            lidars_for_cam = "both_lidar"
                        elif len(active_lidars) == 1:
                            lidars_for_cam = f"{active_lidars[0]}_lidar"
                        else:
                            lidars_for_cam = "no_lidar"
                    else:
                        # Default: Associate left camera with left lidar, right with right.
                        # Do not fall back to the other side if the preferred lidar is missing.
                        pref_l_frame = l_frame if c_name == "left" else None
                        pref_r_frame = r_frame if c_name == "right" else None
                        
                        pts_to_use = merge_lidar_points(
                            pref_l_frame.points if pref_l_frame else None,
                            pref_r_frame.points if pref_r_frame else None,
                            self.T_lidar_to_base_left,
                            self.T_lidar_to_base_right
                        )
                        
                        # Set actual lidar used for this camera
                        active_lidars = []
                        if pref_l_frame: active_lidars.append("left")
                        if pref_r_frame: active_lidars.append("right")
                        
                        if len(active_lidars) == 2:
                            lidars_for_cam = "both_lidar"
                        elif len(active_lidars) == 1:
                            lidars_for_cam = f"{active_lidars[0]}_lidar"
                        else:
                            lidars_for_cam = "no_lidar"
                    
                    rgbd_frames[c_name] = self._get_single_rgbd_frame(c_name, mid_ts, synced_lidar_frames, pts_to_use, lidars_for_cam)

                if len(self.camera_names) == 1:
                    yield rgbd_frames[self.camera_names[0]]
                else:
                    yield MultiRGBDFrame(
                        left=rgbd_frames.get("left"),
                        right=rgbd_frames.get("right"),
                        center=rgbd_frames.get("center"),
                        timestamp=mid_ts
                    )



        except GeneratorExit:
            pass
        except Exception as e:
            import traceback
            print(f"Streamer encountered an error: {e}")
            traceback.print_exc()

    def _get_single_rgbd_frame(self, camera_name, mid_ts, synced_lidar_frames, merged_pts_base, lidars_used=None):
        """Internal helper to process a single camera's RGBD frame."""
        rgb_frame = self.camera.get_closest_frame(mid_ts, camera_name=camera_name)
        if rgb_frame is None:
            return None
            
        rgb_img_maybe, rgb_timestamp, rgb_seq, img_data = rgb_frame
        
        # Decompress here on demand if using MJPEG
        rgb_img = rgb_img_maybe
        if rgb_img is None and img_data is not None:
            import cv2
            rgb_img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        if rgb_img is None:
            return None

        image_frame = ImageFrame(image_raw=img_data, image=rgb_img, timestamp=rgb_timestamp, frame_number=rgb_seq)
        
        l_pts = synced_lidar_frames.get("left")
        r_pts = synced_lidar_frames.get("right")
        
        camera_matrix = self.camera_matrices.get(camera_name)
        dist_coeffs = self.distortion_coefficients.get(camera_name)

        if len(merged_pts_base) == 0 or camera_matrix is None:
            return RGBDFrame(
                timestamp=rgb_timestamp,
                image_frame=image_frame,
                camera_type=camera_name,
                point_cloud=np.zeros((0, 3)),
                point_cloud_base=np.zeros((0, 3)),
                point_colors=np.zeros((0, 3)),
                depth_image=np.zeros(rgb_img.shape[:2], dtype=np.float32),
                robot_id=self.fleet_id,
                timestamp_image=rgb_timestamp,
                timestamp_lidar_left=l_pts.timestamp if l_pts else None,
                timestamp_lidar_right=r_pts.timestamp if r_pts else None,
                lidars_used=lidars_used if lidars_used is not None else " ".join([f"{l}_lidar" for l in synced_lidar_frames.keys()])
            )

        # Transform to Camera Frame
        T_base_to_cam = self.T_base_to_cam[camera_name]
        
        # Fast transformation without hstack
        pts_cam_all = merged_pts_base @ T_base_to_cam[:3, :3].T + T_base_to_cam[:3, 3]

        # Filter points behind camera
        valid_idx = pts_cam_all[:, 2] > 0
        pts_cam_valid = pts_cam_all[valid_idx]
        pts_base_valid = merged_pts_base[valid_idx]

        depth_img = np.zeros(rgb_img.shape[:2], dtype=np.float32)
        pts_cam = np.zeros((0, 3))
        pts_world = np.zeros((0, 3))
        cols = np.zeros((0, 3))

        if len(pts_cam_valid) > 0:
            rvec, tvec = np.zeros(3), np.zeros(3)
            img_pts = project_points(
                pts_cam_valid, rvec, tvec, camera_matrix, dist_coeffs, camera_model="fisheye"
            ).reshape(-1, 2)

            h, w = rgb_img.shape[:2]
            img_pts_int = np.round(img_pts).astype(int)
            u = img_pts_int[:, 0]
            v = img_pts_int[:, 1]

            # Fast FOV filtering
            valid_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            
            if np.any(valid_uv):
                u_valid = u[valid_uv]
                v_valid = v[valid_uv]
                z_vals = pts_cam_valid[valid_uv, 2]

                # Sort by depth to handle occlusions (further points are overwritten by closer ones)
                # We sort descending so that closer points are written last
                sort_idx = np.argsort(z_vals)[::-1]
                v_sorted = v_valid[sort_idx]
                u_sorted = u_valid[sort_idx]
                z_sorted = z_vals[sort_idx]
                
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

        return RGBDFrame(
            timestamp=rgb_timestamp,
            image_frame=image_frame,
            camera_type=camera_name,
            point_cloud=pts_cam,
            point_cloud_base=pts_world,
            point_colors=cols,
            depth_image=depth_img,
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            T_base_to_cam=T_base_to_cam,
            T_lidar_to_base_left=self.T_lidar_to_base_left if l_pts else None,
            T_lidar_to_base_right=self.T_lidar_to_base_right if r_pts else None,
            robot_id=self.fleet_id,
            timestamp_image=rgb_timestamp,
            timestamp_lidar_left=l_pts.timestamp if l_pts else None,
            timestamp_lidar_right=r_pts.timestamp if r_pts else None,
            lidars_used=lidars_used if lidars_used is not None else " ".join([f"{l}_lidar" for l in synced_lidar_frames.keys()])
        )


    def stop(self):
        """Stops the camera and all LiDAR pollers."""
        self.camera.stop()
        for poller in self.lidar_pollers.values():
            poller.stop()


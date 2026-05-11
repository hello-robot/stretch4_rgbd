import warnings
import logging

# Suppress urdf_parser_py "Unknown attribute" warnings that occur due to 
# collision geometry names in the Stretch URDF.
try:
    import urdf_parser_py.xml_reflection.core
    def _silent_on_error(message):
        pass
    urdf_parser_py.xml_reflection.core.on_error = _silent_on_error
except ImportError:
    pass

from stretch4_emulated_rgbd import emulated_rgbd_config as config

try:
    from stretch_body_ii.subsystem.cameras.emulated_rgbd import (
        stream_left_rgbd,
        stream_right_rgbd,
        stream_center_rgbd,
        stream_left_right_rgbd,
        stream_left_right_center_rgbd,
        EmulatedRGBDStreamer,
    )
    HAS_STRETCH_BODY = config.USE_STRETCH_BODY_EMULATED_RGBD
except ImportError:
    HAS_STRETCH_BODY = False

from stretch4_emulated_rgbd.shared_utils import (
    ExtrinsicsCalibration,
    DenseDepthImage,
    ValidityMaskManager,
    render_rgbd,
    unproject_points,
    RGBDFrame,
    get_rotated_intrinsics,
    get_rotated_extrinsics
)
import numpy as np
import cv2

__all__ = [
    "get_emulated_rgbd_stream",
    "get_camera_intrinsics",
    "get_camera_extrinsics",
    "get_pixel_3d_location",
    "create_point_cloud_from_depth",
    "visualize_rgbd_frame",
    "project_base_link_points_to_image",
    "ExtrinsicsCalibration",
    "DenseDepthImage",
    "ValidityMaskManager",
    "RGBDFrame"
]

def get_emulated_rgbd_stream(
    use_left=True,
    use_right=False,
    use_center=False,
    use_left_right=False,
    use_left_right_center=False,
    use_left_lidar=True,
    use_right_lidar=False,
    emulated_rgbd_fps=10.0,
    camera_fps=30,
    resolution_height=800,
    compress=True,
    oak_buffer_size=1,
    calibration: ExtrinsicsCalibration = None,
    ignore_prior_optimizations=False
):
    """
    Unified entry point to stream synchronized RGB-D frames from Stretch 4.
    Automatically selects the FastEmulatedRGBDStreamer if only the left camera and 
    left LiDAR are requested. Otherwise falls back to the standard stretch_body_ii streamer.
    
    Returns:
        (streamer, generator): The streamer instance and the generator yielding RGBDFrames.
    """
    
    use_both_lidars_default = not (use_left_lidar or use_right_lidar)
    if use_both_lidars_default:
        use_left_lidar = True
        use_right_lidar = True

    # Use fast streamer for the optimal case
    if use_left and not use_right and not use_center and not use_left_right and not use_left_right_center and use_left_lidar and not use_right_lidar:
        from stretch4_emulated_rgbd.fast_emulated_rgbd import FastEmulatedRGBDStreamer
        streamer = FastEmulatedRGBDStreamer(
            emulated_rgbd_fps=emulated_rgbd_fps, 
            camera_fps=camera_fps,
            resolution_height=resolution_height, 
            compress=compress, 
            oak_buffer_size=oak_buffer_size,
            calibration=calibration,
            ignore_prior_optimizations=ignore_prior_optimizations
        )
        return streamer, streamer.stream_left_rgbd()

    if not HAS_STRETCH_BODY:
        raise RuntimeError("stretch_body_ii is not installed. Live streaming fallback is not available. Try selecting only left camera and left lidar.")

    streamer = EmulatedRGBDStreamer.get_instance(use_left_lidar=use_left_lidar, use_right_lidar=use_right_lidar)
    
    # Store the factory calibration
    if not hasattr(streamer, 'T_base_to_cam_factory'):
        streamer.T_base_to_cam_factory = {
            cam: T.copy() for cam, T in streamer.T_base_to_cam.items()
        }
    streamer.T_base_to_cam_optimized = None

    # Apply specific calibration if provided
    if calibration is not None:
        for cam in streamer.T_base_to_cam:
            streamer.T_base_to_cam[cam] = calibration.apply_to_camera_extrinsics(streamer.T_base_to_cam_factory[cam])
            if streamer.T_base_to_cam_optimized is None:
                streamer.T_base_to_cam_optimized = {}
            streamer.T_base_to_cam_optimized[cam] = streamer.T_base_to_cam[cam].copy()
    elif not ignore_prior_optimizations:
        # Load any existing calibrations automatically for the fallback streamer
        import os
        fleet_path = os.environ.get("HELLO_FLEET_PATH", "")
        fleet_id = os.environ.get("HELLO_FLEET_ID", "")
        
        for cam in streamer.T_base_to_cam:
            lidar_str = "left_lidar" if use_left_lidar else "right_lidar"
            if use_left_lidar and use_right_lidar:
                lidar_str = "both_lidar"
            
            calib_path = os.path.join(fleet_path, fleet_id, "calibration_cameras", f"emulated_rgbd_extrinsics_{cam}_camera_{lidar_str}.yaml")
            if os.path.exists(calib_path):
                print(f"Loading automatic optimized Emulated RGB-D calibration from {calib_path}")
                calib = ExtrinsicsCalibration.load_from_yaml(calib_path)
                if calib is not None:
                    streamer.T_base_to_cam[cam] = calib.apply_to_camera_extrinsics(streamer.T_base_to_cam_factory[cam])
                    if streamer.T_base_to_cam_optimized is None:
                        streamer.T_base_to_cam_optimized = {}
                    streamer.T_base_to_cam_optimized[cam] = streamer.T_base_to_cam[cam].copy()
    else:
        # If ignoring prior optimizations, reset to factory
        for cam in streamer.T_base_to_cam:
            streamer.T_base_to_cam[cam] = streamer.T_base_to_cam_factory[cam].copy()

    if use_left_right:
        gen = stream_left_right_rgbd(is_rotate=False, use_left_lidar=use_left_lidar, use_right_lidar=use_right_lidar)
    elif use_left:
        gen = stream_left_rgbd(is_rotate=False, use_left_lidar=use_left_lidar, use_right_lidar=use_right_lidar)
    elif use_right:
        gen = stream_right_rgbd(is_rotate=False, use_left_lidar=use_left_lidar, use_right_lidar=use_right_lidar)
    elif use_center:
        gen = stream_center_rgbd(is_rotate=False, use_left_lidar=use_left_lidar, use_right_lidar=use_right_lidar)
    else:
        gen = stream_left_right_center_rgbd(is_rotate=False, use_left_lidar=use_left_lidar, use_right_lidar=use_right_lidar)

    def _inject_transforms(gen, streamer):
        def _process_frame(f, c_name):
            if streamer.calibs.get(c_name):
                M = streamer.calibs[c_name].camera_matrix
                D = streamer.calibs[c_name].distortion_coefficients
            else:
                M = None
                D = None
            T = streamer.T_base_to_cam.get(c_name)
            

                    
            f.camera_matrix = M
            f.distortion_coefficients = D
            f.T_base_to_cam = T
            
            if hasattr(streamer, 'lidar_calib') and streamer.lidar_calib:
                f.T_lidar_to_base_left = streamer.lidar_calib.get_lidar_to_base_transform(is_right_lidar=False)
                f.T_lidar_to_base_right = streamer.lidar_calib.get_lidar_to_base_transform(is_right_lidar=True)
            import os
            f.robot_id = os.environ.get("HELLO_FLEET_ID", "UNKNOWN")
            
            lidar_str = "no_lidar"
            if use_left_lidar and use_right_lidar:
                lidar_str = "both_lidar"
            elif use_left_lidar:
                lidar_str = "left_lidar"
            elif use_right_lidar:
                lidar_str = "right_lidar"
            f.lidars_used = lidar_str
            
            if getattr(f, "timestamp_image", None) is None: 
                f.timestamp_image = f.timestamp
            if getattr(f, "timestamp_lidar_left", None) is None:
                f.timestamp_lidar_left = None
            if getattr(f, "timestamp_lidar_right", None) is None:
                f.timestamp_lidar_right = None

        for frame in gen:
            # If frame is a dictionary/namespace for multiple cameras:
            if hasattr(frame, 'camera_type'):
                _process_frame(frame, frame.camera_type)
                yield frame
            else:
                for c_name in ["left", "right", "center"]:
                    sub_frame = getattr(frame, c_name, None)
                    if sub_frame:
                        _process_frame(sub_frame, c_name)
                yield frame

    return streamer, _inject_transforms(gen, streamer)

def get_camera_intrinsics(streamer, camera_name: str):
    """
    Retrieves the camera intrinsic matrix and distortion coefficients from the streamer.
    
    Args:
        streamer: The Emulated RGB-D streamer instance.
        camera_name: Name of the camera (e.g., "left", "right", "center").
        
    Returns:
        tuple: (camera_matrix, distortion_coefficients)
    """
    calib = streamer.calibs.get(camera_name)
    if calib:
        return calib.camera_matrix, calib.distortion_coefficients
    return None, None

def get_camera_extrinsics(streamer, camera_name: str) -> np.ndarray:
    """
    Retrieves the optimized rigid body extrinsic transformation matrix (4x4).
    This transforms points from the robot's base frame into the camera's optical frame.
    
    Args:
        streamer: The Emulated RGB-D streamer instance.
        camera_name: Name of the camera (e.g., "left", "right", "center").
        
    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    return streamer.T_base_to_cam.get(camera_name)

def create_point_cloud_from_depth(depth_image: np.ndarray, rgb_image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    """
    Generates a dense, colored point cloud from an aligned depth image and RGB image.
    
    Args:
        depth_image: 2D NumPy array of depth values in meters (can be dense or sparse).
        rgb_image: 3D NumPy array of the corresponding RGB image.
        camera_matrix: 3x3 intrinsic camera matrix.
        dist_coeffs: Camera distortion coefficients.
        
    Returns:
        tuple: (pts_cam, colors) where pts_cam is an Nx3 array of 3D points in the camera frame, 
               and colors is an Nx3 array of corresponding RGB colors.
    """
    # Find all valid depth pixels (filtering out Z <= 0.01m for safety)
    v, u = np.where(depth_image > 0.01)
    if len(v) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
        
    z = depth_image[v, u]
    uv = np.vstack((u, v)).T
    
    # Unproject 2D pixels into 3D camera coordinates
    pts_cam = unproject_points(uv, z, camera_matrix, dist_coeffs, camera_model="fisheye")
    
    # Extract corresponding colors
    colors = rgb_image[v, u]
    # Ensure colors are RGB (if the input was BGR)
    # Most OpenCV workflows load BGR, but we assume the user might pass BGR or RGB.
    # To be safe, if we know we use BGR internally we might convert, but usually 
    # it's best to return the color format they passed in. We will return as is.
    
    return pts_cam, colors

def get_pixel_3d_location(u: int, v: int, depth_image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, T_base_to_cam: np.ndarray = None):
    """
    Returns the 3D location (X, Y, Z) of a specific pixel in the camera frame, 
    and optionally in the base frame if T_base_to_cam is provided.
    
    Args:
        u: The x-coordinate (column) of the pixel.
        v: The y-coordinate (row) of the pixel.
        depth_image: The aligned depth image (sparse or dense).
        camera_matrix: 3x3 intrinsic camera matrix.
        dist_coeffs: Camera distortion coefficients.
        T_base_to_cam: Optional 4x4 extrinsic matrix. If provided, the base frame coordinate is also returned.
        
    Returns:
        tuple: (pt_cam, pt_base) where pt_cam is a 3-element array [X, Y, Z] in the camera frame.
               pt_base is a 3-element array in the base frame if T_base_to_cam is provided, else None.
               Returns (None, None) if the depth at the pixel is 0 or invalid.
    """
    if v < 0 or v >= depth_image.shape[0] or u < 0 or u >= depth_image.shape[1]:
        return None, None
        
    z = depth_image[v, u]
    if z <= 0 or np.isinf(z) or np.isnan(z):
        return None, None
        
    uv = np.array([[u, v]], dtype=np.float32)
    z_arr = np.array([z], dtype=np.float32)
    
    pts_cam = unproject_points(uv, z_arr, camera_matrix, dist_coeffs, camera_model="fisheye")
    pt_cam = pts_cam[0]
    
    pt_base = None
    if T_base_to_cam is not None:
        pt_cam_homo = np.append(pt_cam, 1.0)
        pt_base_homo = np.linalg.inv(T_base_to_cam) @ pt_cam_homo
        pt_base = pt_base_homo[:3]
        
    return pt_cam, pt_base

def visualize_rgbd_frame(camera_name: str, frame: RGBDFrame, vig_mask: np.ndarray = None, depth_mask: np.ndarray = None):
    """
    Visualizes an RGBDFrame using ReRun, applying validity masks if provided.
    
    Args:
        camera_name: Name of the camera (e.g., "left").
        frame: The RGBDFrame yielded by the streamer.
        vig_mask: Optional RGB vignette mask.
        depth_mask: Optional depth valid mask.
    """
    render_rgbd(camera_name, frame, vig_mask=vig_mask, depth_mask=depth_mask)

def project_base_link_points_to_image(pts_3d_base: np.ndarray, T_base_to_cam: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    """
    Projects an array of 3D points from the robot's base frame into the 2D image plane of a camera.
    
    Args:
        pts_3d_base: Nx3 NumPy array of 3D points in the base frame.
        T_base_to_cam: 4x4 extrinsic transformation matrix from base to camera.
        camera_matrix: 3x3 intrinsic camera matrix.
        dist_coeffs: Camera distortion coefficients.
        
    Returns:
        tuple: (img_pts, valid_mask, pts_cam)
            - img_pts: Nx2 NumPy array of 2D pixel coordinates.
            - valid_mask: Boolean array of length N indicating which points are in front of the camera (Z > 0).
            - pts_cam: Nx3 NumPy array of the points in the camera frame.
    """
    if len(pts_3d_base) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool), np.zeros((0, 3))
        
    pts_3d_base_homo = np.hstack((pts_3d_base, np.ones((len(pts_3d_base), 1))))
    pts_cam = (T_base_to_cam @ pts_3d_base_homo.T).T[:, :3]
    
    valid_mask = pts_cam[:, 2] > 0.0
    
    pts_cam_safe = pts_cam.copy()
    pts_cam_safe[~valid_mask, 2] = 0.01
    
    pts_cam_np = pts_cam_safe.astype(np.float32).reshape(-1, 1, 3)
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    
    if len(dist_coeffs) == 4:
        img_pts, _ = cv2.fisheye.projectPoints(pts_cam_np, rvec, tvec, camera_matrix, dist_coeffs)
    else:
        img_pts, _ = cv2.projectPoints(pts_cam_np, rvec, tvec, camera_matrix, dist_coeffs)
        
    img_pts = img_pts.reshape(-1, 2)
    return img_pts, valid_mask, pts_cam

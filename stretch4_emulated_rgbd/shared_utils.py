import argparse
import select
import sys
import termios
import tty
import os

# Suppress annoying Qt font warnings from OpenCV
os.environ['QT_LOGGING_RULES'] = '*=false'

import cv2
import numpy as np
import rerun as rr
from dataclasses import dataclass
import time
import yaml
from pathlib import Path

from stretch4_emulated_rgbd import emulated_rgbd_config as config

def project_points(object_points, rvec, tvec, camera_matrix, distortion_coefficients, camera_model="pinhole", xi=0.0):
    if camera_model == "fisheye":
        object_points = object_points.reshape(-1, 1, 3)
        projected_points, _ = cv2.fisheye.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion_coefficients
        )
    elif camera_model == "omnidir":
        object_points = object_points.reshape(-1, 1, 3)
        projected_points, _ = cv2.omnidir.projectPoints(
            object_points, rvec, tvec, camera_matrix, xi, distortion_coefficients
        )
    else:
        projected_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion_coefficients
        )
    return projected_points

def unproject_points(uv, z, camera_matrix, distortion_coefficients, camera_model="pinhole", xi=0.0):
    if len(uv) == 0:
        return np.zeros((0, 3), dtype=np.float32)
        
    uv = uv.reshape(-1, 1, 2).astype(np.float32)
    
    if camera_model == "fisheye":
        normalized = cv2.fisheye.undistortPoints(
            uv, camera_matrix, distortion_coefficients
        )
    elif camera_model == "omnidir":
        xi_arr = np.array([xi], dtype=np.float32)
        normalized = cv2.omnidir.undistortPoints(
            uv, camera_matrix, distortion_coefficients, xi_arr, np.eye(3)
        )
    else:
        normalized = cv2.undistortPoints(
            uv, camera_matrix, distortion_coefficients
        )
        
    normalized = normalized.reshape(-1, 2)
    z = np.asarray(z).reshape(-1)
    
    pts_cam = np.zeros((len(z), 3), dtype=np.float32)
    pts_cam[:, 0] = normalized[:, 0] * z
    pts_cam[:, 1] = normalized[:, 1] * z
    pts_cam[:, 2] = z
    
    return pts_cam

def get_timestamp_from_name(name):
    """Extracts YYYYMMDD_HHMMSS from string."""
    name = Path(name).stem
    parts = name.split('_')
    for i in range(len(parts)-1):
        if len(parts[i]) == 8 and len(parts[i+1]) == 6 and parts[i].isdigit() and parts[i+1].isdigit():
            return int(parts[i] + parts[i+1])
    return 0

def find_latest_data_dir():
    """Finds the most recently collected data directory based on timestamp."""
    data_dir = Path("data")
    if not data_dir.exists():
        return None
    dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("captured_emulated_rgbd_")]
    if not dirs:
        return None
    dirs.sort(key=lambda d: get_timestamp_from_name(d.name))
    return str(dirs[-1])

def merge_lidar_points(l_pts, r_pts, T_lidar_to_base_left, T_lidar_to_base_right):
    """
    Transforms left and right LiDAR points to the base frame and merges them.
    
    Args:
        l_pts: NumPy array of left LiDAR points in the left LiDAR frame.
        r_pts: NumPy array of right LiDAR points in the right LiDAR frame.
        T_lidar_to_base_left: 4x4 rigid body transformation matrix from the left LiDAR frame to the robot base frame.
        T_lidar_to_base_right: 4x4 rigid body transformation matrix from the right LiDAR frame to the robot base frame.
        
    Returns:
        NumPy array of merged 3D points in the robot base frame.
    """
    merged_pts_base = []
    if l_pts is not None and len(l_pts) > 0:
        ones = np.ones((len(l_pts), 1))
        l_base = (T_lidar_to_base_left @ np.hstack([l_pts[:, :3], ones]).T).T[:, :3]
        merged_pts_base.append(l_base)
        
    if r_pts is not None and len(r_pts) > 0:
        ones = np.ones((len(r_pts), 1))
        r_base = (T_lidar_to_base_right @ np.hstack([r_pts[:, :3], ones]).T).T[:, :3]
        merged_pts_base.append(r_base)
        
    if merged_pts_base:
        return np.vstack(merged_pts_base)
    return np.zeros((0, 3))

class ImageFrame:
    def __init__(self, timestamp: float, frame_number: int, image: np.ndarray, image_raw: np.ndarray = None, image_rectified: np.ndarray = None, new_K: np.ndarray = None):
        self.timestamp = timestamp
        self.frame_number = frame_number
        self._image = image
        self.image_raw = image_raw
        self.image_rectified = image_rectified
        self.new_K = new_K

class RGBDFrame:
    def __init__(self, timestamp: float, image_frame: ImageFrame, camera_type: str, point_cloud: np.ndarray, point_cloud_base: np.ndarray, point_colors: np.ndarray, depth_image: np.ndarray, 
                 camera_matrix: np.ndarray = None, distortion_coefficients: np.ndarray = None, T_base_to_cam: np.ndarray = None, T_lidar_to_base_left: np.ndarray = None, T_lidar_to_base_right: np.ndarray = None, 
                 robot_id: str = None, timestamp_image: float = None, timestamp_lidar_left: float = None, timestamp_lidar_right: float = None, lidars_used: str = None):
        self.timestamp = timestamp
        self.image_frame = image_frame
        self.camera_type = camera_type
        self.point_cloud = point_cloud
        self.point_cloud_base = point_cloud_base
        self.point_colors = point_colors
        self._depth_image = depth_image
        self._camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self._T_base_to_cam = T_base_to_cam
        self.T_lidar_to_base_left = T_lidar_to_base_left
        self.T_lidar_to_base_right = T_lidar_to_base_right
        self.robot_id = robot_id
        self.timestamp_image = timestamp_image
        self.timestamp_lidar_left = timestamp_lidar_left
        self.timestamp_lidar_right = timestamp_lidar_right
        self.lidars_used = lidars_used
        
        self._is_upright = False

    def _rotate_image_ccw(self, img):
        import cv2
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    def _rotate_image_cw(self, img):
        import cv2
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    def _decompress_image(self):
        if self.image_frame._image is None and self.image_frame.image_raw is not None:
            import cv2
            img = cv2.imdecode(np.frombuffer(self.image_frame.image_raw, np.uint8), cv2.IMREAD_COLOR)
            import stretch4_emulated_rgbd.emulated_rgbd_config as config
            if self._is_upright and config.ROTATE_IMAGES_TO_VERTICAL and self.camera_type in ["left", "right"]:
                if self.camera_type == "left":
                    img = self._rotate_image_ccw(img)
                elif self.camera_type == "right":
                    img = self._rotate_image_cw(img)
            self.image_frame._image = img

    def _apply_rotation(self):
        if self._is_upright:
            return
            
        import stretch4_emulated_rgbd.emulated_rgbd_config as config
        if config.ROTATE_IMAGES_TO_VERTICAL and self.camera_type in ["left", "right"]:
            self._decompress_image()
            
            if self.image_frame._image is not None:
                if self.camera_type == "left":
                    self.image_frame._image = self._rotate_image_ccw(self.image_frame._image)
                elif self.camera_type == "right":
                    self.image_frame._image = self._rotate_image_cw(self.image_frame._image)
                    
            if self._depth_image is not None:
                if self.camera_type == "left":
                    self._depth_image = self._rotate_image_ccw(self._depth_image)
                elif self.camera_type == "right":
                    self._depth_image = self._rotate_image_cw(self._depth_image)
                    
            if self._camera_matrix is not None:
                if self.image_frame._image is not None:
                    h, w = self.image_frame._image.shape[:2]
                    self._camera_matrix = get_rotated_intrinsics(self._camera_matrix, h, w, is_clockwise=(self.camera_type == "right"))
                    
            if self._T_base_to_cam is not None:
                self._T_base_to_cam = get_rotated_extrinsics(self._T_base_to_cam, is_clockwise=(self.camera_type == "right"))
                
        self._is_upright = True

    @property
    def image(self):
        self._apply_rotation()
        self._decompress_image()
        return self.image_frame._image
        
    @property
    def depth_image(self):
        self._apply_rotation()
        return self._depth_image
        
    @property
    def camera_matrix(self):
        self._apply_rotation()
        return self._camera_matrix
        
    @property
    def T_base_to_cam(self):
        self._apply_rotation()
        return self._T_base_to_cam

    def to_dict(self):
        """
        Serializes the RGBDFrame into a clean dictionary for network transmission.
        Strictly serializes the native data to avoid transporting rotated artifacts.
        """
        data = {
            "timestamp": self.timestamp,
            "camera_type": self.camera_type,
            "point_cloud": self.point_cloud,
            "point_cloud_base": self.point_cloud_base,
            "point_colors": self.point_colors,
            "depth_image": self._depth_image,
            "camera_matrix": self._camera_matrix,
            "distortion_coefficients": self.distortion_coefficients,
            "T_base_to_cam": self._T_base_to_cam,
            "T_lidar_to_base_left": self.T_lidar_to_base_left,
            "T_lidar_to_base_right": self.T_lidar_to_base_right,
            "robot_id": self.robot_id,
            "timestamp_image": self.timestamp_image,
            "timestamp_lidar_left": self.timestamp_lidar_left,
            "timestamp_lidar_right": self.timestamp_lidar_right,
            "lidars_used": self.lidars_used,
            "image_frame_timestamp": self.image_frame.timestamp,
            "image_frame_number": self.image_frame.frame_number,
            "is_upright": self._is_upright
        }
        
        # Optimize image transmission
        if self.image_frame.image_raw is not None and isinstance(self.image_frame.image_raw, (bytes, bytearray, memoryview, np.ndarray)) and (isinstance(self.image_frame.image_raw, (bytes, bytearray, memoryview)) or self.image_frame.image_raw.ndim == 1):
            data["image_raw"] = self.image_frame.image_raw
            data["image"] = None 
        else:
            data["image_raw"] = self.image_frame.image_raw
            data["image"] = self.image_frame._image
            
        data["image_rectified"] = self.image_frame.image_rectified
        data["new_K"] = self.image_frame.new_K
        
        return data

    @classmethod
    def from_dict(cls, data):
        """
        Reconstructs the RGBDFrame from a dictionary.
        """
        img_frame = ImageFrame(
            timestamp=data.get("image_frame_timestamp", 0.0),
            frame_number=data.get("image_frame_number", 0),
            image=data.get("image"),
            image_raw=data.get("image_raw"),
            image_rectified=data.get("image_rectified"),
            new_K=data.get("new_K")
        )
        frame = cls(
            timestamp=data.get("timestamp", 0.0),
            image_frame=img_frame,
            camera_type=data.get("camera_type", "unknown"),
            point_cloud=data.get("point_cloud"),
            point_cloud_base=data.get("point_cloud_base"),
            point_colors=data.get("point_colors"),
            depth_image=data.get("depth_image"),
            camera_matrix=data.get("camera_matrix"),
            distortion_coefficients=data.get("distortion_coefficients"),
            T_base_to_cam=data.get("T_base_to_cam"),
            T_lidar_to_base_left=data.get("T_lidar_to_base_left"),
            T_lidar_to_base_right=data.get("T_lidar_to_base_right"),
            robot_id=data.get("robot_id"),
            timestamp_image=data.get("timestamp_image"),
            timestamp_lidar_left=data.get("timestamp_lidar_left"),
            timestamp_lidar_right=data.get("timestamp_lidar_right"),
            lidars_used=data.get("lidars_used")
        )
        frame._is_upright = data.get("is_upright", False)
        return frame

class CapturedSequence:
    """
    A unified wrapper that encapsulates both the lightweight RGBDFrame
    and the heavy raw 360-degree LiDAR point clouds necessary for offline 
    extrinsic optimization. Handles clean disk I/O.
    """
    def __init__(self, frame: RGBDFrame, raw_lidar_left: np.ndarray = None, raw_lidar_right: np.ndarray = None):
        self.frame = frame
        self.raw_lidar_left = raw_lidar_left
        self.raw_lidar_right = raw_lidar_right
        self.metadata = {}
        
    def save(self, directory_path, save_dense_depth=True, metadata_extras=None):
        import yaml
        import cv2
        import os
        from pathlib import Path
        import time
        
        dir_path = Path(directory_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        if self.frame.image is not None:
            cv2.imwrite(str(dir_path / "rgb.png"), self.frame.image)
            
        if self.frame.depth_image is not None:
            np.savez_compressed(str(dir_path / "sparse_depth.npz"), depth=self.frame.depth_image)
            
        if self.raw_lidar_left is not None:
            np.savez_compressed(str(dir_path / "lidar_left.npz"), pts=self.raw_lidar_left)
        if self.raw_lidar_right is not None:
            np.savez_compressed(str(dir_path / "lidar_right.npz"), pts=self.raw_lidar_right)
            
        if save_dense_depth and self.frame.image is not None and self.frame.depth_image is not None:
            dense_img = DenseDepthImage(
                self.frame.image, 
                self.frame.depth_image, 
                apply_validity_mask=True, 
                camera_name=self.frame.camera_type, 
                lidar_name=getattr(self.frame, "lidars_used", "both_lidar")
            )
            dense_rgbd = dense_img.compute_dense_rgbd()
            if dense_rgbd is not None:
                cv2.imwrite(str(dir_path / "dense_depth.png"), dense_rgbd)
                
        def _safe_float(val):
            if val is None: return None
            try:
                if hasattr(val, "size") and val.size == 0: return None
                return float(np.atleast_1d(val)[0])
            except:
                return float(val)
                
        metadata = {
            "camera_name": self.frame.camera_type,
            "robot_id": self.frame.robot_id,
            "timestamp_system": _safe_float(time.time()),
            "timestamp_rgbd": _safe_float(self.frame.timestamp),
            "timestamp_image": _safe_float(self.frame.timestamp_image),
            "timestamp_lidar_left": _safe_float(self.frame.timestamp_lidar_left),
            "timestamp_lidar_right": _safe_float(self.frame.timestamp_lidar_right),
            "lidars_used": str(self.frame.lidars_used) if self.frame.lidars_used is not None else None,
            "camera_matrix": self.frame.camera_matrix.tolist() if self.frame.camera_matrix is not None else None,
            "distortion_coefficients": self.frame.distortion_coefficients.tolist() if self.frame.distortion_coefficients is not None else None,
            "T_base_to_cam": self.frame.T_base_to_cam.tolist() if self.frame.T_base_to_cam is not None else None,
            "T_lidar_to_base_left": self.frame.T_lidar_to_base_left.tolist() if self.frame.T_lidar_to_base_left is not None else None,
            "T_lidar_to_base_right": self.frame.T_lidar_to_base_right.tolist() if self.frame.T_lidar_to_base_right is not None else None
        }
        
        if metadata_extras:
            for k, v in metadata_extras.items():
                if hasattr(v, "size"): continue # Basic protection against passing unhandled numpy arrays
                metadata[k] = v
                
        with open(dir_path / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f, default_flow_style=None)
            
    @classmethod
    def load(cls, directory_path):
        import yaml
        import cv2
        from pathlib import Path
        dir_path = Path(directory_path)
        
        meta_path = dir_path / "metadata.yaml"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")
            
        with open(meta_path, "r") as f:
            metadata = yaml.safe_load(f)
            
        rgb_img = cv2.imread(str(dir_path / "rgb.png"))
        
        l_pts = np.load(str(dir_path / "lidar_left.npz"))["pts"] if (dir_path / "lidar_left.npz").exists() else None
        r_pts = np.load(str(dir_path / "lidar_right.npz"))["pts"] if (dir_path / "lidar_right.npz").exists() else None
        
        c_name = metadata.get("camera_name", "left")
        T_base_to_cam = np.array(metadata["T_base_to_cam"]) if metadata.get("T_base_to_cam") is not None else None
        T_lidar_to_base_left = np.array(metadata["T_lidar_to_base_left"]) if metadata.get("T_lidar_to_base_left") is not None else None
        T_lidar_to_base_right = np.array(metadata["T_lidar_to_base_right"]) if metadata.get("T_lidar_to_base_right") is not None else None
        camera_matrix = np.array(metadata["camera_matrix"]) if metadata.get("camera_matrix") is not None else None
        dist_coeffs = np.array(metadata["distortion_coefficients"]) if metadata.get("distortion_coefficients") is not None else None
        camera_model = metadata.get("camera_model", "fisheye" if metadata.get("is_fisheye", False) else "pinhole")
        omnidir_xi = metadata.get("omnidir_xi", 0.0)
        timestamp = metadata.get("timestamp_rgbd", 0.0)
        
        frame = reconstruct_rgbd_frame(
            c_name, T_base_to_cam, T_lidar_to_base_left, T_lidar_to_base_right, 
            camera_matrix, dist_coeffs, camera_model, omnidir_xi, rgb_img, l_pts, r_pts, timestamp
        )
        
        frame.robot_id = metadata.get("robot_id")
        frame.timestamp_image = metadata.get("timestamp_image")
        frame.timestamp_lidar_left = metadata.get("timestamp_lidar_left")
        frame.timestamp_lidar_right = metadata.get("timestamp_lidar_right")
        frame.lidars_used = metadata.get("lidars_used")
        
        seq = cls(frame=frame, raw_lidar_left=l_pts, raw_lidar_right=r_pts)
        seq.metadata = metadata
        return seq

class LoopTimer:
    def __init__(self):
        self.iterations = 0
        self.start_time = time.perf_counter()
        self.last_time = time.perf_counter()
        self.total_duration = 0.0

    def start_of_iteration(self):
        self.last_time = time.perf_counter()

    def end_of_iteration(self):
        duration = time.perf_counter() - self.last_time
        self.total_duration += duration
        self.iterations += 1

    def pretty_print(self, minimum=False):
        if self.iterations > 0:
            avg_duration = self.total_duration / self.iterations
            print(f"Average period: {avg_duration*1000:.2f} ms ({1.0/avg_duration:.2f} Hz)")

class ExtrinsicsCalibration:
    """
    Encapsulates the optimized rigid body extrinsic calibration between the LiDAR and cameras.
    
    The optimization algorithm estimates a correction transformation (`T_delta`) that 
    minimizes the spatial misalignment between the projected LiDAR depth and the RGB image.
    This correction mathematically updates the camera's pose within the robot's base frame.
    """
    def __init__(self, T_delta: np.ndarray):
        """
        Args:
            T_delta: A 4x4 transformation matrix representing the optimized correction.
        """
        self.T_delta = T_delta
        self.T_delta_inv = np.linalg.inv(T_delta)

    @classmethod
    def load_from_yaml(cls, filepath: str) -> 'ExtrinsicsCalibration':
        """
        Loads an optimized calibration from a YAML file.
        
        Args:
            filepath: Path to the optimization YAML file (e.g., optimization_results_mi_rgb_*.yaml)
            
        Returns:
            ExtrinsicsCalibration instance, or None if the file is invalid.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"Error: Optimization YAML file '{filepath}' not found.")
            return None
            
        with open(filepath, 'r') as f:
            opt_data = yaml.safe_load(f)
            
        if "results" in opt_data and "best_delta_transform_matrix" in opt_data["results"]:
            T_delta = np.array(opt_data["results"]["best_delta_transform_matrix"])
            print(f"Successfully loaded optimization transform from {filepath}")
            return cls(T_delta)
        else:
            print(f"Error: YAML file '{filepath}' does not contain expected optimization results.")
            return None

    def apply_to_camera_extrinsics(self, T_base_to_cam: np.ndarray) -> np.ndarray:
        """
        Applies the optimized correction to a camera's extrinsic matrix.
        
        The optimization is parameterized such that `T_delta` represents a shift in the 
        camera's pose. To mathematically correct the points projected into the camera, 
        we pre-multiply the camera's original extrinsic matrix (`T_base_to_cam`) by 
        the inverse of `T_delta`.
        
        Mathematically: T_base_to_cam_new = inv(T_delta) @ T_base_to_cam
        
        Args:
            T_base_to_cam: The original 4x4 matrix transforming points from the base frame to the camera frame.
            
        Returns:
            The corrected 4x4 matrix `T_base_to_cam_new`.
        """
        return self.T_delta_inv @ T_base_to_cam

def get_rotated_intrinsics(camera_matrix: np.ndarray, width: int, height: int, is_clockwise=False) -> np.ndarray:
    """
    Computes the new camera matrix after rotating the image by 90 degrees.
    """
    if camera_matrix is None:
        return None
        
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    new_matrix = camera_matrix.copy()
    new_matrix[0, 0] = fy
    new_matrix[1, 1] = fx
    
    if is_clockwise:
        new_matrix[0, 2] = height - cy
        new_matrix[1, 2] = cx
    else:
        new_matrix[0, 2] = cy
        new_matrix[1, 2] = width - cx
        
    return new_matrix

def get_rotated_extrinsics(T_base_to_cam: np.ndarray, is_clockwise=False) -> np.ndarray:
    """
    Computes the new extrinsic matrix after the camera's image is rotated by 90 degrees.
    This effectively rotates the camera's optical frame around its Z-axis.
    """
    if T_base_to_cam is None:
        return None
        
    if is_clockwise:
        R_old_to_new = np.array([
            [ 0, -1,  0,  0],
            [ 1,  0,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1]
        ], dtype=np.float64)
    else:
        R_old_to_new = np.array([
            [ 0,  1,  0,  0],
            [-1,  0,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1]
        ], dtype=np.float64)
        
    return R_old_to_new @ T_base_to_cam

def get_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--camera", type=str, choices=["left", "right", "center", "left_right", "all"], default="left", help="Which camera(s) to use")
    parser.add_argument("--lidar", type=str, choices=["left", "right", "both"], default="both", help="Which lidar to use")
    parser.add_argument("--show_fps", action="store_true", help="Show the FPS of the stream. Default: False.")
    parser.add_argument("--opt_yaml", type=str, default=None, help="Path to a rigid body transform optimization YAML file to apply.")
    
    # Standard streaming arguments
    parser.add_argument("--resolution", type=int, default=800, help="Vertical resolution for low-latency streamer (400, 600, 800, 1200). Default: 800.")
    parser.add_argument("--disable_compression", action='store_true', help="Disable MJPEG compression over USB.")
    parser.add_argument("--oak_buffer_size", type=int, default=1, help="Size of the OAK output queue. Default: 1.")
    parser.add_argument("--emulated_rgbd_fps", type=float, default=10.0, help="Target FPS for the final Emulated RGB-D imagery output (e.g., 10, 5, 3.33). Default: 10.0.")
    parser.add_argument("--camera_fps", type=int, default=30, help="Hardware FPS for the OAK-FFC camera. Available options: 10, 15, 20, 25, 30, 60. Set higher than emulated_rgbd_fps for software over-sampling to reduce phase latency. Default: 30.")
    return parser

class NonBlockingInput:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def get_char(self):
        if select.select([sys.stdin], [], [], 0.0)[0]:
            return sys.stdin.read(1)
        return None

    def cleanup(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

class DenseDepthImage:
    def __init__(self, rgb_image, sparse_depth_image, apply_validity_mask=True, camera_name=None, lidar_name=None):
        self.rgb_image = rgb_image.copy()
        if sparse_depth_image is not None:
            self.sparse_depth_image = sparse_depth_image.copy()
        else:
            self.sparse_depth_image = None
        self.dense_depth_image = None
        self.dense_rgbd_image = None
        self.apply_validity_mask = apply_validity_mask
        self.camera_name = camera_name
        self.lidar_name = lidar_name

    def compute_dense_depth(self, valid_region_mask=None):
        if self.sparse_depth_image is None or self.sparse_depth_image.shape[0] == 0:
            return None
            
        valid_mask = (self.sparse_depth_image > 0) & (self.sparse_depth_image != np.inf) & (~np.isnan(self.sparse_depth_image))
        z = self.sparse_depth_image[valid_mask]
        
        if len(z) == 0:
            self.dense_depth_image = np.zeros_like(self.sparse_depth_image)
            return self.dense_depth_image
            
        # Fast nearest-neighbor interpolation using OpenCV's distance transform with Voronoi labels
        mask = np.where(valid_mask, 0, 255).astype(np.uint8)
        dist, labels = cv2.distanceTransformWithLabels(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
        
        valid_labels = labels[valid_mask]
        max_label = np.max(labels)
        label_to_depth = np.zeros(max_label + 1, dtype=np.float32)
        label_to_depth[valid_labels] = z
        
        self.dense_depth_image = label_to_depth[labels]
        
        # Mask out values that are too far from a valid sparse point
        self.dense_depth_image[dist > config.MAX_LIDAR_INTERPOLATION_DIST_PX] = 0.0
        
        # Apply combined default validity mask if requested and available
        if self.apply_validity_mask and valid_region_mask is None and self.camera_name and self.lidar_name:
            valid_region_mask = _GLOBAL_MASK_MANAGER.get_combined_mask(self.camera_name, self.lidar_name, self.rgb_image.shape)

        if valid_region_mask is not None:
            self.dense_depth_image[~valid_region_mask] = 0.0
        
        return self.dense_depth_image

    def compute_dense_rgbd(self):
        if self.dense_depth_image is None:
            self.compute_dense_depth()
            
        if self.dense_depth_image is None:
            return None
            
        max_depth = config.VISUALIZATION_MAX_DEPTH_M
        depth_norm = np.clip(self.dense_depth_image / max_depth, 0, 1)
        depth_colormap = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        alpha = config.DENSE_RGBD_ALPHA
        self.dense_rgbd_image = cv2.addWeighted(self.rgb_image, 1 - alpha, depth_colormap, alpha, 0)
        return self.dense_rgbd_image

def render_rgbd(c_name: str, frame: RGBDFrame, vig_mask=None, depth_mask=None):
    rr.set_time("timestamp", timestamp=frame.timestamp)
    image_bgr = frame.image.copy()
    
    if vig_mask is not None:
        image_bgr[~vig_mask] = 0
    
    if not config.ROTATE_IMAGES_TO_VERTICAL:
        if c_name == "left":
            image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif c_name == "right":
            image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    
    # Dense depth computations
    depth_vis = None
    dense_processor = None
    if frame.depth_image is not None and frame.depth_image.shape[0] > 0:
        depth_vis = frame.depth_image
        if not config.ROTATE_IMAGES_TO_VERTICAL:
            if c_name == "left":
                depth_vis = cv2.rotate(depth_vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif c_name == "right":
                depth_vis = cv2.rotate(depth_vis, cv2.ROTATE_90_CLOCKWISE)
            
        dense_processor = DenseDepthImage(
            frame.image, 
            frame.depth_image, 
            apply_validity_mask=True, 
            camera_name=c_name, 
            lidar_name=getattr(frame, "lidars_used", "both_lidar")
        )
        
        combined_mask = None
        if vig_mask is not None and depth_mask is not None:
            combined_mask = vig_mask & depth_mask
        elif vig_mask is not None:
            combined_mask = vig_mask
        elif depth_mask is not None:
            combined_mask = depth_mask
            
        dense_processor.compute_dense_depth(valid_region_mask=combined_mask)

    # 2D Data
    if dense_processor is not None and dense_processor.dense_depth_image is not None:
        dd = dense_processor.dense_depth_image.copy()
        if not config.ROTATE_IMAGES_TO_VERTICAL:
            if c_name == "left":
                dd = cv2.rotate(dd, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif c_name == "right":
                dd = cv2.rotate(dd, cv2.ROTATE_90_CLOCKWISE)
        rr.log(f"Cameras/{c_name}/dense_depth", rr.DepthImage(dd, meter=1.0, depth_range=[0.0, config.RERUN_COLOR_MAX_DEPTH_M]))

    if depth_vis is not None:
        rr.log(f"Cameras/{c_name}/depth", rr.DepthImage(depth_vis, meter=1.0, depth_range=[0.0, config.RERUN_COLOR_MAX_DEPTH_M]))

    rr.log(f"Cameras/{c_name}/rgb", rr.Image(image_bgr, color_model="BGR").compress())

    # 3D Data
    if len(frame.point_cloud_base) > 0:
        rr.log(
            f"Pointclouds/base_frame/{c_name}",
            rr.Points3D(frame.point_cloud_base, colors=frame.point_colors, radii=[0.01]),
        )

def apply_shadow_filter(sparse_depth_image, window_size=5, depth_threshold=0.3, use_circular_window=False):
    """
    Applies a moving window shadow filter to a sparse depth image.
    Removes background points that are occluded by foreground points in their neighborhood.
    This resolves the "striping" effect caused by LiDAR sparsity around object edges.
    """
    if window_size <= 1:
        return sparse_depth_image
        
    # Replace 0 with infinity to correctly find the local minimum depth
    depth_inf = sparse_depth_image.copy()
    depth_inf[depth_inf == 0] = np.inf
    
    # Erode operation finds the local minimum within the window
    if use_circular_window:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window_size, window_size))
    else:
        kernel = np.ones((window_size, window_size), np.uint8)
        
    min_depth = cv2.erode(depth_inf, kernel)
    
    # Identify shadowed points: their original depth is significantly larger than the local minimum
    shadowed = (sparse_depth_image > 0) & (sparse_depth_image - min_depth > depth_threshold)
    
    filtered_depth = sparse_depth_image.copy()
    filtered_depth[shadowed] = 0.0
    return filtered_depth

def reconstruct_rgbd_frame(c_name, T_base_to_cam, T_lidar_to_base_left, T_lidar_to_base_right, 
                           camera_matrix, dist_coeffs, camera_model, xi, rgb_img, l_pts, r_pts, current_replay_time):
    """
    Reconstructs an RGBDFrame by projecting LiDAR points into a camera's view.
    
    Args:
        c_name: Name of the camera.
        T_base_to_cam: 4x4 transformation matrix from the base frame to the camera optical frame.
                       This matrix transforms a 3D point in the base frame into the camera frame.
        T_lidar_to_base_left: 4x4 transformation matrix from the left LiDAR frame to the base frame.
        T_lidar_to_base_right: 4x4 transformation matrix from the right LiDAR frame to the base frame.
        camera_matrix: 3x3 intrinsic camera matrix.
        dist_coeffs: Camera distortion coefficients.
        camera_model: The camera projection model ('pinhole', 'fisheye', or 'omnidir').
        xi: The omnidirectional camera parameter (used only if camera_model == 'omnidir').
        rgb_img: The BGR image from the camera.
        l_pts: NumPy array of left LiDAR points.
        r_pts: NumPy array of right LiDAR points.
        current_replay_time: Timestamp for the frame.
    """
    pts_base = merge_lidar_points(l_pts, r_pts, T_lidar_to_base_left, T_lidar_to_base_right)
        
        
    depth_img = np.zeros(rgb_img.shape[:2], dtype=np.float32)
    pts_cam = np.zeros((0, 3))
    cols = np.zeros((0, 3))
    pts_base_valid = np.zeros((0, 3))
    
    if len(pts_base) > 0 and camera_matrix is not None:
        ones = np.ones((len(pts_base), 1))
        pts_cam_all = (T_base_to_cam @ np.hstack([pts_base, ones]).T).T[:, :3]
        
        valid_idx = pts_cam_all[:, 2] > 0
        pts_cam_valid = pts_cam_all[valid_idx]
        pts_base_valid = pts_base[valid_idx]
        
        if len(pts_cam_valid) > 0:
            rvec = np.zeros(3)
            tvec = np.zeros(3)
            img_pts = project_points(
                pts_cam_valid, rvec, tvec, camera_matrix, dist_coeffs, camera_model, xi
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
                        depth_threshold=config.SHADOW_FILTER_DEPTH_THRESHOLD_M,
                        use_circular_window=config.SHADOW_FILTER_USE_CIRCULAR_WINDOW
                    )
                    
                valid_mask = depth_img > 0
                surviving_indices = index_img[valid_mask]
                
                pts_cam = pts_cam_valid[valid_uv][surviving_indices]
                pts_base_valid = pts_base_valid[valid_uv][surviving_indices]
                
                v_filtered, u_filtered = np.where(valid_mask)
                colors_bgr = rgb_img[v_filtered, u_filtered]
                cols = colors_bgr[:, ::-1]
            else:
                pts_cam = np.zeros((0, 3))
                pts_base_valid = np.zeros((0, 3))
                cols = np.zeros((0, 3))
    
    img_frame = ImageFrame(timestamp=current_replay_time, frame_number=0, image=rgb_img)
    img_frame.image_raw = rgb_img
    
    frame = RGBDFrame(
        timestamp=current_replay_time,
        image_frame=img_frame,
        camera_type=c_name,
        point_cloud=pts_cam,
        point_cloud_base=pts_base_valid if len(pts_base) > 0 else np.zeros((0, 3)),
        point_colors=cols,
        depth_image=depth_img,
        camera_matrix=camera_matrix,
        distortion_coefficients=dist_coeffs,
        T_base_to_cam=T_base_to_cam,
        T_lidar_to_base_left=T_lidar_to_base_left,
        T_lidar_to_base_right=T_lidar_to_base_right
    )
    
    # If the loaded image is already vertical (height > width), flag it as upright
    # to prevent _apply_rotation from double-rotating it.
    if rgb_img.shape[0] > rgb_img.shape[1]:
        frame._is_upright = True
        
    return frame

def get_vignette_mask(rgb_img):
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, config.VIGNETTE_MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        vignette_mask = np.zeros_like(gray)
        cv2.drawContours(vignette_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        # Erode the mask aggressively to avoid catching the vignette boundary and its blurred artifacts as an edge
        k_size = config.VIGNETTE_MASK_EROSION_KERNEL_SIZE
        kernel = np.ones((k_size, k_size), np.uint8)
        vignette_mask = cv2.erode(vignette_mask, kernel, iterations=2)
        return vignette_mask > 0
    return np.ones_like(gray, dtype=bool)

def get_saturation_mask(rgb_img, threshold=None):
    if threshold is None:
        threshold = config.SATURATION_THRESHOLD_GRAY
        
    # Convert to grayscale
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    
    # Mask where pixels are excessively bright
    saturation_mask = gray >= threshold
    
    # Dilate the mask slightly to cover blooming edges
    k_size = config.SATURATION_MASK_DILATION_KERNEL_SIZE
    kernel = np.ones((k_size, k_size), np.uint8)
    saturation_mask = cv2.dilate(saturation_mask.astype(np.uint8), kernel, iterations=1)
    
    return saturation_mask > 0

def generate_rgb_vignetting_mask(rgb_images):
    """
    Generate an RGB vignetting mask from a list of RGB images.
    
    Why: Wide-angle and fisheye lenses often have dark, occluded corners (vignetting) 
    caused by the physical housing of the camera. This function estimates those bounds 
    so we can explicitly mask them out. This prevents computer vision models from 
    interpreting the black housing rims as physical objects in the environment.
    """
    if not rgb_images:
        return None
    
    # Calculate average image across all captured frames to smooth out 
    # environmental lighting changes and sensor noise.
    sum_rgb = np.zeros(rgb_images[0].shape[:2], dtype=np.float32)
    for img in rgb_images:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sum_rgb += img
    
    avg_rgb = (sum_rgb / len(rgb_images)).astype(np.uint8)
    
    # Threshold the average image to isolate the bright interior (the valid pixels)
    # from the dark vignetted edges. We use the global config threshold for consistency.
    _, thresh = cv2.threshold(avg_rgb, config.VIGNETTE_MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # The vignetted area is roughly circular. Find the largest continuous bright 
    # contour and fit an enclosing circle to it. This produces a smooth, clean mask
    # rather than a noisy, jagged threshold edge.
    conts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = avg_rgb.shape
    rgb_vignette_mask = np.zeros((h, w), dtype=np.uint8)
    
    if conts:
        c_max = max(conts, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(c_max)
        center = (int(cx), int(cy))
        radius = int(radius)
        cv2.circle(rgb_vignette_mask, center, radius, 255, -1)
        
    return rgb_vignette_mask

def generate_depth_validity_mask(depth_images):
    """
    Generate a depth validity mask from a list of sparse depth images.
    
    Why: LiDAR sensors have a specific field of view. When we project the sparse 
    LiDAR points into the camera frame and interpolate them to create a "dense" 
    depth map, the interpolation algorithm can wildly extrapolate depths into 
    regions of the camera image that the LiDAR physically cannot see. This function
    accumulates sparse points over time to find the true, stable boundary of the 
    LiDAR's physical coverage, allowing us to mask out unbounded extrapolations.
    """
    if not depth_images:
        return None
        
    # Accumulate all valid sparse depth pixels across the entire temporal sequence.
    # Because LiDAR is sparse and the robot might be moving slightly, this builds a 
    # complete "map" of everywhere the LiDAR has successfully returned a point.
    accumulated_lidar = np.zeros(depth_images[0].shape[:2], dtype=np.uint8)
    for depth in depth_images:
        accumulated_lidar[depth > 0] = 1
        
    h, w = accumulated_lidar.shape
    
    if config.USE_DENSITY_BASED_DEPTH_MASK:
        # Density-based approach
        window_size = config.DEPTH_MASK_DENSITY_WINDOW_SIZE
        # Use boxFilter to compute the fraction of valid pixels in the window
        density = cv2.boxFilter(accumulated_lidar.astype(np.float32), -1, (window_size, window_size), normalize=True)
        
        # Calculate median density of valid points
        valid_densities = density[accumulated_lidar > 0]
        if len(valid_densities) == 0:
            return np.zeros((h, w), dtype=np.uint8)
            
        median_density = np.median(valid_densities)
        threshold = median_density * config.DEPTH_MASK_DENSITY_THRESHOLD_RATIO
        
        # Threshold to get valid region
        valid_region = (density >= threshold).astype(np.uint8) * 255
        
        # Morphological closing to fill holes
        kernel_size = config.DEPTH_MASK_CLOSING_KERNEL_SIZE
        closing_iterations = getattr(config, 'DEPTH_MASK_CLOSING_ITERATIONS', 1)
        # Use an elliptical kernel to be isotropic
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        depth_valid_mask = cv2.morphologyEx(valid_region, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)
        
    else:
        # Distance-transform approach
        # Pad accumulated_lidar so the distance transform and erosion can expand and contract
        # naturally at the image boundaries without artificial clipping.
        max_gap_px = config.MAX_LIDAR_INTERPOLATION_DIST_PX
        erosion_radius = getattr(config, 'OUTER_BOUNDARY_EROSION_RADIUS', int(max_gap_px))
        pad_size = int(erosion_radius)
        
        padded_lidar = cv2.copyMakeBorder(accumulated_lidar, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
        
        mask_inv = np.where(padded_lidar > 0, 0, 255).astype(np.uint8)
        dist, _ = cv2.distanceTransformWithLabels(mask_inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
        
        base_mask = np.where(dist <= max_gap_px, 255, 0).astype(np.uint8)
        
        # Shrink the outer edge using erosion
        kernel_size = 2 * erosion_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        eroded_mask = cv2.erode(base_mask, kernel)
        
        # Pixels that were removed by erosion (the border region)
        eroded_pixels = cv2.bitwise_xor(base_mask, eroded_mask)
        
        # Re-assess the border region using a tighter threshold
        tight_thresh = getattr(config, 'OUTER_BOUNDARY_DISTANCE_THRESH_PX', 10.0)
        tight_mask = np.where(dist <= tight_thresh, 255, 0).astype(np.uint8)
        
        # Final mask keeps the eroded interior PLUS the border pixels that meet the tight threshold
        padded_valid_mask = cv2.bitwise_or(eroded_mask, cv2.bitwise_and(eroded_pixels, tight_mask))
        
        # Crop back to the original size
        depth_valid_mask = padded_valid_mask[pad_size:-pad_size, pad_size:-pad_size]
        
    # Find largest connected component to remove noise
    conts, _ = cv2.findContours(depth_valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_depth_valid_mask = np.zeros((h, w), dtype=np.uint8)
    if conts:
        c_max = max(conts, key=cv2.contourArea)
        cv2.drawContours(clean_depth_valid_mask, [c_max], -1, 255, -1)
        
    return clean_depth_valid_mask

class ValidityMaskManager:
    """
    Manages loading and applying static validity masks for the cameras.
    
    Why: The validity masks (vignetting & depth bounds) are generated once per camera 
    at its highest resolution (e.g., 800p). However, users might stream the cameras at 
    lower resolutions (e.g., 400p) for performance. This class caches the masks and 
    dynamically scales them down to precisely match the active stream's resolution, 
    avoiding the need to generate new masks for every possible resolution.
    """
    def __init__(self, masks_dir=None):
        if masks_dir is None:
            fleet_path = os.environ.get("HELLO_FLEET_PATH", "")
            fleet_id = os.environ.get("HELLO_FLEET_ID", "")
            if fleet_path and fleet_id:
                masks_dir = os.path.join(fleet_path, fleet_id, "calibration_cameras")
            else:
                masks_dir = "data/validity_masks"
        self.masks_dir = Path(masks_dir)
        self.masks_cache = {}

    def get_masks(self, camera_name, lidar_name, shape):
        """
        Returns (vignette_mask, depth_valid_mask) as boolean arrays scaled to shape (h, w).
        If not found on disk, returns (None, None) for each respectively.
        """
        h, w = shape[:2]
        key = (camera_name, lidar_name, h, w)
        if key in self.masks_cache:
            return self.masks_cache[key]
            
        vig_path = self.masks_dir / f"rgb_vignette_mask_{camera_name}_camera.png"
        depth_path = self.masks_dir / f"depth_valid_mask_{camera_name}_camera_{lidar_name}.png"
        
        vig_mask = None
        depth_mask = None
        
        if vig_path.exists():
            vig_img = cv2.imread(str(vig_path), cv2.IMREAD_GRAYSCALE)
            if vig_img is not None:
                if config.ROTATE_IMAGES_TO_VERTICAL and camera_name in ["left", "right"]:
                    # Only rotate if the mask on disk is still in the old horizontal format
                    if vig_img.shape[0] < vig_img.shape[1]:
                        is_cw = (camera_name == "right")
                        vig_img = cv2.rotate(vig_img, cv2.ROTATE_90_CLOCKWISE if is_cw else cv2.ROTATE_90_COUNTERCLOCKWISE)
                vig_mask = cv2.resize(vig_img, (w, h), interpolation=cv2.INTER_NEAREST) > 0
                
        if depth_path.exists():
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
            if depth_img is not None:
                if config.ROTATE_IMAGES_TO_VERTICAL and camera_name in ["left", "right"]:
                    # Only rotate if the mask on disk is still in the old horizontal format
                    if depth_img.shape[0] < depth_img.shape[1]:
                        is_cw = (camera_name == "right")
                        depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE if is_cw else cv2.ROTATE_90_COUNTERCLOCKWISE)
                depth_mask = cv2.resize(depth_img, (w, h), interpolation=cv2.INTER_NEAREST) > 0
                
        self.masks_cache[key] = (vig_mask, depth_mask)
        return self.masks_cache[key]

    def get_combined_mask(self, camera_name, lidar_name, shape):
        """
        Returns the logical AND of the vignette mask and the depth mask.
        If neither exist, returns None.
        If only one exists, returns that one.
        """
        vig_mask, depth_mask = self.get_masks(camera_name, lidar_name, shape)
        if vig_mask is not None and depth_mask is not None:
            return vig_mask & depth_mask
        elif vig_mask is not None:
            return vig_mask
        elif depth_mask is not None:
            return depth_mask
        return None

_GLOBAL_MASK_MANAGER = ValidityMaskManager()

def load_and_confirm_optimization_masks(data_path, camera_name, lidar_name):
    import cv2
    from pathlib import Path
    import numpy as np
    
    data_path = Path(data_path)
    masks_dir = data_path / "validity_masks_for_extrinsic_optimization"
    rgb_mask_path = masks_dir / f"rgb_vignette_mask_{camera_name}_camera.png"
    depth_mask_path = masks_dir / f"depth_valid_mask_{camera_name}_camera_{lidar_name}.png"
    
    if not rgb_mask_path.exists() or not depth_mask_path.exists():
        raise RuntimeError(
            f"Validity masks not found in {masks_dir}.\n"
            f"Please run `python3 scripts/create_validity_masks_for_extrinsic_optimization.py {data_path}` "
            "to generate them before proceeding."
        )
        
    rgb_vignette_mask = cv2.imread(str(rgb_mask_path), cv2.IMREAD_GRAYSCALE)
    depth_valid_mask = cv2.imread(str(depth_mask_path), cv2.IMREAD_GRAYSCALE)
    
    if rgb_vignette_mask is None or depth_valid_mask is None:
        raise RuntimeError("Failed to read the validity masks. They might be corrupted.")

    # Ensure boolean masks
    rgb_vignette_mask = rgb_vignette_mask > 0
    depth_valid_mask = depth_valid_mask > 0

    # Visualize and confirm
    cv2.imshow(f"RGB Vignette Mask ({camera_name})", (rgb_vignette_mask.astype(np.uint8) * 255))
    cv2.imshow(f"Depth Valid Mask ({camera_name}, {lidar_name})", (depth_valid_mask.astype(np.uint8) * 255))
    
    print("\n" + "="*80)
    print("Please inspect the validity masks in the OpenCV windows.")
    print("Press 'y' in any window to confirm they are ready to be used, or 'n' to abort.")
    print("="*80 + "\n")
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            print("Masks confirmed. Proceeding...")
            break
        elif key == ord('n') or key == 27: # 27 is ESC
            cv2.destroyAllWindows()
            raise RuntimeError("Aborted by user.")
            
    cv2.destroyAllWindows()
    return rgb_vignette_mask, depth_valid_mask

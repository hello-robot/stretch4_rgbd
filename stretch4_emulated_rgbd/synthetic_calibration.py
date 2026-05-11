#!/usr/bin/env python3
"""
synthetic_calibration.py

This script converts existing camera models (e.g., Kannala-Brandt fisheye) 
to the OpenCV omnidirectional (Mei) camera model by simulating a synthetic calibration.
It generates a 3D checkerboard, places it in various random poses in front of the camera,
projects the points using the existing intrinsics, and then runs cv2.omnidir.calibrate
on the point pairs to find the optimal omnidir parameters (including the xi structural parameter).
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import yaml
import shutil
from scipy.spatial.transform import Rotation

from stretch4_emulated_rgbd.shared_utils import find_latest_data_dir

from scipy.optimize import least_squares

def generate_rays(num_rays=5000, fov_deg=180):
    """Generate random 3D rays within the specified field of view."""
    phi = np.random.uniform(0, 2 * np.pi, num_rays)
    
    # max theta based on FOV
    max_theta = np.deg2rad(fov_deg / 2.0)
    
    # Uniformly sample on the spherical cap
    # cos(theta) is uniformly distributed between cos(max_theta) and 1
    cos_theta = np.random.uniform(np.cos(max_theta), 1.0, num_rays)
    theta = np.arccos(cos_theta)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return np.column_stack((x, y, z))

def convert_to_omnidir(img_size, camera_matrix, dist_coeffs, is_fisheye):
    # Generate 3D rays covering a wide field of view
    pts3d = generate_rays(num_rays=10000, fov_deg=180)
    
    # Project with the existing model
    if is_fisheye:
        imgpts, _ = cv2.fisheye.projectPoints(pts3d.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    else:
        imgpts, _ = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
        
    imgpts = imgpts.reshape(-1, 2)
    
    # We only care about matching the projection for points that actually fall 
    # somewhat near the image sensor. A 20% margin is included to ensure smooth behavior at the edges.
    u = imgpts[:, 0]
    v = imgpts[:, 1]
    margin_w = img_size[0] * 0.2
    margin_h = img_size[1] * 0.2
    
    valid = (u >= -margin_w) & (u < img_size[0] + margin_w) & (v >= -margin_h) & (v < img_size[1] + margin_h)
    
    pts3d_valid = pts3d[valid]
    imgpts_valid = imgpts[valid]
    
    if len(pts3d_valid) < 100:
        print("Not enough valid points generated inside the sensor area.")
        return None, None, None, None
        
    print(f"Optimizing omnidir model using {len(pts3d_valid)} rays...")
    
    def residuals(params):
        fx, fy, cx, cy, xi, k1, k2, p1, p2 = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        D = np.array([k1, k2, p1, p2], dtype=np.float64)
        
        proj, _ = cv2.omnidir.projectPoints(pts3d_valid.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, float(xi), D)
        proj = proj.reshape(-1, 2)
        return (proj - imgpts_valid).flatten()
        
    # Start with the existing camera matrix, xi=0, and no distortion
    initial_guess = [
        camera_matrix[0,0], camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2],
        0.0, 0.0, 0.0, 0.0, 0.0
    ]
    
    res = least_squares(residuals, initial_guess, max_nfev=1000)
    
    fx, fy, cx, cy, xi, k1, k2, p1, p2 = res.x
    K_new = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D_new = np.array([k1, k2, p1, p2], dtype=np.float64)
    
    rms = np.sqrt(2 * res.cost / len(pts3d_valid))
    print(f"Synthetic calibration RMS error: {rms:.4f} pixels")
    
    return K_new, xi, D_new, rms

def process_data_dir(data_dir):
    seq_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("emulated_rgbd_")])
    if not seq_dirs:
        print("No sequence directories found.")
        return

    # Cache calibration results based on original parameters to avoid redundant work
    calib_cache = {}

    for seq_dir in seq_dirs:
        cam_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
        for d in cam_dirs:
            meta_path = d / "metadata.yaml"
            if not meta_path.exists(): continue
            
            with open(meta_path, "r") as f:
                metadata = yaml.safe_load(f)
                
            cam_mat = metadata.get("camera_matrix")
            dist_coeffs = metadata.get("distortion_coefficients")
            is_fisheye = metadata.get("is_fisheye", False)
            camera_model = metadata.get("camera_model", "fisheye" if is_fisheye else "pinhole")
            
            if cam_mat is None or dist_coeffs is None:
                continue
                
            if camera_model == "omnidir":
                # Already converted
                continue
                
            rgb_path = d / "rgb.png"
            if not rgb_path.exists(): continue
            rgb_img = cv2.imread(str(rgb_path))
            if rgb_img is None: continue
            
            h, w = rgb_img.shape[:2]
            img_size = (w, h)
            
            # Create a hashable key for the intrinsics
            cam_mat_arr = np.array(cam_mat)
            dist_arr = np.array(dist_coeffs)
            cache_key = (cam_mat_arr.tobytes(), dist_arr.tobytes(), is_fisheye, img_size)
            
            if cache_key in calib_cache:
                K, xi, D = calib_cache[cache_key]
            else:
                print(f"Converting intrinsics for camera {metadata.get('camera_name')}...")
                K, xi, D, rms = convert_to_omnidir(img_size, cam_mat_arr, dist_arr, is_fisheye)
                if K is None:
                    print("Failed to convert.")
                    continue
                calib_cache[cache_key] = (K, xi, D)
                
            # Backup original
            backup_path = d / "metadata.yaml.bak"
            if not backup_path.exists():
                shutil.copy(meta_path, backup_path)
                
            metadata["camera_model"] = "omnidir"
            metadata["omnidir_xi"] = float(xi)
            metadata["camera_matrix"] = K.tolist()
            metadata["distortion_coefficients"] = D.flatten().tolist()
            
            with open(meta_path, "w") as f:
                yaml.dump(metadata, f, default_flow_style=None)
                
    print("Synthetic calibration complete.")

def main():
    parser = argparse.ArgumentParser(description="Convert existing fisheye models to OpenCV omnidirectional model via synthetic calibration.")
    parser.add_argument("--data_path", type=str, default=None, help="Path to captured data directory.")
    args = parser.parse_args()
    
    data_path_str = args.data_path
    if data_path_str is None:
        data_path_str = find_latest_data_dir()
        if data_path_str is None:
            print("Error: Could not find any captured data directory. Please specify --data_path.")
            return
        print(f"Auto-selected latest data directory: {data_path_str}")
        
    data_dir = Path(data_path_str)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' not found.")
        return
        
    process_data_dir(data_dir)

if __name__ == "__main__":
    main()

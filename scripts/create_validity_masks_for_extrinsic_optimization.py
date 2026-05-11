#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml
import time
from datetime import datetime

from stretch4_emulated_rgbd.shared_utils import project_points, find_latest_data_dir, merge_lidar_points, generate_rgb_vignetting_mask, generate_depth_validity_mask
from stretch4_emulated_rgbd import emulated_rgbd_config as config


def _project_to_depth(pts_base, T_base_to_cam, camera_matrix, dist_coeffs, camera_model, xi, w, h):
    depth_img = np.zeros((h, w), dtype=np.float32)
    if len(pts_base) == 0 or camera_matrix is None:
        return depth_img
        
    ones = np.ones((len(pts_base), 1))
    pts_cam_all = (T_base_to_cam @ np.hstack([pts_base, ones]).T).T[:, :3]
    
    valid_idx = pts_cam_all[:, 2] > 0
    pts_cam_valid = pts_cam_all[valid_idx]
    
    if len(pts_cam_valid) == 0: return depth_img
    
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    img_pts = project_points(
        pts_cam_valid, rvec, tvec, camera_matrix, dist_coeffs, camera_model, xi
    ).reshape(-1, 2)
    
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
        depth_img[v_sorted, u_sorted] = z_sorted
        
    return depth_img


def generate_masks(data_path):
    data_dir = Path(data_path)
    seq_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("emulated_rgbd_")])
    if not seq_dirs:
        print("Error: No sequence directories found.")
        return

    print("Accumulating RGB images and LiDAR projections...")
    rgb_images = []
    depth_images = []
    
    # Process all sequences
    for seq_dir in seq_dirs:
        cam_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
        for d in cam_dirs:
            meta_path = d / "metadata.yaml"
            if not meta_path.exists(): continue
            
            with open(meta_path, "r") as f:
                metadata = yaml.safe_load(f)
                
            rgb_path = d / "rgb.png"
            if rgb_path.exists():
                rgb_img = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
                if rgb_img is not None:
                    rgb_images.append(rgb_img)
                    h, w = rgb_img.shape
            else:
                continue
            
            T_base_to_cam = np.array(metadata["T_base_to_cam"])
            T_lidar_to_base_left = np.array(metadata["T_lidar_to_base_left"])
            T_lidar_to_base_right = np.array(metadata["T_lidar_to_base_right"])
            camera_matrix = np.array(metadata.get("camera_matrix", []))
            dist_coeffs = np.array(metadata.get("distortion_coefficients", []))
            camera_model = metadata.get("camera_model", "fisheye" if metadata.get("is_fisheye", False) else "pinhole")
            xi = metadata.get("omnidir_xi", 0.0)
            
            if len(camera_matrix) == 0: continue
            
            l_pts = np.load(str(d / "lidar_left.npz"))["pts"] if (d / "lidar_left.npz").exists() else None
            r_pts = np.load(str(d / "lidar_right.npz"))["pts"] if (d / "lidar_right.npz").exists() else None
            pts_base = merge_lidar_points(l_pts, r_pts, T_lidar_to_base_left, T_lidar_to_base_right)
            
            if len(pts_base) == 0: 
                depth_images.append(np.zeros((h, w), dtype=np.float32))
                continue
            depth_img = _project_to_depth(pts_base, T_base_to_cam, camera_matrix, dist_coeffs, camera_model, xi, w, h)
            depth_images.append(depth_img)

    if not rgb_images or not depth_images:
        print("Error: Could not process images or LiDAR data.")
        return

    # Determine the camera and lidar names from metadata
    camera_name = "left" # Default fallback
    lidar_name = "left_lidar"
    if len(seq_dirs) > 0:
        cam_dirs = sorted([d for d in seq_dirs[0].iterdir() if d.is_dir()])
        if len(cam_dirs) > 0:
            meta_path = cam_dirs[0] / "metadata.yaml"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    metadata = yaml.safe_load(f)
                    camera_name = metadata.get("camera_name", "left")
            
            # Extract lidar name from directory name
            dir_parts = cam_dirs[0].name.split('_')
            for i, part in enumerate(dir_parts):
                if part == "camera" and i+2 < len(dir_parts) and dir_parts[i+2] == "lidar":
                    lidar_name = f"{dir_parts[i+1]}_lidar"
                    break

    # Determine target directory
    target_dir = Path(data_path) / "validity_masks_for_extrinsic_optimization"
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate RGB Vignetting Mask
    print("Generating RGB vignetting mask...")
    rgb_vignette_mask = generate_rgb_vignetting_mask(rgb_images)
    
    out_rgb_path = target_dir / f"rgb_vignette_mask_{camera_name}_camera.png"
    cv2.imwrite(str(out_rgb_path), rgb_vignette_mask)
    print(f"Saved RGB vignetting mask to: {out_rgb_path}")

    # 2. Generate Depth Valid Mask
    print("Generating Depth valid LiDAR mask...")
    clean_depth_valid_mask = generate_depth_validity_mask(depth_images)
        
    out_depth_path = target_dir / f"depth_valid_mask_{camera_name}_camera_{lidar_name}.png"
    cv2.imwrite(str(out_depth_path), clean_depth_valid_mask)
    print(f"Saved Depth valid mask to: {out_depth_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate RGB Vignetting and Depth LiDAR Validity Masks.")
    parser.add_argument("data_path", type=str, help="Path to the captured data directory.")
    args = parser.parse_args()

    data_path_str = args.data_path

    generate_masks(data_path_str)

if __name__ == "__main__":
    main()

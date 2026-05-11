#!/usr/bin/env python3
"""
estimate_lidar_gap.py

This script empirically measures the maximum distance between sparse LiDAR points 
across the collected emulated RGB-D data. It is highly useful for tuning the 
`MAX_LIDAR_INTERPOLATION_DIST_PX` parameter in `emulated_rgbd_config.py`.

It projects the valid LiDAR points into the camera frames, computes an OpenCV 
Voronoi distance transform, and specifically extracts the maximum distance values 
*strictly within the interior convex hull* of the LiDAR projection. This prevents 
measuring the empty space stretching out toward the unobserved corners of the camera.
"""

import sys
import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml

from stretch4_emulated_rgbd.shared_utils import project_points, find_latest_data_dir, merge_lidar_points


def main():
    parser = argparse.ArgumentParser(description="Estimate the maximum gap between sparse LiDAR lines in pixel space.")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the captured data directory.")
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
        
    print(f"\nAnalyzing LiDAR projection gaps in: {data_dir.name}...")
    
    # We look for all subdirectories that start with 'emulated_rgbd_'
    seq_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("emulated_rgbd_")])
    if not seq_dirs:
        print("Error: No 'emulated_rgbd_*' sequences found in the provided data directory.")
        return
        
    max_gap = 0.0
    gap_samples = []

    for seq_dir in seq_dirs:
        # Each sequence has multiple directories representing cameras or snapshots
        cam_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
        for d in cam_dirs:
            meta_path = d / "metadata.yaml"
            if not meta_path.exists(): continue
            
            with open(meta_path, "r") as f:
                metadata = yaml.safe_load(f)
                
            T_base_to_cam = np.array(metadata["T_base_to_cam"])
            T_lidar_to_base_left = np.array(metadata["T_lidar_to_base_left"])
            T_lidar_to_base_right = np.array(metadata["T_lidar_to_base_right"])
            camera_matrix = np.array(metadata.get("camera_matrix", []))
            dist_coeffs = np.array(metadata.get("distortion_coefficients", []))
            camera_model = metadata.get("camera_model", "fisheye" if metadata.get("is_fisheye", False) else "pinhole")
            xi = metadata.get("omnidir_xi", 0.0)
            
            if len(camera_matrix) == 0: continue
            
            rgb_img = cv2.imread(str(d / "rgb.png"))
            if rgb_img is None: continue
            h, w = rgb_img.shape[:2]
            
            # Merge left and right LiDAR
            l_pts = np.load(str(d / "lidar_left.npz"))["pts"] if (d / "lidar_left.npz").exists() else None
            r_pts = np.load(str(d / "lidar_right.npz"))["pts"] if (d / "lidar_right.npz").exists() else None
            pts_base = merge_lidar_points(l_pts, r_pts, T_lidar_to_base_left, T_lidar_to_base_right)
            if len(pts_base) == 0: continue
            
            # Project points into camera frame
            ones = np.ones((len(pts_base), 1))
            pts_cam_all = (T_base_to_cam @ np.hstack([pts_base, ones]).T).T[:, :3]
            
            valid_idx = pts_cam_all[:, 2] > 0
            pts_cam_valid = pts_cam_all[valid_idx]
            if len(pts_cam_valid) == 0: continue
            
            rvec = np.zeros(3)
            tvec = np.zeros(3)
            img_pts = project_points(
                pts_cam_valid, rvec, tvec, camera_matrix, dist_coeffs, camera_model, xi
            ).reshape(-1, 2)
            
            img_pts_int = np.round(img_pts).astype(int)
            u = img_pts_int[:, 0]
            v = img_pts_int[:, 1]
            
            # Filter points to those strictly within the image boundaries
            valid_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u_valid = u[valid_uv]
            v_valid = v[valid_uv]
            
            if len(u_valid) == 0: continue
            
            # Create a dense binary mask of valid LiDAR pixels
            depth_img = np.zeros((h, w), dtype=np.float32)
            depth_img[v_valid, u_valid] = 1.0 
            
            valid_mask = depth_img > 0
            mask = np.where(valid_mask, 0, 255).astype(np.uint8)
            
            # Compute distance to the nearest valid LiDAR pixel
            dist, _ = cv2.distanceTransformWithLabels(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
            
            # Constrain the search strictly to the convex hull of the points.
            # This prevents measuring the empty space stretching out to the camera corners.
            pts_2d = np.column_stack((u_valid, v_valid))
            hull = cv2.convexHull(pts_2d)
            hull_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(hull_mask, [hull], -1, 255, -1)
            
            # Erode hull slightly to avoid border effects where the gap naturally widens at the periphery
            kernel = np.ones((15, 15), np.uint8)
            hull_mask = cv2.erode(hull_mask, kernel, iterations=1)
            
            # Extract only the interior distances
            interior_dist = dist[hull_mask > 0]
            if len(interior_dist) > 0:
                frame_max = np.max(interior_dist)
                gap_samples.append(frame_max)
                if frame_max > max_gap:
                    max_gap = frame_max
                    
    if gap_samples:
        print("\n--- LiDAR Gap Analysis Results ---")
        print(f"Total frames analyzed: {len(gap_samples)}")
        print(f"Average internal gap (pixels): {np.mean(gap_samples):.2f}")
        print(f"Median internal gap  (pixels): {np.median(gap_samples):.2f}")
        print(f"Maximum internal gap (pixels): {np.max(gap_samples):.2f}")
        print("\nRecommendation:")
        print(f"Set MAX_LIDAR_INTERPOLATION_DIST_PX in emulated_rgbd_config.py to a value slightly")
        print(f"larger than the maximum internal gap ({np.max(gap_samples):.2f}) to ensure dense depth fills completely.")
    else:
        print("Error: Could not calculate gaps. No valid LiDAR projections were found.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import yaml
from pathlib import Path
import time
import argparse

from stretch4_emulated_rgbd.shared_utils import DenseDepthImage, project_points, merge_lidar_points, find_latest_data_dir, get_timestamp_from_name, ExtrinsicsCalibration
from stretch4_emulated_rgbd import emulated_rgbd_config as config


def find_latest_opt_yaml():
    data_dir = Path("data")
    if not data_dir.exists():
        return None
        
    yaml_files = list(data_dir.rglob("optimization_results_*.yaml"))
    if not yaml_files:
        return None
        
    yaml_files.sort(key=lambda f: get_timestamp_from_name(f.name))
    return str(yaml_files[-1])

def reconstruct_and_log(prefix, T_base_to_cam, seq, current_time, rgb_vignette_mask=None, depth_valid_mask=None):
    # Merge LiDAR points
    pts_base = merge_lidar_points(
        seq.get("l_pts"), 
        seq.get("r_pts"), 
        seq.get("T_lidar_to_base_left"), 
        seq.get("T_lidar_to_base_right")
    )
        
    rgb_img = seq["rgb"]
    h, w = rgb_img.shape[:2]
    depth_img = np.zeros((h, w), dtype=np.float32)
    
    if len(pts_base) > 0 and seq.get("camera_matrix") is not None:
        ones = np.ones((len(pts_base), 1))
        pts_cam_all = (T_base_to_cam @ np.hstack([pts_base, ones]).T).T[:, :3]
        
        valid_idx = pts_cam_all[:, 2] > 0
        pts_cam_valid = pts_cam_all[valid_idx]
        
        if len(pts_cam_valid) > 0:
            rvec = np.zeros(3)
            tvec = np.zeros(3)
            c_model = seq.get("camera_model", "fisheye" if seq.get("is_fisheye", False) else "pinhole")
            xi = seq.get("omnidir_xi", 0.0)
            img_pts = project_points(
                pts_cam_valid, rvec, tvec, seq["camera_matrix"], seq["dist_coeffs"], c_model, xi
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
                
    # Logging
    rr.set_time("timestamp", timestamp=current_time)
    
    lidar_name = "both_lidar"
    if seq.get("l_pts") is not None and seq.get("r_pts") is None:
        lidar_name = "left_lidar"
    elif seq.get("r_pts") is not None and seq.get("l_pts") is None:
        lidar_name = "right_lidar"
        
    image_bgr = rgb_img.copy()
    c_name = seq["camera_name"]
    
    # The images and point clouds are already correctly oriented (upright)
    # when loaded from disk or projected using the upright camera matrix.
    # No further rotation is needed.
            
    dense_processor = DenseDepthImage(
        image_bgr, 
        depth_img if len(pts_base) > 0 else None, 
        apply_validity_mask=False, 
        camera_name=c_name, 
        lidar_name=lidar_name
    )
    dd = dense_processor.compute_dense_depth()
    
    if dd is not None and rgb_vignette_mask is not None and depth_valid_mask is not None:
        valid_idx = rgb_vignette_mask & depth_valid_mask
        dd[~valid_idx] = 0.0
        
    rr.log(f"{prefix}/{c_name}/rgb", rr.Image(image_bgr, color_model="BGR").compress())
    
    if np.any(depth_img > 0):
        rr.log(f"{prefix}/{c_name}/sparse_depth", rr.DepthImage(depth_img, meter=1.0, depth_range=[0.0, config.RERUN_COLOR_MAX_DEPTH_M]))
        
    if dd is not None:
        rr.log(f"{prefix}/{c_name}/dense_depth", rr.DepthImage(dd, meter=1.0, depth_range=[0.0, config.RERUN_COLOR_MAX_DEPTH_M]))

def replay_sequence(data_path, calibration: ExtrinsicsCalibration):
    print(f"Replaying from {data_path}")
    base_dir = Path(data_path)
    seq_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("emulated_rgbd_")])
    
    if not seq_dirs:
        print("No sequence directories found.")
        return
        
    current_replay_time = 0.0
    masks_loaded = False
    rgb_vignette_mask = None
    depth_valid_mask = None
    
    for seq_dir in seq_dirs:
        distinct_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
        for d in distinct_dirs:
            meta_path = d / "metadata.yaml"
            if not meta_path.exists(): continue
            
            with open(meta_path, "r") as f:
                metadata = yaml.safe_load(f)
                
            rgb_img = cv2.imread(str(d / "rgb.png"))
            if rgb_img is None: continue
            
            seq = {
                "camera_name": metadata["camera_name"],
                "T_base_to_cam": np.array(metadata["T_base_to_cam"]),
                "T_lidar_to_base_left": np.array(metadata["T_lidar_to_base_left"]),
                "T_lidar_to_base_right": np.array(metadata["T_lidar_to_base_right"]),
                "camera_matrix": np.array(metadata["camera_matrix"]) if metadata.get("camera_matrix") else None,
                "dist_coeffs": np.array(metadata["distortion_coefficients"]) if metadata.get("distortion_coefficients") else None,
                "is_fisheye": metadata.get("is_fisheye", False),
                "camera_model": metadata.get("camera_model", "fisheye" if metadata.get("is_fisheye", False) else "pinhole"),
                "omnidir_xi": metadata.get("omnidir_xi", 0.0),
                "rgb": rgb_img,
                "l_pts": np.load(str(d / "lidar_left.npz"))["pts"] if (d / "lidar_left.npz").exists() else None,
                "r_pts": np.load(str(d / "lidar_right.npz"))["pts"] if (d / "lidar_right.npz").exists() else None
            }
            
            if not masks_loaded:
                lidar_name = "both_lidar"
                if seq.get("l_pts") is not None and seq.get("r_pts") is None:
                    lidar_name = "left_lidar"
                elif seq.get("r_pts") is not None and seq.get("l_pts") is None:
                    lidar_name = "right_lidar"
                    
                from stretch4_emulated_rgbd.shared_utils import load_and_confirm_optimization_masks
                rgb_vignette_mask, depth_valid_mask = load_and_confirm_optimization_masks(
                    data_path, seq["camera_name"], lidar_name
                )
                masks_loaded = True
                
            # Original unoptimized transform (prior optimized)
            reconstruct_and_log("Unoptimized", seq["T_base_to_cam"], seq, current_replay_time, rgb_vignette_mask, depth_valid_mask)
            
            from stretch4_emulated_rgbd.shared_utils import get_rotated_extrinsics
            
            # Extract unrotated factory baseline
            T_fac_list = metadata.get("T_base_to_cam_factory")
            if T_fac_list is not None:
                T_base_to_cam_factory_unrot = np.array(T_fac_list)
            else:
                T_base_to_cam_factory_unrot = get_rotated_extrinsics(seq["T_base_to_cam"], is_clockwise=not (seq["camera_name"] == "right"))

            # Apply correction to the unrotated baseline, then rotate to match image
            T_base_to_cam_opt_unrot = calibration.apply_to_camera_extrinsics(T_base_to_cam_factory_unrot)
            T_base_to_cam_new = get_rotated_extrinsics(T_base_to_cam_opt_unrot, is_clockwise=(seq["camera_name"] == "right"))
            
            reconstruct_and_log("Optimized", T_base_to_cam_new, seq, current_replay_time, rgb_vignette_mask, depth_valid_mask)
            
        current_replay_time += 1.0
        time.sleep(0.1)
        
    rr.set_time("timestamp", timestamp=current_replay_time)
    rr.log("TimelineEnd", rr.TextDocument("Replay Complete"))

def main():
    parser = argparse.ArgumentParser(description="Compare Unoptimized vs Optimized RGB-D alignments in ReRun")
    parser.add_argument("--data_path", type=str, default=None, help="Path to captured data directory to replay")
    parser.add_argument("--opt_yaml", type=str, default=None, help="Path to optimization YAML file")
    args = parser.parse_args()
    
    data_path = args.data_path
    if data_path is None:
        data_path = find_latest_data_dir()
        if data_path is None:
            print("Error: Could not find any captured data directory.")
            return
        print(f"Auto-selected latest data directory: {data_path}")
        
    opt_yaml = args.opt_yaml
    if opt_yaml is None:
        opt_yaml = find_latest_opt_yaml()
        if opt_yaml is None:
            print("Error: Could not find any optimization YAML file.")
            return
        print(f"Auto-selected latest optimization file: {opt_yaml}")
        
    calibration = ExtrinsicsCalibration.load_from_yaml(opt_yaml)
    if calibration is None:
        return
        
    print("Initializing ReRun...")
    rr.init("Stretch Optimization Comparison", spawn=False)
    rr.spawn(memory_limit="2GiB")
    
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Unoptimized", origin="Unoptimized"),
            rrb.Spatial2DView(name="Optimized", origin="Optimized"),
        ),
        rrb.BlueprintPanel(expanded=True),
        rrb.SelectionPanel(expanded=True),
        rrb.TimePanel(expanded=True, play_state="following"),
    )
    rr.send_blueprint(blueprint)
    
    replay_sequence(data_path, calibration)
    print("Comparison replay finished.")

if __name__ == "__main__":
    main()

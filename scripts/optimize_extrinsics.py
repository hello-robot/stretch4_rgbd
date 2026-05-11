#!/usr/bin/env python3
import os
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import yaml
import time
import argparse
import platform
import subprocess
from pathlib import Path
from scipy.spatial.transform import Rotation
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cma

from stretch4_emulated_rgbd import emulated_rgbd_config as config

# Apply optimize_extrinsics specific shadow filter overrides
config.ENABLE_SHADOW_FILTER = config.OPTIMIZE_EXTRINSICS_ENABLE_SHADOW_FILTER
config.SHADOW_FILTER_WINDOW_SIZE = config.OPTIMIZE_EXTRINSICS_SHADOW_FILTER_WINDOW_SIZE
config.SHADOW_FILTER_DEPTH_THRESHOLD_M = config.OPTIMIZE_EXTRINSICS_SHADOW_FILTER_DEPTH_THRESHOLD_M

from stretch4_emulated_rgbd.shared_utils import (
    render_rgbd, reconstruct_rgbd_frame, DenseDepthImage, 
    get_vignette_mask, get_saturation_mask, project_points, unproject_points, merge_lidar_points
)

def get_sys_info():
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }
    if platform.system() == "Linux":
        try:
            cpu_info = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().strip()
            info["cpu_model"] = cpu_info.split(":")[1].strip()
        except:
            pass
    return info

class ExtrinsicOptimizer:
    def __init__(self, data_path, camera="left", lidar="left", visualize=False, debugging=False):
        self.data_path = Path(data_path)
        self.camera_name = camera
        self.lidar_name = f"{lidar}_lidar" if not lidar.endswith("_lidar") else lidar
        self.ignore_prior_optimizations = config.EXTRINSICS_IGNORE_PRIOR_OPTIMIZATIONS
        self.visualize = visualize
        self.debugging = debugging
        self.use_nmi = config.EXTRINSICS_USE_NMI
        self.grad_ksize = config.EXTRINSICS_GRAD_KSIZE
        self.sequences = []
        self.capture_fleet_id = "UNKNOWN"
        
        self.best_cost = float('inf')
        self.best_delta = np.zeros(6)
        self.current_replay_time = 0.0
        self.interim_save_counter = 0
        self.interim_save_context = None
        
        if self.visualize:
            rr.init("Stretch Optimizer", spawn=True)
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Unoptimized", origin="Unoptimized"),
                    rrb.Spatial2DView(name="Optimized", origin="Optimized"),
                    rrb.Spatial3DView(
                        name="Optimized 3D Map", 
                        origin="Optimized_3D_Map",
                        contents=[
                            "+ Optimized_3D_Map/**",
                        ],
                        overrides={
                            "Optimized_3D_Map/left_lidar_cloud_with_rgb_colors": [rr.components.Visible(False)],
                            "Optimized_3D_Map/right_lidar_cloud_with_rgb_colors": [rr.components.Visible(False)],
                            "Optimized_3D_Map/center_lidar_cloud_with_rgb_colors": [rr.components.Visible(False)]
                        }
                    ),
                ),
                rrb.BlueprintPanel(expanded=False),
                rrb.SelectionPanel(expanded=False),
                rrb.TimePanel(expanded=True),
            )
            rr.send_blueprint(blueprint)
            
        self._load_data()
        self._precompute()
        
    def set_interim_save_context(self, args, start_time, initial_cost, get_num_iterations_fn):
        self.interim_save_context = {
            "args": args,
            "start_time": start_time,
            "initial_cost": initial_cost,
            "get_num_iterations": get_num_iterations_fn,
        }

    def _save_interim_results(self):
        if self.interim_save_context is None: return
        
        ctx = self.interim_save_context
        args = ctx["args"]
        duration = time.time() - ctx["start_time"]
        num_iterations = ctx["get_num_iterations"]()
        
        best_delta = self.best_delta
        t = best_delta[:3]
        r = Rotation.from_rotvec(best_delta[3:6]).as_matrix()
        T_delta = np.eye(4)
        T_delta[:3, :3] = r
        T_delta[:3, 3] = t
        
        suffix = "A" if self.interim_save_counter % 2 == 0 else "B"
        self.interim_save_counter += 1
        
        yaml_filename = f"INCOMPLETE_optimization_results_mi_rgb_{suffix}.yaml"
        yaml_path = self.data_path / yaml_filename
        
        result_data = {
            "metadata": {
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "duration_seconds": duration,
                "data_path": str(args.data_path),
                "num_sequences_evaluated": len(self.sequences),
                "status": "INCOMPLETE",
                "capture_fleet_id": self.capture_fleet_id,
            },
            "system_profile": get_sys_info(),
            "hyperparameters": {
                "method": "mi_rgb",
                "translation_sigma_m": config.EXTRINSICS_CMA_TRANSLATION_SIGMA_M,
                "rotation_sigma_deg": config.EXTRINSICS_CMA_ROTATION_SIGMA_DEG,
                "popsize": config.EXTRINSICS_CMA_POPSIZE,
                "maxiter": config.EXTRINSICS_CMA_MAXITER,
                "num_iterations_executed": num_iterations,
                "use_nmi": config.EXTRINSICS_USE_NMI,
                "grad_ksize": config.EXTRINSICS_GRAD_KSIZE,
            },
            "convergence": {
                "initial_cost": float(ctx["initial_cost"]),
                "current_cost": float(self.best_cost),
            },
            "results": {
                "best_delta_transform_array": best_delta.tolist(),
                "best_delta_transform_matrix": T_delta.tolist(),
            }
        }
        
        with open(yaml_path, "w") as f:
            yaml.dump(result_data, f, default_flow_style=None)

    def _load_data(self):
        print(f"Loading data from {self.data_path} for camera '{self.camera_name}' and lidar '{self.lidar_name}'...")
        
        capture_meta_path = self.data_path / "capture_metadata.yaml"
        if capture_meta_path.exists():
            with open(capture_meta_path, "r") as f:
                c_meta = yaml.safe_load(f) or {}
                self.capture_fleet_id = c_meta.get("capture_fleet_id", "UNKNOWN")
        
        seq_dirs = sorted([d for d in self.data_path.iterdir() if d.is_dir() and d.name.startswith("emulated_rgbd_")])
        if not seq_dirs:
            print("No sequence directories found!")
            return
            
        from stretch4_emulated_rgbd.shared_utils import CapturedSequence
        for seq_dir in seq_dirs:
            distinct_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
            for d in distinct_dirs:
                meta_path = d / "metadata.yaml"
                if not meta_path.exists(): continue
                
                if f"{self.camera_name}_camera_{self.lidar_name}" not in d.name: continue
                
                try:
                    seq = CapturedSequence.load(d)
                except Exception as e:
                    print(f"Failed to load sequence {d}: {e}")
                    continue
                    
                pts_base = merge_lidar_points(seq.raw_lidar_left, seq.raw_lidar_right, seq.frame.T_lidar_to_base_left, seq.frame.T_lidar_to_base_right)
                
                from stretch4_emulated_rgbd.shared_utils import get_rotated_extrinsics
                
                # Fetch factory and optimized calibrations
                T_base_to_cam_rotated = seq.frame.T_base_to_cam
                
                # The optimizer MUST search in the unrotated (native URDF) frame.
                # If we don't have the unrotated factory matrix, we un-rotate the rotated one.
                T_fac_list = seq.metadata.get("T_base_to_cam_factory")
                if T_fac_list is not None:
                    T_base_to_cam_factory_unrot = np.array(T_fac_list)
                else:
                    # Inverse of get_rotated_extrinsics
                    T_base_to_cam_factory_unrot = get_rotated_extrinsics(T_base_to_cam_rotated, is_clockwise=not (self.camera_name == "right"))
                
                T_base_to_cam_opt_list = seq.metadata.get("T_base_to_cam_optimized")
                T_base_to_cam_optimized_unrot = np.array(T_base_to_cam_opt_list) if T_base_to_cam_opt_list is not None else None
                
                self.sequences.append({
                    "path": d,
                    "metadata": seq.metadata,
                    "rgb": seq.frame.image,
                    "pts_base": pts_base,
                    "l_pts": seq.raw_lidar_left,
                    "r_pts": seq.raw_lidar_right,
                    "camera_matrix": seq.frame.camera_matrix, # Rotated
                    "dist_coeffs": seq.frame.distortion_coefficients,
                    "camera_model": seq.metadata.get("camera_model", "fisheye" if seq.metadata.get("is_fisheye", False) else "pinhole"),
                    "omnidir_xi": seq.metadata.get("omnidir_xi", 0.0),
                    "is_fisheye": seq.metadata.get("is_fisheye", False),
                    "T_base_to_cam_rotated": T_base_to_cam_rotated,
                    "T_base_to_cam_factory_unrot": T_base_to_cam_factory_unrot,
                    "T_base_to_cam_optimized_unrot": T_base_to_cam_optimized_unrot,
                    "T_lidar_to_base_left": seq.frame.T_lidar_to_base_left,
                    "T_lidar_to_base_right": seq.frame.T_lidar_to_base_right,
                    "is_clockwise": (self.camera_name == "right")
                })
        print(f"Loaded {len(self.sequences)} valid RGB-D sequence frames.")
        
        if not self.sequences:
            raise ValueError("No matching sequences found!")
            
        # Determine initial x0 for CMA-ES from prior optimum
        self.initial_x0 = np.zeros(6)
        if not self.ignore_prior_optimizations:
            sample_seq = self.sequences[0]
            if sample_seq["T_base_to_cam_optimized_unrot"] is not None:
                print("Found prior optimization. Calculating initial x0 for warm start...")
                T_opt_unrot = sample_seq["T_base_to_cam_optimized_unrot"]
                T_fac_unrot = sample_seq["T_base_to_cam_factory_unrot"]
                # T_opt = inv(T_delta) @ T_fac  => T_delta = T_fac @ inv(T_opt)
                # This math operates entirely in the unrotated URDF frame!
                T_delta_prior = T_fac_unrot @ np.linalg.inv(T_opt_unrot)
                t_prior = T_delta_prior[:3, 3]
                r_prior = Rotation.from_matrix(T_delta_prior[:3, :3]).as_rotvec()
                self.initial_x0 = np.concatenate([t_prior, r_prior])
                print(f"Warm start x0: {self.initial_x0}")
        
        from stretch4_emulated_rgbd.shared_utils import load_and_confirm_optimization_masks
        self.rgb_vignette_mask, self.depth_valid_mask = load_and_confirm_optimization_masks(
            self.data_path, self.camera_name, self.lidar_name
        )
        
        # We no longer logically AND the masks here. They are eroded and handled separately in _precompute.

    def _precompute(self):
        if not self.sequences: return
        print("Precomputing targets for mi_rgb...")
        
        # Erode the individual validity masks by the Sobel kernel size so the gradients we select
        # are mathematically guaranteed to not contain any artifacts from the mask boundaries.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.grad_ksize, self.grad_ksize))
        self.eroded_rgb_mask = cv2.erode(self.rgb_vignette_mask.astype(np.uint8), kernel) > 0
        self.eroded_depth_mask = cv2.erode(self.depth_valid_mask.astype(np.uint8), kernel) > 0
        
        # We restrict the mutual information calculation using the intersection of the eroded masks
        self.eroded_intersection_mask = self.eroded_rgb_mask & self.eroded_depth_mask
        
        for seq in self.sequences:
            seq["rgb_gray"] = cv2.cvtColor(seq["rgb"], cv2.COLOR_BGR2GRAY)
            seq["rgb_grad_x"] = cv2.Sobel(seq["rgb_gray"], cv2.CV_32F, 1, 0, ksize=self.grad_ksize)
            seq["rgb_grad_y"] = cv2.Sobel(seq["rgb_gray"], cv2.CV_32F, 0, 1, ksize=self.grad_ksize)
                
        print("Precomputation complete.")

    def _get_T_delta(self, delta):
        t = delta[:3]
        r = Rotation.from_rotvec(delta[3:6]).as_matrix()
        T_delta = np.eye(4)
        T_delta[:3, :3] = r
        T_delta[:3, 3] = t
        return T_delta

    def _project_to_depth(self, pts_base, seq, T_delta=None):
        from stretch4_emulated_rgbd.shared_utils import get_rotated_extrinsics
        
        T_base_to_cam_factory_unrot = seq["T_base_to_cam_factory_unrot"]
        
        if T_delta is not None:
            # T_delta is in the unrotated URDF frame!
            T_delta_inv = np.linalg.inv(T_delta)
            T_base_to_cam_unrot_new = T_delta_inv @ T_base_to_cam_factory_unrot
        else:
            T_base_to_cam_unrot_new = T_base_to_cam_factory_unrot
            
        # Rotate the resulting extrinsics to match the rotated camera matrix & image
        T_base_to_cam_rotated = get_rotated_extrinsics(T_base_to_cam_unrot_new, is_clockwise=seq["is_clockwise"])
            
        cam_mat = seq["camera_matrix"] # This is rotated
        d_coeffs = seq["dist_coeffs"]
        c_model = seq.get("camera_model", "fisheye" if seq.get("is_fisheye", False) else "pinhole")
        current_xi = seq.get("omnidir_xi", 0.0)
        rgb_img = seq["rgb"]
        h, w = rgb_img.shape[:2]
        
        depth_img = np.zeros((h, w), dtype=np.float32)
        if len(pts_base) == 0 or cam_mat is None:
            return depth_img
            
        ones = np.ones((len(pts_base), 1))
        pts_cam_all = (T_base_to_cam_rotated @ np.hstack([pts_base, ones]).T).T[:, :3]
        
        valid_idx = pts_cam_all[:, 2] > 0
        pts_cam_valid = pts_cam_all[valid_idx]
        
        if len(pts_cam_valid) == 0: return depth_img
        
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        img_pts = project_points(
            pts_cam_valid, rvec, tvec, cam_mat, d_coeffs, c_model, current_xi
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

    def _compute_mutual_information(self, img1, img2, bins=None):
        if bins is None:
            bins = config.EXTRINSICS_MI_BINS
            
        # Use percentiles to robustly determine histogram ranges and ignore massive outliers
        p1_min, p1_max = np.percentile(img1, [1, 99])
        p2_min, p2_max = np.percentile(img2, [1, 99])
        
        # Add a tiny epsilon to avoid zero-width ranges if the image is flat
        p1_max = max(p1_max, p1_min + 1e-5)
        p2_max = max(p2_max, p2_min + 1e-5)
        
        hist_2d, _, _ = np.histogram2d(
            img1.ravel(), img2.ravel(), 
            bins=bins, 
            range=[[p1_min, p1_max], [p2_min, p2_max]]
        )
        pxy = hist_2d / float(max(1, np.sum(hist_2d)))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        
        if getattr(self, "use_nmi", False):
            px_nz = px[px > 0]
            py_nz = py[py > 0]
            hx = -np.sum(px_nz * np.log(px_nz))
            hy = -np.sum(py_nz * np.log(py_nz))
            if hx + hy > 0:
                return 2.0 * mi / (hx + hy)
            return 0.0
            
        return mi

    def cost_function(self, delta):
        t_mag = np.linalg.norm(delta[:3])
        r_mag = np.linalg.norm(delta[3:6]) * (180.0 / np.pi)
        
        penalty = 0.0
        if t_mag > config.MAX_TRANSLATION_M: 
            penalty += (t_mag - config.MAX_TRANSLATION_M) * config.BOUNDS_PENALTY_WEIGHT
        if r_mag > config.MAX_ROTATION_DEG: 
            penalty += (r_mag - config.MAX_ROTATION_DEG) * config.BOUNDS_PENALTY_WEIGHT
            
        total_cost = 0.0
        frame_costs = []
        
        for seq in self.sequences:
            T_delta = self._get_T_delta(delta)
            lidar_depth = self._project_to_depth(seq["pts_base"], seq, T_delta)
            
            valid_mask = (lidar_depth > 0) & self.rgb_vignette_mask
            if not np.any(valid_mask):
                total_cost += 1e6
                frame_costs.append(1e6)
                continue
                
            frame_cost = 0.0
            dense_processor = DenseDepthImage(seq["rgb"], lidar_depth, apply_validity_mask=False)
            lidar_dense = dense_processor.compute_dense_depth()
            if lidar_dense is None:
                total_cost += 1e6
                frame_costs.append(1e6)
                continue
                
            # We never zero out the actual depth or RGB arrays here to prevent 
            # artificial step-edges from dominating the Sobel gradients.
            lidar_grad_x = cv2.Sobel(lidar_dense, cv2.CV_32F, 1, 0, ksize=self.grad_ksize)
            lidar_grad_y = cv2.Sobel(lidar_dense, cv2.CV_32F, 0, 1, ksize=self.grad_ksize)
            
            # Erode the dynamic lidar validity mask by the Sobel kernel size to prevent 
            # artificial gradients at the boundaries of missing LiDAR data from corrupting the MI.
            dynamic_lidar_valid = (lidar_dense > 0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.grad_ksize, self.grad_ksize))
            eroded_dynamic_lidar = cv2.erode(dynamic_lidar_valid, kernel) > 0
            
            valid_dense_mask = eroded_dynamic_lidar & self.eroded_intersection_mask
            
            # Ensure the overlap is still substantial. If it drops below 50% of the max possible overlap, penalize it.
            overlap_ratio = np.count_nonzero(valid_dense_mask) / float(max(1, np.count_nonzero(self.eroded_intersection_mask)))
            if overlap_ratio < 0.5:
                total_cost += 1e6
                frame_costs.append(1e6)
                continue
            
            if self.debugging:
                rgb_gx = np.zeros_like(seq["rgb_grad_x"])
                rgb_gx[valid_dense_mask] = np.abs(seq["rgb_grad_x"][valid_dense_mask])
                lidar_gx = np.zeros_like(lidar_grad_x)
                lidar_gx[valid_dense_mask] = np.abs(lidar_grad_x[valid_dense_mask])
                max_rgb_x = np.max(rgb_gx) if np.max(rgb_gx) > 0 else 1.0
                max_lidar_x = np.max(lidar_gx) if np.max(lidar_gx) > 0 else 1.0
                rgb_gx_vis = (rgb_gx / max_rgb_x).astype(np.float32)
                lidar_gx_vis = (lidar_gx / max_lidar_x).astype(np.float32)
                
                rgb_gy = np.zeros_like(seq["rgb_grad_y"])
                rgb_gy[valid_dense_mask] = np.abs(seq["rgb_grad_y"][valid_dense_mask])
                lidar_gy = np.zeros_like(lidar_grad_y)
                lidar_gy[valid_dense_mask] = np.abs(lidar_grad_y[valid_dense_mask])
                max_rgb_y = np.max(rgb_gy) if np.max(rgb_gy) > 0 else 1.0
                max_lidar_y = np.max(lidar_gy) if np.max(lidar_gy) > 0 else 1.0
                rgb_gy_vis = (rgb_gy / max_rgb_y).astype(np.float32)
                lidar_gy_vis = (lidar_gy / max_lidar_y).astype(np.float32)
                
                vis_x = np.hstack([rgb_gx_vis, lidar_gx_vis])
                vis_y = np.hstack([rgb_gy_vis, lidar_gy_vis])
                
                cv2.imshow("Gradients X (Left: RGB, Right: LiDAR)", vis_x)
                cv2.imshow("Gradients Y (Left: RGB, Right: LiDAR)", vis_y)
                
                vis_mask = valid_dense_mask.astype(np.float32)
                cv2.imshow("Valid MI Mask", vis_mask)
                cv2.waitKey(1)

            mi_x = self._compute_mutual_information(lidar_grad_x[valid_dense_mask], seq["rgb_grad_x"][valid_dense_mask])
            mi_y = self._compute_mutual_information(lidar_grad_y[valid_dense_mask], seq["rgb_grad_y"][valid_dense_mask])
            frame_cost = -(mi_x + mi_y)
            
            total_cost += frame_cost
            frame_costs.append(frame_cost)

        avg_cost = total_cost / len(self.sequences) + penalty
        
        if avg_cost < self.best_cost:
            self.best_cost = avg_cost
            self.best_delta = delta.copy()
            self._print_and_visualize_best(frame_costs)
                
        return avg_cost

    def _print_and_visualize_best(self, frame_costs):
        t_mag_mm = np.linalg.norm(self.best_delta[:3]) * 1000
        r_mag_deg = np.linalg.norm(self.best_delta[3:6]) * (180.0 / np.pi)
        
        print(f"\n--- New Best Correction ---")
        print(f"Cost: {self.best_cost:.4f} | Trans: {t_mag_mm:.2f} mm | Rot: {r_mag_deg:.2f} deg")
            
        for i, (seq, cost) in enumerate(zip(self.sequences, frame_costs)):
            c_name = seq["metadata"]["camera_name"]
            seq_name = seq["path"].name
            print(f"  [{i}] {seq_name} ({c_name}): {cost:.4f}")
            
        self._save_interim_results()
            
        if not self.visualize:
            return
            
        print("Updating ReRun visualization...")
        t = self.best_delta[:3]
        r = Rotation.from_rotvec(self.best_delta[3:6]).as_matrix()
        T_delta = np.eye(4)
        T_delta[:3, :3] = r
        T_delta[:3, 3] = t
        
        # Images are already vertically oriented via reconstruct_rgbd_frame
        for seq in self.sequences:
            rr.set_time("timestamp", timestamp=self.current_replay_time)
            c_name = seq["metadata"]["camera_name"]
            c_model = seq.get("camera_model", "fisheye" if seq.get("is_fisheye", False) else "pinhole")
            
            # --- Unoptimized View ---
            unopt_camera_matrix = seq["camera_matrix"]
            unopt_dist_coeffs = seq["dist_coeffs"]
            unopt_xi = seq.get("omnidir_xi", 0.0)
            
            from stretch4_emulated_rgbd.shared_utils import get_rotated_extrinsics
            T_delta_initial = self._get_T_delta(self.initial_x0)
            T_delta_initial_inv = np.linalg.inv(T_delta_initial)
            T_base_to_cam_unrot_initial = T_delta_initial_inv @ seq["T_base_to_cam_factory_unrot"]
            T_base_to_cam_rotated_initial = get_rotated_extrinsics(T_base_to_cam_unrot_initial, is_clockwise=seq["is_clockwise"])
            
            unopt_frame = reconstruct_rgbd_frame(
                c_name, T_base_to_cam_rotated_initial, seq["T_lidar_to_base_left"], seq["T_lidar_to_base_right"],
                unopt_camera_matrix, unopt_dist_coeffs, c_model, unopt_xi, seq["rgb"],
                seq["l_pts"], seq["r_pts"], self.current_replay_time
            )
            
            image_bgr = unopt_frame.image.copy()
            rr.log(f"Unoptimized/{c_name}/rgb", rr.Image(image_bgr, color_model="BGR").compress())
            
            if unopt_frame.depth_image is not None and unopt_frame.depth_image.shape[0] > 0:
                depth_vis = unopt_frame.depth_image
                rr.log(f"Unoptimized/{c_name}/sparse_depth", rr.DepthImage(depth_vis, meter=1.0, depth_range=[0.0, config.RERUN_COLOR_MAX_DEPTH_M]))
                
                dense_processor = DenseDepthImage(unopt_frame.image, unopt_frame.depth_image, apply_validity_mask=False)
                dd = dense_processor.compute_dense_depth()
                if dd is not None:
                    rr.log(f"Unoptimized/{c_name}/dense_depth", rr.DepthImage(dd, meter=1.0, depth_range=[0.0, config.RERUN_COLOR_MAX_DEPTH_M]))

            # --- Optimized View ---
            T_delta = self._get_T_delta(self.best_delta)
            T_delta_inv = np.linalg.inv(T_delta)
            
            from stretch4_emulated_rgbd.shared_utils import get_rotated_extrinsics
            T_base_to_cam_unrot_new = T_delta_inv @ seq["T_base_to_cam_factory_unrot"]
            T_base_to_cam_new = get_rotated_extrinsics(T_base_to_cam_unrot_new, is_clockwise=seq["is_clockwise"])
            
            opt_camera_matrix = seq["camera_matrix"]
            opt_dist_coeffs = seq["dist_coeffs"]
            opt_xi = seq.get("omnidir_xi", 0.0)
            
            opt_frame = reconstruct_rgbd_frame(
                c_name, T_base_to_cam_new, seq["T_lidar_to_base_left"], seq["T_lidar_to_base_right"],
                opt_camera_matrix, opt_dist_coeffs, c_model, opt_xi, seq["rgb"],
                seq["l_pts"], seq["r_pts"], self.current_replay_time
            )
            
            rr.log(f"Optimized/{c_name}/rgb", rr.Image(image_bgr, color_model="BGR").compress())
            
            if opt_frame.depth_image is not None and opt_frame.depth_image.shape[0] > 0:
                depth_vis = opt_frame.depth_image
                rr.log(f"Optimized/{c_name}/sparse_depth", rr.DepthImage(depth_vis, meter=1.0, depth_range=[0.0, config.RERUN_COLOR_MAX_DEPTH_M]))
                
                dense_processor = DenseDepthImage(opt_frame.image, opt_frame.depth_image, apply_validity_mask=False)
                dd = dense_processor.compute_dense_depth()
                if dd is not None:
                    rr.log(f"Optimized/{c_name}/dense_depth", rr.DepthImage(dd, meter=1.0, depth_range=[0.0, config.RERUN_COLOR_MAX_DEPTH_M]))
            
            # Define an AnnotationContext so the segmentation masks render transparently over the RGB image
            ctx = rr.AnnotationContext([
                (0, "Valid Region", (0, 0, 0, 0)),
                (1, "Eroded RGB Area", (255, 0, 0, 128)),
                (2, "Eroded Depth Area", (255, 165, 0, 128))
            ])
            rr.log(f"Optimized/{c_name}", ctx)
            
            # Overlay the inverted eroded masks as semi-transparent segmentation layers
            seg_rgb_mask = np.zeros(self.eroded_rgb_mask.shape, dtype=np.uint8)
            seg_rgb_mask[~self.eroded_rgb_mask] = 1
            rr.log(f"Optimized/{c_name}/inverted_eroded_rgb_mask", rr.SegmentationImage(seg_rgb_mask))
            
            seg_depth_mask = np.zeros(self.eroded_depth_mask.shape, dtype=np.uint8)
            seg_depth_mask[~self.eroded_depth_mask] = 2
            rr.log(f"Optimized/{c_name}/inverted_eroded_depth_mask", rr.SegmentationImage(seg_depth_mask))
            
            # --- Optimized 3D Map ---
            if len(opt_frame.point_cloud_base) > 0:
                rr.log(f"Optimized_3D_Map/{c_name}_lidar_cloud_with_rgb_colors", rr.Points3D(opt_frame.point_cloud_base, colors=opt_frame.point_colors, radii=[0.01]))
                
            if opt_frame.depth_image is not None and opt_frame.depth_image.shape[0] > 0:
                v, u = np.where(opt_frame.depth_image > 0)
                z = opt_frame.depth_image[v, u]
                uv = np.vstack((u, v)).T
                
                pts_cam = unproject_points(uv, z, opt_camera_matrix, opt_dist_coeffs, c_model, opt_xi)
                
                T_cam_to_base = np.linalg.inv(T_base_to_cam_new)
                ones = np.ones((len(pts_cam), 1))
                pts_base = (T_cam_to_base @ np.hstack([pts_cam, ones]).T).T[:, :3]
                
                colors_bgr = opt_frame.image[v, u]
                colors_rgb = colors_bgr[:, ::-1] # BGR to RGB
                
                rr.log(f"Optimized_3D_Map/{c_name}_rgbd_cloud", rr.Points3D(pts_base, colors=colors_rgb, radii=[0.01]))
            
            self.current_replay_time += 10.0
            
        rr.set_time("timestamp", timestamp=self.current_replay_time)
        rr.log("TimelineEnd", rr.TextDocument(f"Best Cost: {self.best_cost:.4f}"))
        
        # Add a larger gap in the timeline to separate this optimization update from the next one
        self.current_replay_time += 50.0

def main():
    parser = argparse.ArgumentParser(description="Optimize LiDAR-to-Camera Extrinsics")
    parser.add_argument("--data_path", type=str, default=None, help="Path to a captured data directory. If not provided, uses the most recent directory.")
    parser.add_argument("--visualize", action="store_true", help="Enable ReRun visualization of optimization progress")
    parser.add_argument("--camera", type=str, choices=["left", "right", "center"], default="left", help="Which camera to optimize")
    parser.add_argument("--lidar", type=str, choices=["left", "right", "both"], default="left", help="Which lidar to optimize against")
    parser.add_argument("--debug", action="store_true", help="Enable OpenCV debugging visualizations")
    
    args = parser.parse_args()
    
    data_path = args.data_path
    if data_path is None:
        from stretch4_emulated_rgbd.shared_utils import find_latest_data_dir
        data_path = find_latest_data_dir()
        if data_path is None:
            print("Error: Could not find any captured data directory. Please specify --data_path.")
            return
        print(f"Auto-selected latest data directory: {data_path}")
        
    start_time = time.time()
    
    optimizer = ExtrinsicOptimizer(
        data_path, 
        camera=args.camera, 
        lidar=args.lidar, 
        visualize=args.visualize,
        debugging=args.debug
    )
    if not optimizer.sequences:
        return
        
    # If a warm start is being used, evaluate it explicitly FIRST so it shows up in the timeline
    if not np.allclose(optimizer.initial_x0, 0):
        print("\nEvaluating Warm Start...")
        warm_cost = optimizer.cost_function(optimizer.initial_x0)
        print(f"Warm Start Configuration Cost: {warm_cost:.4f}")
        
    initial_cost = optimizer.cost_function(np.zeros(6))
    print(f"Zero-State Configuration Cost: {initial_cost:.4f}")
    
    num_iterations = 0
    optimizer.set_interim_save_context(args, start_time, initial_cost, lambda: num_iterations)
    
    print("\nStarting CMA-ES Optimization...")
    initial_state = optimizer.initial_x0
    
    trans_sigma = config.EXTRINSICS_CMA_TRANSLATION_SIGMA_M
    rot_sigma_rad = config.EXTRINSICS_CMA_ROTATION_SIGMA_DEG * (np.pi / 180.0)
    
    cma_options = {
        'popsize': config.EXTRINSICS_CMA_POPSIZE, 
        'maxiter': config.EXTRINSICS_CMA_MAXITER,
        'tolfun': config.EXTRINSICS_CMA_TOLFUN,
        'tolx': config.EXTRINSICS_CMA_TOLX,
        'CMA_stds': [trans_sigma, trans_sigma, trans_sigma, rot_sigma_rad, rot_sigma_rad, rot_sigma_rad]
    }
    
    es = cma.CMAEvolutionStrategy(initial_state, 1.0, cma_options)
    
    while not es.stop():
        solutions = es.ask()
        costs = [optimizer.cost_function(s) for s in solutions]
        es.tell(solutions, costs)
        es.logger.add()
        es.disp()
        num_iterations += 1
        
    res = es.result
    
    if args.visualize:
        print("\nVisualizing final result...")
        optimizer.best_cost = float('inf')
        optimizer.cost_function(res[0])
        
        optimizer.current_replay_time += 10.0
        rr.set_time("timestamp", timestamp=optimizer.current_replay_time)
        rr.log("TimelineEnd", rr.TextDocument("Final Optimization Complete"))
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\nOptimization Complete!")
    print(f"Total time taken: {duration:.2f} seconds")
    print(f"Final Cost: {res[1]:.4f}")
    
    best_delta = res[0]
    t = best_delta[:3]
    r = Rotation.from_rotvec(best_delta[3:6]).as_matrix()
    T_delta = np.eye(4)
    T_delta[:3, :3] = r
    T_delta[:3, 3] = t
    
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    lidar_str = args.lidar if args.lidar.endswith("_lidar") else f"{args.lidar}_lidar"
    yaml_filename = f"optimization_results_mi_rgb_{args.camera}_camera_{lidar_str}_{timestamp_str}.yaml"
    yaml_path = optimizer.data_path / yaml_filename
    
    # Get the unique data subdirectories used
    data_subdirs = [str(seq["path"].name) for seq in optimizer.sequences]
    
    result_data = {
        "metadata": {
            "timestamp": timestamp_str,
            "duration_seconds": duration,
            "data_path": str(args.data_path),
            "data_subdirectories": data_subdirs,
            "camera": args.camera,
            "lidar": lidar_str,
            "num_sequences_evaluated": len(optimizer.sequences),
            "capture_fleet_id": optimizer.capture_fleet_id
        },
        "system_profile": get_sys_info(),
        "hyperparameters": {
            "method": "mi_rgb",
            "translation_sigma_m": config.EXTRINSICS_CMA_TRANSLATION_SIGMA_M,
            "rotation_sigma_deg": config.EXTRINSICS_CMA_ROTATION_SIGMA_DEG,
            "popsize": config.EXTRINSICS_CMA_POPSIZE,
            "maxiter": config.EXTRINSICS_CMA_MAXITER,
            "num_iterations_executed": num_iterations,
            "use_nmi": config.EXTRINSICS_USE_NMI,
            "grad_ksize": config.EXTRINSICS_GRAD_KSIZE,
            "tolfun": config.EXTRINSICS_CMA_TOLFUN,
            "tolx": config.EXTRINSICS_CMA_TOLX
        },
        "convergence": {
            "initial_cost": float(initial_cost),
            "final_cost": float(res[1]),
        },
        "results": {
            "best_delta_transform_array": best_delta.tolist(),
            "best_delta_transform_matrix": T_delta.tolist(),
        }
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(result_data, f, default_flow_style=None)
        
    for suffix in ["A", "B"]:
        interim_file = optimizer.data_path / f"INCOMPLETE_optimization_results_mi_rgb_{suffix}.yaml"
        if interim_file.exists():
            interim_file.unlink()
            
    print(f"Results successfully saved to {yaml_path}")

if __name__ == "__main__":
    main()

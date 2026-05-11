#!/usr/bin/env python3
import argparse
import os
import shutil
import yaml
from datetime import datetime

def main():
    """
    Helper script to safely install a new optimized calibration file as the robot's default.
    Checks for fleet mismatches and prevents accidentally downgrading to an older calibration.
    """
    parser = argparse.ArgumentParser(description="Install a new optimized Emulated RGB-D calibration to the robot.")
    parser.add_argument("optimization_yaml", type=str, help="Path to the optimization results YAML file to install.")
    parser.add_argument("-f", "--force", action="store_true", help="Force installation, bypassing safety checks.")
    args = parser.parse_args()
    
    source_path = args.optimization_yaml
    if not os.path.exists(source_path):
        print(f"Error: Source file '{source_path}' does not exist.")
        return
        
    try:
        with open(source_path, "r") as f:
            source_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading source YAML: {e}")
        return
        
    metadata = source_data.get("metadata", {})
    capture_fleet_id = metadata.get("capture_fleet_id", "UNKNOWN")
    source_timestamp_str = metadata.get("timestamp", "0")
    
    camera = metadata.get("camera", "left")
    lidar = metadata.get("lidar", "left_lidar")
    if not lidar.endswith("_lidar"):
        lidar = f"{lidar}_lidar"
    
    validity_masks = metadata.get("validity_masks", {})
    rgb_mask_info = validity_masks.get("rgb_vignette_mask", {})
    depth_mask_info = validity_masks.get("depth_valid_mask", {})
    
    rgb_mask_filename = rgb_mask_info.get("filename")
    depth_mask_filename = depth_mask_info.get("filename")
    
    fleet_id = os.environ.get("HELLO_FLEET_ID")
    fleet_path = os.environ.get("HELLO_FLEET_PATH")
    
    if not fleet_id or not fleet_path:
        print("Error: HELLO_FLEET_ID or HELLO_FLEET_PATH environment variables are missing.")
        print("This script must be run on a configured Stretch robot.")
        return
        
    print(f"Installing Optimized Calibration...")
    print(f"Current Robot: {fleet_id}")
    print(f"Data Captured On: {capture_fleet_id}")
    
    # Check for fleet mismatch
    if capture_fleet_id != fleet_id and not args.force:
        print("\n" + "!" * 80)
        print("WARNING: FLEET MISMATCH DETECTED!")
        print(f"The optimized calibration data was captured on robot '{capture_fleet_id}', but the current robot is '{fleet_id}'.")
        print("Using extrinsics from another robot will likely degrade performance and result in misaligned RGB-D.")
        print("!" * 80 + "\n")
        
        response = input(f"Are you sure you want to install this calibration on {fleet_id}? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Installation aborted.")
            return

    target_dir = os.path.join(fleet_path, fleet_id, "calibration_cameras")
    target_path = os.path.join(target_dir, f"emulated_rgbd_extrinsics_{camera}_camera_{lidar}.yaml")
    
    if not os.path.exists(target_dir):
        print(f"Error: Target calibration directory '{target_dir}' does not exist.")
        return
        
    # Check for older calibration
    if os.path.exists(target_path) and not args.force:
        try:
            with open(target_path, "r") as f:
                target_data = yaml.safe_load(f)
                
            target_metadata = target_data.get("metadata", {})
            target_timestamp_str = target_metadata.get("timestamp", "0")
            
            if source_timestamp_str < target_timestamp_str:
                print("\n" + "!" * 80)
                print("WARNING: OLDER CALIBRATION DETECTED!")
                print(f"The calibration you are installing (timestamp: {source_timestamp_str}) is older")
                print(f"than the currently installed calibration (timestamp: {target_timestamp_str}).")
                print("!" * 80 + "\n")
                
                response = input("Are you sure you want to overwrite the newer calibration? [y/N]: ")
                if response.lower() not in ['y', 'yes']:
                    print("Installation aborted.")
                    return
        except Exception as e:
            print(f"Warning: Could not parse existing calibration to check age: {e}")
            
    # Install YAML
    try:
        shutil.copy2(source_path, target_path)
        print(f"\nSuccessfully installed optimized calibration to:")
        print(f"{target_path}")
        
        # Install Masks
        source_dir = os.path.dirname(os.path.abspath(source_path))
        if rgb_mask_filename:
            source_rgb_mask = os.path.join(source_dir, rgb_mask_filename)
            if os.path.exists(source_rgb_mask):
                target_rgb_mask = os.path.join(target_dir, rgb_mask_filename)
                shutil.copy2(source_rgb_mask, target_rgb_mask)
                print(f"Installed: {target_rgb_mask}")
            else:
                print(f"Warning: Could not find bundled RGB mask '{source_rgb_mask}'")
                
        if depth_mask_filename:
            source_depth_mask = os.path.join(source_dir, depth_mask_filename)
            if os.path.exists(source_depth_mask):
                target_depth_mask = os.path.join(target_dir, depth_mask_filename)
                shutil.copy2(source_depth_mask, target_depth_mask)
                print(f"Installed: {target_depth_mask}")
            else:
                print(f"Warning: Could not find bundled depth mask '{source_depth_mask}'")
                
    except Exception as e:
        print(f"\nError installing files: {e}")

if __name__ == "__main__":
    main()

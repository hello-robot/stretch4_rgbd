#!/usr/bin/env python3
import os
import subprocess
import argparse
import tempfile
import shutil
import sys

# Ensure the root directory is in sys.path so we can import rgbd_networking
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from stretch4_emulated_rgbd.rgbd_networking import robot_ip
except ImportError:
    robot_ip = '100.90.83.97' # fallback

def main():
    parser = argparse.ArgumentParser(description="Sync camera calibration models and validity masks from the robot to the local desktop.")
    parser.add_argument('--ip', type=str, default=robot_ip, help="The IP address of the robot")
    parser.add_argument('--user', type=str, default='hello-robot', help="The SSH user for the robot")
    parser.add_argument('--include-images', action='store_true', help="Include the large calibration_images directory when syncing")
    args = parser.parse_args()

    local_fleet_path = os.environ.get('HELLO_FLEET_PATH', os.path.expanduser('~/stretch_user'))
    
    print(f"Connecting to {args.user}@{args.ip} to fetch camera calibration data...")
    
    # 1. Determine the robot ID by inspecting the remote directory
    # We look for a folder starting with "stretch-" in the user's stretch_user directory
    find_id_cmd = ['ssh', f'{args.user}@{args.ip}', f'ls -1 /home/{args.user}/stretch_user/ | grep -E "^stretch-" | head -n 1']
    try:
        result = subprocess.run(find_id_cmd, check=True, stdout=subprocess.PIPE, text=True)
        robot_id = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error fetching robot ID via SSH: {e}")
        return
        
    if not robot_id:
        print("Error: Could not find a valid robot_id (e.g., stretch-se4-4010) on the remote machine.")
        return
        
    print(f"Discovered robot_id: {robot_id}")
    
    # 2. Sync the files
    with tempfile.TemporaryDirectory() as temp_dir:
        remote_path = f'/home/{args.user}/stretch_user/{robot_id}/calibration_cameras'
        
        command = [
            'rsync', 
            '-avz',
        ]
        
        if not args.include_images:
            command.append('--exclude=calibration_images/')
            
        command.extend([
            f'{args.user}@{args.ip}:{remote_path}/', 
            temp_dir
        ])
        
        try:
            print("Downloading calibration files...")
            subprocess.run(command, check=True)
            print("Successfully downloaded calibration files to a temporary directory.")
        except subprocess.CalledProcessError as e:
            print(f"Error syncing calibration files: {e}")
            return
            
        local_dir = os.path.join(local_fleet_path, robot_id, 'calibration_cameras')
        os.makedirs(local_dir, exist_ok=True)
        
        # 3. Move files to the correct local directory
        copied_files = 0
        for item_name in os.listdir(temp_dir):
            src_path = os.path.join(temp_dir, item_name)
            dst_path = os.path.join(local_dir, item_name)
            
            # Remove destination if it exists so shutil.move doesn't nest or error out
            if os.path.exists(dst_path):
                if os.path.isdir(dst_path):
                    shutil.rmtree(dst_path)
                else:
                    os.remove(dst_path)
                    
            shutil.move(src_path, dst_path)
            copied_files += 1
            
        print(f"Successfully synced {copied_files} items to {local_dir}")

if __name__ == "__main__":
    main()

import subprocess

# Set these values for your network
                                                                                                                                   
# Calder 4010
#robot_ip = '100.90.83.97'

# Dali 4021
robot_ip = '100.97.201.62'                                                                                                               

remote_computer_ip = '100.69.89.24'

# Set these to your preferred port numbers
rgbd_and_joints_port = 4410

network_debugging = False

def print_network_info():
    print("================================================================")
    print("WARNING: If you experience ZMQ connection issues between a") 
    print("remote computer and the robot, please ensure the ports are") 
    print("not blocked by a firewall.")
    print("")
    print("For example, the Uncomplicated Firewall (UFW) can be configured")
    print(f"to allow incoming TCP connections on port {rgbd_and_joints_port} by running: ")
    print(f"sudo ufw allow {rgbd_and_joints_port}/tcp")
    print("================================================================")
    if network_debugging:
        print("\n[NETWORK DEBUG] Active Network Interfaces:")
        try:
            result = subprocess.run(['ip', '-4', '-br', 'addr'], capture_output=True, text=True, timeout=2)
            print(result.stdout)
            if 'tailscale0' in result.stdout:
                print("[NETWORK DEBUG] -> Tailscale active. If connecting remotely, ensure IPs use the 100.x.x.x Tailscale subnet.")
            else:
                print("[NETWORK DEBUG] -> Tailscale NOT found. Ensure devices are on the same local subnet.")
        except Exception as e:
            print(f"[NETWORK DEBUG] Failed to list interfaces: {e}")
        print("----------------------------------------------------------------\n")

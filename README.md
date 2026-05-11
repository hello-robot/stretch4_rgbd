# Stretch 4 Emulated RGB-D

This repository contains the software for optimizing and visualizing RGB-D images on the Stretch 4 robot. RGB-D images are created by combining images from a head-mounted RGB camera with scans from one or two of the head-mounted LiDAR sensors. The code provides methods to generate temporally-synchronized and spatially-aligned RGB-D images at 10Hz with low latency. 

Tools to optimize the rigid-body alignment between an RGB camera and a LiDAR sensor (i.e., camera-LiDAR extrinsics) achieve spatial alignment between the RGB image and depth image components of the emulated RGB-D images. Specialized data capture and synchronization code provide RGB-D images with temporally-synchronized RGB image and depth image componentsat at up to 10Hz (i.e., the maximum LiDAR sensor frame rate) with low latency. 

## Table of Contents
- [Installation](#installation)
  - [Option 1: Standard Pip Installation](#option-1-standard-pip-installation)
  - [Option 2: Automated Install Script](#option-2-automated-install-script)
- [Usage](#usage)
  - [1. Data Capture](#1-data-capture)
  - [2. Preprocessing](#2-preprocessing)
  - [3. Visualize Data](#3-visualize-data)
  - [4. Optimize the Camera-LiDAR Extrinsics](#4-optimize-the-camera-lidar-extrinsics)
  - [5. Visualize the Optimization Results](#5-visualize-the-optimization-results)
  - [6. Installation of Optimized Calibrations (On Robot)](#6-installation-of-optimized-calibrations)
  - [7. Validity Mask Estimation (On Robot)](#7-validity-mask-estimation-on-robot)
  - [8. Visualize Live RGB-D Imagery with the New Calibration (On Robot)](#8-visualize-live-rgb-d-imagery-with-the-new-calibration-on-robot)
- [API Usage and Reference](#api-usage-and-reference)
- [Configuration (`emulated_rgbd_config.py`)](#configuration-emulated_rgbd_configpy)
  - [Extrinsic Optimization Parameters](#extrinsic-optimization-parameters)
  - [Depth Alignment & Spatial Corrections](#depth-alignment--spatial-corrections)
  - [Sparsity Shadow Filter](#sparsity-shadow-filter)
  - [Native Image Orientation](#native-image-orientation)
- [Spatial Alignment Overview](#spatial-alignment-overview)
  - [Motivation](#motivation)
- [Temporal Quality Overview](#temporal-quality-overview)
  - [Frame Rate & Phase Alignment](#frame-rate--phase-alignment)
  - [Latency](#latency)
  - [High-Frequency Pipeline Methods](#high-frequency-pipeline-methods)
- [Extrinsic Calibration Details](#extrinsic-calibration-details)
  - [Example Output File](#example-output-file)
  - [Mathematical Interpretation and Application](#mathematical-interpretation-and-application)

## Installation
Many of the optimization and visualization components of this repository can be run on a desktop computer without installing the `stretch_body_ii` package. 

### Option 1: Standard Pip Installation
If you already manage your own virtual environments (e.g., via `conda`, `venv`, or `pyenv`), you can install the package directly using `pip`. This will automatically install the package and its minimal dependencies as defined in `setup.py`:

```bash
pip install -e .
```

> **Note for On-Robot Usage:** If you are running this on the Stretch 4 robot to capture data, the script requires the `stretch_body_ii` system package. Make sure to create your virtual environment with the `--system-site-packages` flag (e.g., `python3 -m venv --system-site-packages venv`) so it can access system-level dependencies.

### Option 2: Automated Install Script
Alternatively, to automatically create a new isolated virtual environment and install the minimal set of dependencies into it, run the provided install script. This script automatically uses the `--system-site-packages` flag, making it ideal for both desktop and on-robot usage:

```bash
./install_dependencies.sh
```

To activate the environment created by the script:
```bash
source venv/bin/activate
```

## Usage

### 1. Data Capture
To capture raw data for alignment from the robot, use the following script. Currently only the left fisheye camera and left LiDAR have been tested using three RGB-D images consisting of different views of a static office space. Specifically, the robot was rotated in place to see different parts of the office. 

*(Note: It's important to capture images of a static scene. For better results, the images should include foreground objects with prominent depth edges across the RGB camera's field of view).*

Run the script, position the robot to a view you want to capture, and then press the space bar to capture the view for calibration. Repeat this process to acquire at least two more views and then press 'Q' to quit.

```bash
python3 scripts/capture_emulated_rgbd.py --camera left --lidar left
```
*(Note: This must be run on the Stretch 4 robot. It requires `stretch_body_ii` and the use of the robot's cameras and LiDARs)*

### 2. Preprocessing
Before optimizing, ensure you have computed the static validity masks for the sensors:
```bash
python3 scripts/create_validity_masks_for_extrinsic_optimization.py ./data/captured_emulated_rgbd_<timestamp>/
```

You can estimate the maximum pixel distance between the sparse depth points resulting from the LiDAR 3D points using the following script. This can help you determine an appropriate value for MAX_LIDAR_INTERPOLATION_DIST_PX in the optimization configuration file, `stretch4_emulated_rgbd/emulated_rgbd_config.py`

```bash
python3 scripts/estimate_lidar_gap.py --data_path ./data/captured_emulated_rgbd_<timestamp>/
```
### 3. Visualize Data

You should now visualize the captured data to make sure that it is of sufficient quality for extrinsic optimization. For example, it's important that the camera images have reasonable brightness without overly dark or bright areas. Similarly, it's important that the LiDAR points appear to be accurate without missing sections due to a partial scan. As noted above, it is also important to have at least three distinct views with prominent foreground objects across the field of view. Foreground objects are important since they will typically result in edges in both the LiDAR-based depth image and the RGB camera image that can be aligned via mutual information. 

To visualize the data, run the following script.

```bash
python3 scripts/visualize_emulated_rgbd.py --data_path ./data/captured_emulated_rgbd_<timestamp>/
```
*(Note: The visualization will first have you review the validity masks using OpenCV windows. Click on a window and press 'y' to approve. Then the visualization will use Rerun to visualize the RGB-D calibration data. The Rerun timeline can be used to visualize different captured views in the data.)* 

### 4. Optimize the Camera-LiDAR Extrinsics
Run the core CMA-ES optimization to refine the 6D rigid body extrinsics representing the pose of the camera:
```bash
python3 scripts/optimize_extrinsics.py --data_path ./data/captured_emulated_rgbd_<timestamp>/
```
This script leverages Mutual Information (MI) to structurally align the edges of the projected LiDAR depth map with the visual edges in the RGB image. It does not optimize camera intrinsics. 

Using the `--visualize` command line argument will visualize the optimizations progress via Rerun. Using the `--debug` command line argument will result in OpenCV windows that show the extracted image gradients and intersection masks used by the mutual information objective function.

### 5. Visualize the Optimization Results

Now, run the following script to visualize the results of the calibration. The left Rerun panel shows the captured data before calibration and the right panel shows the data after calibration. By hiding the sparse depth overlay and then repeatedly hiding and revealingthe the dense depth image overlay on top of the RGB image, you can assess the spatial alignment between the dense depth image and the RGB image.

```bash
python3 scripts/visualize_optimized_rgbd.py --data_path ./data/captured_emulated_rgbd_<timestamp>/ --opt_yaml ./data/captured_emulated_rgbd_<timestamp>/optimization_results_mi_rgb_left_camera_left_<calibration_timestamp>.yaml
```
*(Note: The visualization will first have you review the validity masks using OpenCV windows. Click on a window and press 'y' to approve.)*

### 6. Installation of Optimized Calibrations (On Robot)
Once you are satisfied with the calibration, you can install the resulting YAML file as the robot's default calibration. The provided helper script checks for fleet mismatches (i.e. capturing data on one robot and installing on another) and prevents accidental downgrading to older calibrations.
```bash
python3 scripts/install_optimized_calibration.py ./data/captured_emulated_rgbd_<timestamp>/optimization_results_mi_rgb_<calibration_timestamp>.yaml
```
Once installed, the `FastEmulatedRGBDStreamer` and `rgbd_rtmo_pose_estimation.py` will automatically load and apply this optimized calibration.

### 7. Validity Mask Estimation (On Robot)
After optimizing the extrinsics and installing them, new physical validity masks should be estimated directly on the robot using the optimized calibration. This step dynamically calculates masks that zero out invalid RGB pixels caused by hardware vignetting and prevents the interpolation of the dense depth map from bleeding into regions with no physical LiDAR coverage.

To capture the live frames and estimate the masks, run the following script on the robot:
```bash
python3 scripts/estimate_validity_masks.py --camera left --lidar left
```
*(Note: This must be run on the Stretch 4 robot. The script captures exactly 30 synchronized frames, computes the masks at the highest active resolution, and saves them locally to `data/validity_masks/` for the visualizers to use automatically).*

### 8. Visualize Live RGB-D Imagery with the New Calibration (On Robot)
To visualize captured or live data streams run:

```bash
python3 scripts/visualize_emulated_rgbd.py
```

## API Usage and Reference

For developers writing custom applications, the repository provides a unified API in `stretch4_emulated_rgbd.api` to stream and process synchronized RGB-D frames.

> [!TIP]
> **Quick Start:** For a complete, runnable demonstration of the API capabilities—including lazy properties, calibration extraction, validity masking, dense depth interpolation, and colored 3D point cloud generation—see [`examples/api_example.py`](file:///home/hello-robot/repos/stretch4_emulated_rgbd/examples/api_example.py).

#### Summary of Processing Steps
When the `FastEmulatedRGBDStreamer` captures and aligns an RGB-D frame, it executes the following steps internally *before* yielding it to you:
1. **Temporal Synchronization**: Waits for the LiDAR sweep to finish and grabs the temporally closest high-frequency RGB image to minimize phase lag.
2. **Spatial Transformation & Culling**: Projects 3D LiDAR points into the camera's frame of reference, dropping any points physically behind the camera.
3. **Zero-Copy Distorted Projection**: Projects the 3D points directly into the raw, distorted 2D fisheye image space.
4. **Z-Buffer Sorting**: If multiple LiDAR points land on the exact same 2D pixel, only the closest point (minimum Z) is kept.
5. **Sparsity Shadow Filter**: If enabled, checks local neighborhoods to aggressively remove background depth points that visually "bleed" onto foreground objects due to parallax.

> [!IMPORTANT]  
> The streamer returns a **sparse** depth map (only specific pixels hit by LiDAR have non-zero depth) and does **not** apply validity masks internally. You must apply masks or compute dense depth explicitly using the API as shown below.

#### Code Examples

**Receiving the Stream**
```python
from stretch4_emulated_rgbd.api import get_emulated_rgbd_stream

# Automatically loads the optimized calibration for the current fleet
streamer, generator = get_emulated_rgbd_stream(
    use_left=True, 
    use_left_lidar=True,
    emulated_rgbd_fps=10.0
)

# Fetch a single synchronized frame
frame = next(generator)
```

**Applying Validity Masks**
```python
from stretch4_emulated_rgbd.api import ValidityMaskManager
import cv2

mask_manager = ValidityMaskManager()
# Automatically loads the robot's physical vignetting and LiDAR bounds masks
vig_mask, depth_mask = mask_manager.get_masks("left", "left_lidar", frame.image.shape)

# Apply vignette mask to black out the physical camera housing
masked_rgb = frame.image.copy()
masked_rgb[~vig_mask] = 0
```

**Generating a Dense Depth Image**
```python
from stretch4_emulated_rgbd.api import DenseDepthImage

# Initialize the processor with raw RGB and sparse depth
dense_processor = DenseDepthImage(frame.image, frame.depth_image)

# Combine masks to prevent depth bleeding into physically impossible areas
combined_mask = vig_mask & depth_mask if (vig_mask is not None and depth_mask is not None) else None

# Interpolate using distance transforms
dense_depth = dense_processor.compute_dense_depth(valid_region_mask=combined_mask)
```

**Creating a Point Cloud from an RGB-D Image**
```python
from stretch4_emulated_rgbd.api import get_camera_intrinsics, create_point_cloud_from_depth

# Fetch intrinsics dynamically
cam_matrix, dist_coeffs = get_camera_intrinsics(streamer, "left")

# Re-project the 2D dense depth image back into a 3D Nx3 point cloud array
pts_cam, colors = create_point_cloud_from_depth(
    dense_depth, 
    frame.image, 
    cam_matrix, 
    dist_coeffs
)
```

**Obtaining the 3D Location of a Single Pixel**
```python
from stretch4_emulated_rgbd.api import get_pixel_3d_location

u, v = 640, 400 # Center of the image

# Get the 3D coordinate of the pixel in the camera frame and the base frame
pt_cam, pt_base = get_pixel_3d_location(
    u, v, 
    dense_depth, 
    cam_matrix, 
    dist_coeffs, 
    T_base_to_cam=T_base_to_cam
)

if pt_cam is not None:
    print(f"Pixel ({u}, {v}) is at {pt_cam} in camera frame.")
    print(f"Pixel ({u}, {v}) is at {pt_base} in base frame.")
else:
    print(f"Pixel ({u}, {v}) has no valid depth.")
```

**Accessing Intrinsics and Extrinsics**
To ensure you never accidentally use the wrong calibration, the exact intrinsic and extrinsic matrices used to generate an `RGBDFrame` are permanently attached to the frame object itself.

```python
# Camera Intrinsics
cam_matrix = frame.camera_matrix
dist_coeffs = frame.distortion_coefficients

# Extrinsic transform from the robot's base frame to the camera optical frame
T_base_to_cam = frame.T_base_to_cam

# Extrinsic transforms from the LiDAR frames to the robot's base frame
T_lidar_to_base_left = frame.T_lidar_to_base_left
T_lidar_to_base_right = frame.T_lidar_to_base_right

print("Optimized Extrinsics (T_base_to_cam):\n", T_base_to_cam)
```

**Visualizing with ReRun**
```python
import rerun as rr
from stretch4_emulated_rgbd.api import visualize_rgbd_frame

rr.init("Stretch API Example", spawn=True)

# Helper function automatically visualizes the raw RGB, sparse depth, 
# dense depth (if computed via DenseDepthImage), and 3D point cloud overlays
visualize_rgbd_frame("left", frame, vig_mask=vig_mask, depth_mask=depth_mask)
```

We also provide two examples demonstrating transmitting the tightly-synchronized `RGBDFrame` generator over a PyZMQ network socket:
- **Publisher**: `python3 examples/send_rgbd_images.py`
- **Subscriber**: `python3 examples/recv_rgbd_images.py`

## Configuration (`emulated_rgbd_config.py`)

All key hyperparameters used throughout the optimization, depth alignment, and shadow filtering processes are centralized in `stretch4_emulated_rgbd/emulated_rgbd_config.py`. You can adjust these settings to fine-tune the pipeline for your specific requirements.

The configuration file is organized into several main categories:

### Extrinsic Optimization Parameters
Because the factory calibration from Hello Robot is already of high quality, the CMA-ES optimizer is constrained to a local search radius to prevent divergent or absurd alignments. These settings control the boundaries and behavior of the optimization:
- **Optimization Constraints**: Bounds on the maximum allowed translational shift (e.g., `MAX_TRANSLATION_M` = 0.03 m) and rotational shift (e.g., `MAX_ROTATION_DEG` = 10.0 deg) applied to the `T_delta` matrix. Exceeding these bounds applies a heavy penalty (`BOUNDS_PENALTY_WEIGHT`). You can tune these values if you suspect a larger physical shift occurred (e.g., if a camera was physically bumped or remounted).
- **CMA-ES Hyperparameters**: Settings such as population size, maximum iterations, and early stopping tolerances (e.g., `EXTRINSICS_CMA_TOLFUN` and `EXTRINSICS_CMA_TOLX`) to balance accuracy and optimization speed.
- **Objective Function Settings**: Adjusts how Mutual Information is calculated, including `EXTRINSICS_USE_NMI` (normalized vs standard), `EXTRINSICS_GRAD_KSIZE` (Sobel kernel size for edges), and `EXTRINSICS_MI_BINS` (histogram bins).
- **Initialization**: `EXTRINSICS_IGNORE_PRIOR_OPTIMIZATIONS` allows starting from factory calibration instead of resuming from a previously optimized state.

### Depth Alignment & Spatial Corrections
These parameters control how sparse LiDAR points are processed and how invalid regions are masked out:
- **Depth Masking**: Thresholds and window sizes for density-based depth validity masks and morphological closing operations.
- **Interpolation Limits**: Maximum pixel distance (`MAX_LIDAR_INTERPOLATION_DIST_PX`) a sparse depth point can be interpolated before being marked invalid.
- **Saturation Masking**: Thresholds for excluding overexposed RGB pixels from the edge alignment process.
- **Physical Boundaries**: Minimum and maximum physical depths (e.g., `MIN_PHYSICAL_DEPTH_M`, `MAX_PHYSICAL_DEPTH_M`) to clip the synthetic depth map.

### Sparsity Shadow Filter
This filter resolves an artifact where the edges on one side of objects appear to have alternating foreground and background "stripes" in the dense depth image. 

Because the left and right LiDAR sensors are physically offset from the RGB cameras, they can see 3D points that should be occluded from a camera's perspective. Since the LiDAR points are sparse, both the foreground depth points and the background depth points that should be occluded project to the same region on the camera's image plane without overlapping exactly. This effect is most prominent when an object is close to the sensors and reduces as the object moves farther away, resulting in more similar views of the object from the LiDAR sensors and camera.

A moving window shadow filter is used to reduce this effect. The filter acts as a sparse Z-buffer:
- **`ENABLE_SHADOW_FILTER`**: When enabled, the filter attempts to remove occluded background points that "bleed" into foreground objects by utilizing the depth information from neighboring pixels.
- **`SHADOW_FILTER_WINDOW_SIZE`**: This is the pixel width of the square window used to decide if a depth point should be removed. If a point in the local neighborhood is closer to the RGB camera than the depth point being evaluated by more than `SHADOW_FILTER_DEPTH_THRESHOLD_M`, the point is considered an occluded background point (i.e., shadowed point) and removed. If the window size is too large, it can filter points that should be visible to the RGB camera. If it is too small, it leaves points that should not be visible to the RGB camera.
- **`SHADOW_FILTER_DEPTH_THRESHOLD_M`**: This is the depth threshold used to determine if a point is an occluded background point. A large value will only remove points on objects with a background that is far away. A small value will remove points with a background that is nearby, but will also remove points on surfaces that are angled away from the RGB camera.
- **`SHADOW_FILTER_USE_CIRCULAR_WINDOW`**: Toggles whether the moving window uses a circular (elliptical) structuring element instead of a square. A circular window provides more mathematically accurate, isotropic filtering and prevents blocky artifacts around object contours, making it advantageous for larger window sizes (e.g., >= 5). Circular kernels are non-separable and introduce latency, so they are disabled by default.

*Implementation Note:* Under the hood, the filter uses an image erosion operation (`cv2.erode`) to efficiently find the minimum depth within the local neighborhood window for all pixels simultaneously. Any point whose original depth exceeds this local minimum by more than the threshold is classified as a shadowed point and removed.

### Native Image Orientation
The raw images from the head-mounted fisheye cameras natively output in a rotated, horizontal orientation due to how the sensors are physically mounted. To address this, the pipeline can rotate the images 90 degrees to an upright vertical orientation early on.
- **`ROTATE_IMAGES_TO_VERTICAL`**: The master toggle that enables 90-degree rotations across the pipeline, ensuring downstream applications receive upright images.
- **`USE_BOARD_LEVEL_ROTATION`**: Attempts to perform the 90-degree rotation directly on the Luxonis OAK-FFC board's hardware before MJPEG compression. 

> [!WARNING]
> **Hardware Limitations of Board-Level Rotation**
> The Luxonis board (Myriad X VPU) lacks a fast, zero-cost 90-degree memory transpose operation for the NV12 format. Instead, it processes 90-degree rotations using a generic hardware warp engine (`ImageManip`), which is heavily compute-intensive. 
> 
> Because of this:
> 1. Board-level rotation is incompatible with the 600p resolution when using MJPEG compression, as the rotated width (600) is not a multiple of 16.
> 2. Attempting to rotate high-resolution frames overwhelms the hardware, resulting in dropped frames and noticeable pipeline latency between the RGB and depth components. **When `USE_BOARD_LEVEL_ROTATION` is True, the `camera_fps` must be restricted to 10 or lower.**
> 
> **STRONGLY RECOMMENDED:** For applications requiring precise temporal synchronization or high frame rates, keep `USE_BOARD_LEVEL_ROTATION = False`. The pipeline utilizes a **"lazy evaluation"** architecture that automatically transports the natively compressed MJPEG frames over the network to save bandwidth. The software fallback then transparently applies a highly efficient (<1ms) 90-degree rotation on the host CPU exactly when the developer accesses the `frame.image` or `frame.depth_image` properties, completely insulating the user from the native orientation.

## Spatial Alignment Overview

This repository provides optimization methods for spatially aligning the RGB image and depth image components of emulated RGB-D images from Stretch 4. The optimization uses Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to optimize the extrinsic 6D rigid body transformation between a LiDAR sensor and a camera sensor. The objective function uses the normalized mutual information (NMI) between the gradients of the RGB image and the gradients of the projected LiDAR depth image. 

In practice, this optimization results in the RGB and depth images being well-aligned, which greatly simplifies their use. The code also provides visualizations of the alignment results and the optimization procees. Other helpful utilities includes a script that generates data-driven masks representing the valid pixels in the depth and RGB images. 

### Motivation

The RGB image comes directly from one of the three cameras in Stretch 4's head: the left fisheye camera, the right fisheye camera, or the high-resolution wide-angle center camera. The depth image is created by transforming 3D points from one or both LiDAR sensors into the camera's frame of reference and then projecting them onto the focal plane of a camera model. 

Prior to shipping a Stretch 4 robot, Hello Robot uses an extensive calibration procedure involving a calibration pattern mounted to the end of the robot's arm. This procedure yields high-quality intrinsic camera parameters, including the focal length, principal point, and distortion coefficients for each of the three cameras. The calibration pattern also has reflective fiducial markers whose 3D locations relative to the visible-light calibration pattern are known. This information is used to compute an extrinsic 6D rigid body transformation between each LiDAR sensor and each camera sensor. 

While this extrinsic calibration is of high quality, the spatial alignment between the depth image and the RGB image are highly sensitive to the 6D rigid body transformation (i.e., extrinsics for the LiDAR and camera). The remaining error creates challenges for applications. For example, points and regions output by computer vision models applied to the RGB image cannot be easily associated with the corresponding 3D points in the depth image.

## Temporal Quality Overview

### Frame Rate & Phase Alignment
The Emulated RGB-D pipeline is driven by the physical rotation of the Hesai LiDAR, which completes a 360-degree sweep at 10Hz (100ms per rotation). The pipeline's target output rate (`--emulated_rgbd_fps`) is strictly tied to fractions of this rotation (e.g., 10Hz, 5Hz).

Because there is no hardware sync signal aligning the *phase* of the RGB camera's exposure with the LiDAR's physical rotation, running the camera at 10Hz can result in up to 50ms of phase misalignment latency. For example, if the LiDAR sweep midpoint occurs at $T=50$ms, but the camera exposes at $T=0$ms and $T=100$ms, the minimum temporal gap between the data is 50ms.

**Software Over-Sampling**: To minimize phase lag, the pipeline employs an inverted, LiDAR-driven architecture. The camera hardware runs at a higher frame rate (configured via `--camera_fps`, e.g., 30Hz), continuously buffering compressed MJPEG frames in a background thread. The pipeline waits for a LiDAR sweep to finish, calculates its exact temporal midpoint, and immediately pulls the single RGB frame from the buffer that minimizes the temporal gap. At 30Hz, this drops the maximum phase misalignment from ~50ms to ~16ms without incurring the CPU cost of decompressing unused frames.

### Latency
Maintaining low latency and strict temporal synchronization between the instantaneous RGB image capture and the 100ms LiDAR sweep requires several advanced software mitigations to overcome hardware and network limitations:

1. **Hardware Clock Offset Estimation**: The PyHesai driver outputs point clouds tagged with the LiDAR's internal hardware clock timestamp. However, this clock is often desynchronized from the host computer's monotonic clock (which the Luxonis RGB cameras use). Furthermore, standard UDP buffers within the driver can create significant "buffer bloat," causing timestamps captured at the *reception* of the packet to lag the true physical capture time by up to 300ms. To eliminate this, the `LidarPoller` continuously calculates the minimum transmission delay between the LiDAR's hardware clock and the host's monotonic clock (`time.monotonic()`). By dynamically maintaining this `clock_offset`, it assigns precise, jitter-free host timestamps to the LiDAR points that reflect the exact physical moment the light hit the sensor.

2. **Physical Sweep Lookahead**: Because a global shutter RGB camera captures an image instantaneously at time `T`, the LiDAR sweep that optimally captures the same state of the world is the one spanning from `T - 50ms` to `T + 50ms`. Since this physical rotation will not finish until `T + 50ms`, eagerly requesting the "closest" point cloud at time `T` previously caused the system to fetch the *prior* sweep (ending at `T - 50ms`), injecting 100ms of structural latency. The `get_closest_frame` method solves this by intentionally blocking and waiting until the ideal sweep finishes rotating and arrives over the network, achieving near-perfect temporal synchronization at the cost of a strictly bounded ~50ms mechanical delay.

3. **Global Shutter & Sweep Midpoint Synchronization**: The synchronization logic mathematically matches the instantaneous RGB timestamp with the temporal *midpoint* of the LiDAR sweep (calculated from the first and last point timestamps). This perfectly balances the temporal error across the 100ms rotation window, ensuring that moving objects in the center of the RGB image map tightly to the corresponding LiDAR points.

### High-Frequency Pipeline Methods
To achieve and stabilize the 10Hz target without CPU or USB bottlenecking, the default `stretch_body_ii` pipeline was heavily refactored into the low-latency `FastEmulatedRGBDStreamer` and `HeadCamera` classes:

1. **Direct Hardware Access & Zero-Copy Structuring**: The overhead of inter-process message passing and excessive buffering was removed. Data is polled directly from the sensor drivers into a tight Python generator loop.
2. **Non-Blocking Queues & Memory Pools**: The Luxonis OAK-FFC DepthAI pipeline is configured with strictly bounded memory pools (`setNumFramesPools=2`) and a non-blocking output queue (`maxSize=1`). If the host CPU stalls, the camera driver instantly overwrites the oldest frame rather than queuing it, guaranteeing the host *always* pulls the absolute freshest physical frame.
3. **USB MJPEG Compression**: Passing uncompressed 1200p or 800p RGB arrays at 10Hz concurrently with intense UDP LiDAR traffic can easily saturate the USB bus, leading to dropped frames or variable latency. The streamer delegates MJPEG compression directly to the OAK-FFC's hardware VideoEncoder, securing deterministic transmission times.
4. **Inverse Distorted Projection**: Instead of executing an expensive dense image unwarp (`cv2.undistort`) on the high-resolution RGB stream at 10Hz—a heavy $O(N_{pixels})$ operation—the math is inverted. The sparse 3D LiDAR point cloud is projected directly into the raw, distorted fisheye image space—a lightweight $O(N_{lidar\_points})$ operation. This drastically reduces CPU load and garbage collection pauses.

## Extrinsic Calibration Details

The output of the `optimize_extrinsics.py` script is a YAML file containing the results of the CMA-ES optimization process. This file contains metadata, the system profile, hyperparameter values used during the run, the convergence metrics, and the optimized transform matrices.

### Example Output File
A typical `optimization_results_mi_rgb_{camera}_camera_{lidar}_<timestamp>.yaml` file looks like this:

```yaml
convergence:
  final_cost: -0.6432
  initial_cost: -0.4281
hyperparameters:
  grad_ksize: 5
  maxiter: 500
  method: mi_rgb
  num_iterations_executed: 142
  popsize: 30
  sigma0: 2.0
  tolfun: 1.0e-11
  tolx: 1.0e-11
  use_nmi: true
metadata:
  camera: left
  lidar: left_lidar
  data_path: ./data/captured_emulated_rgbd_20260429_162908
  data_subdirectories:
  - left_camera_left_lidar_20260502_214533
  duration_seconds: 481.52
  num_sequences_evaluated: 3
  timestamp: '20260502_214533'
  capture_fleet_id: stretch-se4-4010
  validity_masks:
    depth_valid_mask:
      filename: depth_valid_mask_left_camera_left_lidar.png
      generated_at: '20260502_214533'
    rgb_vignette_mask:
      filename: rgb_vignette_mask_left_camera.png
      generated_at: '20260502_214533'
results:
  best_delta_transform_array:
  - 0.0125
  - -0.0034
  - 0.0011
  - 0.0452
  - 0.0121
  - -0.0210
  best_delta_transform_matrix:
  - [0.9997, 0.0210, 0.0121, 0.0125]
  - [-0.0210, 0.9988, 0.0452, -0.0034]
  - [-0.0111, -0.0455, 0.9989, 0.0011]
  - [0.0, 0.0, 0.0, 1.0]
system_profile:
  hostname: stretch-se4-4010
  machine: x86_64
  os: Linux
```

### Mathematical Interpretation and Application

The optimized transformation output under `results.best_delta_transform_matrix` (often referred to as `T_delta`) represents a highly precise 6D rigid body correction to the *camera's pose in the robot's base frame*. 

To project LiDAR points into the camera frame, you typically multiply a 3D point in the base frame by the camera's extrinsic matrix:
```
P_camera = T_base_to_cam @ P_base
```

The optimizer is parameterized to find a positional shift `T_delta` for the camera relative to its initial calibrated position. To correctly apply this correction, you must pre-multiply the camera's original extrinsic matrix by the **inverse** of the optimized transform:

```python
import numpy as np

# Load original extrinsics and optimization delta
T_base_to_cam_original = ... 
T_delta = np.array(yaml_data["results"]["best_delta_transform_matrix"])

# Compute corrected camera extrinsics
T_base_to_cam_corrected = np.linalg.inv(T_delta) @ T_base_to_cam_original

# Project points using the corrected extrinsics
P_camera_corrected = T_base_to_cam_corrected @ P_base
```

**Using the Shared Utilities:**
To make this process seamless for downstream applications, the repository provides the `ExtrinsicsCalibration` helper class in `stretch4_emulated_rgbd.shared_utils`.

```python
from stretch4_emulated_rgbd.shared_utils import ExtrinsicsCalibration

# Automatically load the inverse transform and manage the math
calibration = ExtrinsicsCalibration.load_from_yaml("path/to/optimization_results.yaml")

# Apply cleanly to your original transform
T_base_to_cam_corrected = calibration.apply_to_camera_extrinsics(T_base_to_cam_original)
```

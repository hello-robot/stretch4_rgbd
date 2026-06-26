"""
This configuration file is intended to hold the key parameters relevant to emulated RGB-D generation with the Stretch 4 robot. 
"""

# ==============================================================================
# GENERAL CONFIGURATION
# ==============================================================================

# If True, enables the legacy stretch4_body emulated RGB-D streamer fallback.
# If False, disables stretch4_body and restricts the streamer to only the 
# FastEmulatedRGBDStreamer (requires use_left=True, use_left_lidar=True).
USE_STRETCH_BODY_EMULATED_RGBD = False

# ==============================================================================
# RIGID BODY EXTRINSIC OPTIMIZATION (optimize_extrinsics.py)
# ==============================================================================

# Maximum allowed translational shift (in meters) for the RGB camera pose.
# If the optimizer proposes a translation larger than this, it receives a massive penalty.
MAX_TRANSLATION_M = 0.03

# Maximum allowed rotational shift (in degrees) for the RGB camera pose.
# If the optimizer proposes a rotation larger than this, it receives a massive penalty.
MAX_ROTATION_DEG = 10.0

# Penalty weight applied when the optimizer exceeds the bounds above.
BOUNDS_PENALTY_WEIGHT = 1e6

# Default CMA-ES Hyperparameters for Extrinsics
# Standard deviation for the translational parameters (meters) during CMA-ES sampling.
EXTRINSICS_CMA_TRANSLATION_SIGMA_M = 0.005
# Standard deviation for the rotational parameters (degrees) during CMA-ES sampling.
EXTRINSICS_CMA_ROTATION_SIGMA_DEG = 0.5
EXTRINSICS_CMA_POPSIZE = 10 #10 #30
EXTRINSICS_CMA_MAXITER = 1000 #1000

# CMA-ES Early Stopping Hyperparameters
# Tolerance on the function value. Stop if range of function values in the last
# generation and the difference between best and worst function values are below this.
EXTRINSICS_CMA_TOLFUN = 1e-4 #1e-3 #1e-4

# Tolerance on the variables. Stop if step size and standard deviations of the
# variables are below this threshold.
EXTRINSICS_CMA_TOLX = 1e-4 #1e-3 #1e-4

# Use Normalized Mutual Information (NMI) instead of standard Mutual Information (MI).
# NMI is generally more robust to varying overlap areas between modalities.
EXTRINSICS_USE_NMI = True

# Kernel size (in pixels) for the Sobel gradient operator used during optimization.
# Must be an odd integer (e.g., 3, 5, 7). Larger sizes capture broader edge structures.
EXTRINSICS_GRAD_KSIZE = 5

# Number of histogram bins used to compute Mutual Information between gradients.
EXTRINSICS_MI_BINS = 50

# If True, starts optimization from factory calibration instead of using the prior optimum.
EXTRINSICS_IGNORE_PRIOR_OPTIMIZATIONS = True #False

# ==============================================================================
# DEPTH ALIGNMENT & SPATIAL CORRECTIONS
# ==============================================================================

# If True, generates depth validity mask based on density of projected LiDAR points.
# If False, uses a distance-transform approach.
USE_DENSITY_BASED_DEPTH_MASK = False

# Window size (in pixels) used to compute the local density of valid LiDAR points.
DEPTH_MASK_DENSITY_WINDOW_SIZE = 75

# Ratio used to calculate the threshold density.
# Threshold = median_density * DEPTH_MASK_DENSITY_THRESHOLD_RATIO
# 0.5 means density must be at least half the median to be valid.
DEPTH_MASK_DENSITY_THRESHOLD_RATIO = 0.2

# Kernel size (in pixels) for morphological closing to fill holes in the mask.
DEPTH_MASK_CLOSING_KERNEL_SIZE = 75

# Number of iterations to apply the morphological closing operation.
DEPTH_MASK_CLOSING_ITERATIONS = 3

# Grayscale threshold (0-255) above which a pixel is considered saturated/overexposed.
# Saturated regions are excluded from edge optimization and their depth is forced to match LiDAR.
SATURATION_THRESHOLD_GRAY = 250

# Maximum distance (in pixels) that a sparse LiDAR depth value can be interpolated.
# Interpolated pixels beyond this distance will be marked as invalid (0.0).
MAX_LIDAR_INTERPOLATION_DIST_PX = 75.0

# The aggressive distance threshold used to constrain the outer boundary of the valid depth region.
# This keeps the outer edge of the mask tight to the LiDAR data while allowing larger interior gaps to be filled.
OUTER_BOUNDARY_DISTANCE_THRESH_PX = 10.0

# The erosion radius used to identify the outer boundary region. Should generally match MAX_LIDAR_INTERPOLATION_DIST_PX.
OUTER_BOUNDARY_EROSION_RADIUS = 75

# Absolute physical depth boundaries (in meters) for the synthetic depth map.
# Values outside this range will be clipped to these boundaries.
MIN_PHYSICAL_DEPTH_M = 0.1
MAX_PHYSICAL_DEPTH_M = 10.0 # 60 m is the specified maximum range for the Hesai JT128

# ==============================================================================
# SPARSITY SHADOW FILTER (Z-BUFFERING FOR SPARSE POINTS)
# ==============================================================================

# Enable or disable the shadow filter which removes occluded background points 
# that "bleed" into foreground objects due to LiDAR sparsity.
ENABLE_SHADOW_FILTER = True

# The size of the local neighborhood window (in pixels) to search for closer foreground points.
# Larger windows handle greater sparsity but may erroneously remove points seen through real gaps.
# Must be an odd integer (e.g., 3, 5, 7).
SHADOW_FILTER_WINDOW_SIZE = 13 #5 #7 #11 #13 #15 #17

# The depth difference threshold (in meters) to consider a point "shadowed" by a foreground point.
# If a neighbor is closer by more than this threshold, the background point is removed.
SHADOW_FILTER_DEPTH_THRESHOLD_M = 0.3 #0.15 #0.2 #0.3

# ==============================================================================
# OPTIMIZE_EXTRINSICS SHADOW FILTER OVERRIDES
# ==============================================================================

# Enable or disable the shadow filter specifically during extrinsic optimization (optimize_extrinsics.py)
OPTIMIZE_EXTRINSICS_ENABLE_SHADOW_FILTER = False

# The size of the local neighborhood window for extrinsic optimization
OPTIMIZE_EXTRINSICS_SHADOW_FILTER_WINDOW_SIZE = 13

# The depth difference threshold for extrinsic optimization
OPTIMIZE_EXTRINSICS_SHADOW_FILTER_DEPTH_THRESHOLD_M = 0.3

# If True, uses a circular (elliptical) structuring element instead of a square for the filter.
# A circular window provides more isotropic filtering and reduces blocky artifacts around edges,
# which can be advantageous when using larger window sizes (e.g., >= 5).
# Note: Circular kernels are non-separable and introduce latency, so they are disabled by default.
SHADOW_FILTER_USE_CIRCULAR_WINDOW = True

# ==============================================================================
# VISUALIZATION & DENSE RGB-D OVERLAY
# ==============================================================================

# Maximum physical depth (in meters) to use when visualizing depth maps.
# Note: Used in OpenCV visualizations (e.g. cv2.imshow)
VISUALIZATION_MAX_DEPTH_M = 3.0

# Maximum depth (in meters) to use for color mapping in ReRun's DepthImage visualizations.
RERUN_COLOR_MAX_DEPTH_M = 10.0

# Alpha blending value (0.0 to 1.0) for the dense RGB-D overlay visualization.
# Lower values make the depth map more transparent.
DENSE_RGBD_ALPHA = 0.5

# ==============================================================================
# VALIDITY MASK GENERATION
# ==============================================================================

# Grayscale intensity threshold (0-255) for finding the valid interior of the camera vignette.
VIGNETTE_MASK_THRESHOLD = 20

# Size of the erosion kernel (in pixels) used to aggressively shrink the vignette mask 
# away from the blurry vignette boundary.
VIGNETTE_MASK_EROSION_KERNEL_SIZE = 11

# Size of the dilation kernel (in pixels) used to expand the saturation mask slightly
# to cover blooming edges around extremely bright pixels.
SATURATION_MASK_DILATION_KERNEL_SIZE = 5

# ==============================================================================
# NATIVE IMAGE ORIENTATION
# ==============================================================================

# Enable this to rotate the raw fisheye images from the head cameras by 90 degrees 
# to a vertical orientation early in the pipeline.
ROTATE_IMAGES_TO_VERTICAL = True

# If True, attempts to perform the image rotation on the Luxonis OAK-FFC board itself 
# using hardware acceleration. 
# 
# HARDWARE LIMITATIONS:
# 1. The 600p resolution is incompatible with MJPEG compression when rotated, 
#    as the rotated width (600) is not a multiple of 16.
# 2. The Luxonis board (Myriad X VPU) lacks a zero-cost 90-degree memory transpose 
#    operation for the NV12 format. It routes rotations through a generic hardware 
#    warp engine (ImageManip) which consumes significant compute cycles. 
#    Attempting to process high resolutions at high frame rates will overwhelm the SHAVE 
#    cores and introduce severe latency. When enabled, camera_fps is restricted to <= 10.
# 
# STRONGLY RECOMMENDED: Set USE_BOARD_LEVEL_ROTATION = False. The software fallback 
# provides highly efficient (<1ms) 90-degree rotation on the host CPU. 
# It utilizes a "lazy evaluation" architecture where the pipeline maintains and 
# transports the hardware-compressed MJPEG frames natively over the network. 
# The developer receives an RGBDFrame object, and the 90-degree rotation is 
# transparently applied on-the-fly exactly when `frame.image` or `frame.depth_image` 
# is accessed, completely insulating the user from the native orientation.
USE_BOARD_LEVEL_ROTATION = False

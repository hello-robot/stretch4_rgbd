try:
    import depthai as dai
except ImportError:
    dai = None
import numpy as np
import time
import os
import json
import tempfile
import threading
import collections

from stretch4_emulated_rgbd import emulated_rgbd_config as config
from stretch4_emulated_rgbd.shared_utils import get_rotated_intrinsics

CACHE_FILE_PATH = os.path.join(tempfile.gettempdir(), f"luxonis_device_cache_{os.getuid()}.json")
CACHE_EXPIRY_SECONDS = 24 * 60 * 60  # Invalidate after 24 hours

def _load_cache():
    try:
        if os.path.exists(CACHE_FILE_PATH):
            with open(CACHE_FILE_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load device cache: {e}")
    return {}

def _save_cache(cache_data):
    try:
        with open(CACHE_FILE_PATH, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        print(f"Warning: Failed to save device cache: {e}")

def get_device_port_by_product_name(product_name:str):
    """When multiple Luxonis cameras are connected, we should query their serial number to use them."""
    devices = dai.Device.getAllAvailableDevices()

    cache_data = _load_cache()
    current_time = time.time()

    # Check cache first
    for info in devices:
        cached_device = cache_data.get(info.deviceId)
        if cached_device:
            # Check expiry
            if current_time - cached_device.get("timestamp", 0) <= CACHE_EXPIRY_SECONDS:
                if cached_device.get("product_name") == product_name:
                    return info.name

    # If not found in cache, fallback and update cache
    for info in devices:
        # Skip if we already have valid cache for this specific device
        cached_device = cache_data.get(info.deviceId)
        if cached_device and current_time - cached_device.get("timestamp", 0) <= CACHE_EXPIRY_SECONDS:
            continue

        with dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS, nameOrDeviceId=info.deviceId) as device:
            actual_product_name = device.getProductName()
            
            cache_data[info.deviceId] = {
                "product_name": actual_product_name,
                "timestamp": current_time,
                "name": info.name
            }
            _save_cache(cache_data)

            if product_name == actual_product_name:
                device.close()
                time.sleep(1) # Sleep is needed so that device goes back to ready state, it's quite slow.
                return info.name
            
    raise RuntimeError(f"Could not find the {product_name} device. Is it connected?")


class HeadCamera:
    """
    A low-latency wrapper for the Stretch 4 Head Cameras.
    Currently optimized for the Left Fisheye camera (OAK-FFC AR0234 M12).
    
    FRAME RATE & BUFFERING STRATEGY:
    To maintain a strict 10Hz frame rate without latency bloat or memory exhaustion, several 
    OAK-D / DepthAI pipeline optimizations are enforced:
    
    1. Minimal Non-Blocking Queues: By creating an output queue with `maxSize=1` and `blocking=False`, 
       the camera will overwrite unread frames. If the host PC stalls, it drops the old frame instead 
       of buffering it. The host ALWAYS pulls the absolute newest physical frame.
    2. Restricted Memory Pools: `setNumFramesPools` is strictly bounded to the `oak_buffer_size`. 
       This prevents the internal ISP (Image Signal Processor) from queuing up hidden frames in memory.
    3. MJPEG USB Compression: At 10Hz, passing raw 1200p or 800p uncompressed frames can saturate 
       the USB bus alongside Hesai LiDAR UDP traffic. The hardware MJPEG VideoEncoder compresses 
       the stream on the camera board, ensuring deterministic transmission times.
    4. Background Ring Buffer: A daemon thread continuously drains the non-blocking queue into a 
       `collections.deque` history buffer. This allows the pipeline to decouple the camera's hardware 
       frame rate from the pipeline's output rate, enabling software over-sampling (e.g. running the 
       camera at 30Hz) to minimize phase latency without blocking.
    """
    def __init__(self, device_id=None, fps=10, resolution_height=800, compress=True, oak_buffer_size=1):
        if device_id is None or device_id == "3.3.1":
            self.device_id = get_device_port_by_product_name("OAK-FFC-3P")
        else:
            self.device_id = device_id
            
        self.fps = fps
        self.resolution_height = resolution_height
        self.compress = compress
        self.oak_buffer_size = oak_buffer_size
        
        # Mapping vertical resolution to (width, height)
        res_map = {
            400: (640, 400),
            600: (960, 600),
            800: (1280, 800),
            1200: (1920, 1200)
        }
        if self.resolution_height not in res_map:
            raise ValueError(f"Invalid resolution height {self.resolution_height}. Supported: {list(res_map.keys())}")
        
        self.image_size = res_map[self.resolution_height]
        
        self.device = dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS, nameOrDeviceId=self.device_id)
        self.pipeline = dai.Pipeline(defaultDevice=self.device)
        
        # Optional: set chunk size to 0 for lower latency
        self.pipeline.setXLinkChunkSize(0)

        # Left camera is CAM_C on the OAK-FFC 3P board
        self.cam_left = self.pipeline.create(dai.node.Camera)
        self.cam_left.setSensorType(dai.CameraSensorType.COLOR)
        self.cam_left.build(boardSocket=dai.CameraBoardSocket.CAM_C, sensorFps=self.fps)
        
        # Add buffer size limits to the camera node
        self.cam_left.setNumFramesPools(isp=self.oak_buffer_size + 1, raw=self.oak_buffer_size + 1, imgmanip=self.oak_buffer_size + 1)
        
        # Request full 16:10 output
        self.out_left = self.cam_left.requestOutput(
            size=self.image_size,
            type=dai.ImgFrame.Type.NV12,
            resizeMode=dai.ImgResizeMode.CROP,
            enableUndistortion=False,
        )

        source_for_next_stage = self.out_left

        if config.USE_BOARD_LEVEL_ROTATION:
            if self.resolution_height == 600 and self.compress:
                raise ValueError("Board-level rotation is incompatible with 600p resolution when using MJPEG compression because the rotated width (600) is not a multiple of 16. Please disable board-level rotation or choose a different resolution (e.g., 800p).")
            
            if self.fps > 10:
                raise ValueError(
                    f"Camera FPS of {self.fps} is too high for board-level rotation. "
                    "The Luxonis board (Myriad X VPU) lacks a zero-cost 90-degree memory transpose operation "
                    "for the NV12 format. Instead, it relies on a generic hardware warp engine (ImageManip) "
                    "which is highly compute-intensive. At frame rates above 10 FPS, the SHAVE cores cannot "
                    "keep up, resulting in severe pipeline latency and frame drops. "
                    "Please select a camera_fps <= 10 (e.g., 10) or disable USE_BOARD_LEVEL_ROTATION."
                )
            else:
                print("\n\033[93m" + "="*80)
                print("WARNING: USE_BOARD_LEVEL_ROTATION is enabled.")
                print("This feature routes 90-degree NV12 image rotations through the Luxonis generic")
                print("hardware warp engine (ImageManip), which consumes significant compute cycles.")
                print("This introduces noticeable latency and synchronization issues between the RGB")
                print("and depth components. It is STRONGLY RECOMMENDED to set USE_BOARD_LEVEL_ROTATION")
                print("to False in emulated_rgbd_config.py to leverage the highly efficient (<1ms)")
                print("software rotation fallback instead.")
                print("="*80 + "\033[0m\n")
                
            self.manip = self.pipeline.create(dai.node.ImageManip)
            self.manip.initialConfig.addRotateDeg(270)
            self.manip.setMaxOutputFrameSize(self.image_size[0] * self.image_size[1] * 3)
            
            # Configure the input queue to drop frames if the SHAVE cores can't keep up
            # with the 270-degree rotation, to prioritize low latency over frame rate.
            self.manip.inputImage.setBlocking(False)
            self.manip.inputImage.setMaxSize(1)
            
            self.out_left.link(self.manip.inputImage)
            source_for_next_stage = self.manip.out

        if self.compress:
            self.videoEnc = self.pipeline.create(dai.node.VideoEncoder)
            self.videoEnc.setDefaultProfilePreset(self.fps, dai.VideoEncoderProperties.Profile.MJPEG)
            self.videoEnc.setQuality(80)
            self.videoEnc.setNumFramesPool(self.oak_buffer_size + 1)
            source_for_next_stage.link(self.videoEnc.input)
            self.q_left = self.videoEnc.bitstream.createOutputQueue(maxSize=self.oak_buffer_size, blocking=False)
        else:
            self.q_left = source_for_next_stage.createOutputQueue(maxSize=self.oak_buffer_size, blocking=False)

        self.history_size = max(100, self.fps * 2) # Buffer 2 seconds worth
        self.history_buffer = collections.deque(maxlen=self.history_size)
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        """Starts the DepthAI pipeline and the background poller thread."""
        self.pipeline.start()
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stops the pipeline and background thread."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        if self.device is not None:
            self.device.close()

    def get_intrinsics(self, camera_name="head_left"):
        """Returns the camera matrix and distortion coefficients."""
        M, D = None, None
        if self.device is not None:
            try:
                calib = self.device.readCalibration()
                M = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, self.image_size[0], self.image_size[1]), dtype=np.float64)
                D = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C), dtype=np.float64)
            except Exception as e:
                print(f"Warning: could not read factory calibration from OAK-FFC: {e}. Falling back to fleet calibration.")
                
                try:
                    from stretch4_body.subsystem.cameras.models.camera_calibration import RGBCameraCalibration
                    from stretch4_body.subsystem.cameras import RGBCameras
                    fleet_calib = RGBCameraCalibration.load_calibration_from_fleet_path(
                        camera_type=RGBCameras[camera_name], is_flip_width_and_height=False
                    )
                    if fleet_calib and fleet_calib.camera_matrix is not None:
                        M = np.array(fleet_calib.camera_matrix, dtype=np.float64)
                        D = np.array(fleet_calib.distortion_coefficients, dtype=np.float64)
                        
                        # Scale the intrinsic matrix to the requested resolution
                        orig_h, orig_w = fleet_calib.height, fleet_calib.width
                        scale_x = self.image_size[0] / orig_w
                        scale_y = self.image_size[1] / orig_h
                        
                        M[0, 0] *= scale_x
                        M[1, 1] *= scale_y
                        M[0, 2] *= scale_x
                        M[1, 2] *= scale_y
                        
                except Exception as ex:
                    print(f"Error loading fleet calibration: {ex}")

        return M, D

    def _poll_loop(self):
        """Continuously drains the queue in the background."""
        while self.running:
            msg = self.q_left.tryGet()
            if msg is not None:
                img_data = None
                if self.compress:
                    import cv2
                    img_data = msg.getData()
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                else:
                    img = msg.getCvFrame()
                
                # Timestamp synced with host monotonic clock
                timestamp = msg.getTimestamp().total_seconds()
                seq_num = msg.getSequenceNum()
                
                with self.lock:
                    self.history_buffer.append((img, timestamp, seq_num, img_data))
            else:
                time.sleep(0.005)

    def get_closest_frame(self, target_timestamp):
        """Returns the frame (img, timestamp, seq, img_data) closest to the target_timestamp."""
        with self.lock:
            if not self.history_buffer:
                return None, None, None, None
                
            closest_frame = min(self.history_buffer, key=lambda x: abs(x[1] - target_timestamp))
            return closest_frame

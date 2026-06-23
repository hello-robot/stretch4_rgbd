import time
import threading
import collections
try:
    from stretch4_pyhesai_wrapper.stream_lidar import stream_lidar_left_blocking
except ImportError:
    stream_lidar_left_blocking = None

class LidarPoller:
    """
    Continuously polls the PyHesai LiDAR stream in a background thread and maintains a history buffer
    of recent point clouds.
    
    MOTIVATION & TEMPORAL CONSIDERATIONS:
    Achieving a stable 10Hz emulated RGB-D stream requires strict temporal alignment between the 
    instantaneous global shutter RGB image and the 100ms rolling mechanical sweep of the LiDAR. 
    
    1. UDP Buffer Bloat: If the main thread stutters, UDP queues in the driver build up. If we 
       timestamp frames upon python-level arrival, latency can bloat up to 300ms.
       Fix: We bypass Python arrival times and use the LiDAR's internal hardware timestamps.
    
    2. Hardware Clock Offset: The LiDAR hardware clock is desynchronized from the host monotonic 
       clock used by the Luxonis cameras (often by several seconds).
       Fix: We dynamically estimate the stable transmission offset (`clock_offset`) to precisely 
       translate the LiDAR hardware time into host monotonic time.
    """
    def __init__(self, history_size=100):
        self.history_size = history_size
        self.history_buffer = collections.deque(maxlen=self.history_size)
        self.lock = threading.Lock()
        self.running = True
        
        # Used to synchronize LiDAR's internal hardware clock to host monotonic clock
        self.clock_offset = -9999999.0
        
        self.thread = threading.Thread(target=self._poll_loop)
        self.thread.daemon = True
        self.thread.start()

    def _poll_loop(self):
        try:
            for lidar_frame in stream_lidar_left_blocking():
                if not self.running:
                    break
                if lidar_frame is not None and hasattr(lidar_frame, 'timestamp') and len(lidar_frame.timestamp) > 0:
                    now = time.monotonic()
                    ts_end = lidar_frame.timestamp[-1]
                    ts_start = lidar_frame.timestamp[0]
                    
                    # The frame processed with the LEAST UDP buffer delay will have the largest (ts_end - now)
                    diff = ts_end - now
                    if diff > self.clock_offset:
                        self.clock_offset = diff
                        
                    # Calculate the physical host time that the sweep started and ended
                    host_ts_start = ts_start - self.clock_offset
                    host_ts_end = ts_end - self.clock_offset
                    
                    # We use the midpoint of the sweep as the representative timestamp for the frame
                    host_ts_mid = (host_ts_start + host_ts_end) / 2.0
                    
                    with self.lock:
                        self.history_buffer.append((host_ts_mid, host_ts_end, lidar_frame))
        except Exception as e:
            print(f"LidarPoller encountered an error: {e}")

    def stop(self):
        """Stops the poller thread."""
        self.running = False
        self.thread.join()

    def get_closest_frame(self, target_timestamp, max_time_diff=0.5, timeout=0.15):
        """
        Retrieves the LiDAR frame whose mathematical sweep midpoint is closest to `target_timestamp`.
        
        TEMPORAL LOOKAHEAD JUSTIFICATION:
        Because a 360-degree LiDAR sweep takes ~100ms (at 10Hz), the sweep perfectly centered on an
        instantaneous RGB image captured at time `T` is the sweep covering `[T - 50ms, T + 50ms]`. 
        This means the physical hardware has not finished painting the scene until `T + 50ms`. 
        
        If we naively fetch the closest frame instantly at time `T`, we will pull the *previous* 
        sweep ending at `T - 50ms`, injecting 100ms of structural latency. Instead, this function 
        efficiently blocks and waits for the ideal future frame to finish rotating and arrive over 
        the network. This adds a bounded ~50ms delay to the pipeline, but guarantees near-perfect 
        spatial-temporal synchronization between the camera and LiDAR for moving objects.
        """
        start_wait = time.monotonic()
        while time.monotonic() - start_wait < timeout:
            with self.lock:
                if self.history_buffer:
                    latest_mid, latest_end, _ = self.history_buffer[-1]
                    # If the latest frame's end time is past the target timestamp, 
                    # it means the physical sweep covering the target has completed!
                    if latest_end >= target_timestamp:
                        break
            # Yield briefly to the LiDAR thread
            time.sleep(0.005)

        with self.lock:
            if not self.history_buffer:
                return None
            
            closest_frame = None
            min_diff = float('inf')
            
            for mid_ts, end_ts, frame in self.history_buffer:
                diff = abs(mid_ts - target_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest_frame = frame
                    
            if min_diff > max_time_diff:
                return None
                
            return closest_frame

    def wait_for_next_frame(self, last_mid_ts=None, timeout=1.0):
        """
        Blocks until a new LiDAR frame (whose midpoint timestamp is greater than `last_mid_ts`)
        is available, and returns (mid_ts, end_ts, frame).
        """
        start_wait = time.monotonic()
        while time.monotonic() - start_wait < timeout:
            with self.lock:
                if self.history_buffer:
                    latest_mid, latest_end, latest_frame = self.history_buffer[-1]
                    if last_mid_ts is None or latest_mid > last_mid_ts:
                        return latest_mid, latest_end, latest_frame
            time.sleep(0.005)
        return None, None, None

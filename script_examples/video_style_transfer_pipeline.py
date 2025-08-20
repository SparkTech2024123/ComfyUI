#!/usr/bin/env python3
"""
Video Style Transfer Pipeline
============================

Optimized and refactored video processing pipeline for style transfer operations.
This script integrates all tested component functionalities to create a complete
video processing pipeline with improved code organization and maintainability.

Key Features:
- Clean separation of concerns with focused classes
- Consolidated server management and selection logic
- Comprehensive error handling and retry mechanisms
- Multi-threaded processing with intelligent load balancing
- Detailed statistics tracking and reporting
- Memory management and resource cleanup

Input Requirements:
- Video file: ./input/AAAzhenhuan.mp4
- Shot detection: ./input/AAAzhenhuan_shot_detection.json

Author: Refactored from test_comprehensive_video_pipeline.py
"""

import os
import sys
import time
import json
import uuid
import threading
import urllib.request
import urllib.error
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Tuple, List, Dict, Optional
from multiprocessing import shared_memory
from PIL import Image
import cv2
import tempfile
import glob
import json

# Add ComfyUI root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
comfyui_root = os.path.dirname(current_dir)
if comfyui_root not in sys.path:
    sys.path.insert(0, comfyui_root)

# Import ComfyUI video segmentation function
from script_examples.video_stylize_depends.video_segmentation_api import comfyui_video_segmentation


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

class PipelineConfig:
    """Centralized configuration management for the video pipeline."""
    
    # File paths
    TEST_VIDEO_PATH = "./input/AAAzhenhuan.mp4"
    TEST_SHOT_DETECTION_PATH = "./input/AAAzhenhuan_shot_detection.json"
    OUTPUT_DIR = "./output"
    
    # Style configuration
    DEFAULT_STYLE_DESCRIPTION = "Rick and Morty Anime"
    
    # Server configuration
    STYLE_TRANSFER_SERVERS = [
        "127.0.0.1:8281", "127.0.0.1:8282", "127.0.0.1:8283", "127.0.0.1:8284",
        "127.0.0.1:8285", "127.0.0.1:8286", "127.0.0.1:8287", "127.0.0.1:8288"
    ]
    
    # Timeout and retry settings
    RETRY_ATTEMPTS = 5
    RETRY_DELAY = 5
    SERVER_HEALTH_CHECK_TIMEOUT = 10
    WORKFLOW_EXECUTION_TIMEOUT_STAGE1 = 300   # 5 minutes
    WORKFLOW_EXECUTION_TIMEOUT_STAGE2 = 3000  # 50 minutes
    
    # Server selection settings
    SERVER_POLLING_INTERVAL = 2.0
    SERVER_MAX_WAIT_TIME = 300.0
    SERVER_RETRY_POLLING_INTERVAL = 1.0
    SERVER_RETRY_MAX_WAIT_TIME = 120.0
    
    # Processing settings
    DEFAULT_MAX_WORKERS = 5
    DEFAULT_FAILURE_TOLERANCE = 0.5
    
    @classmethod
    def get_client_id(cls) -> str:
        """Generate a unique client ID for the session."""
        return str(uuid.uuid4())


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def import_stage1_functions():
    """Import proven Stage 1 functions from test_stage1_style_transfer.py"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "test_stage1", 
            os.path.join(current_dir, "video_stylize_depends", "test_stage1_style_transfer.py")
        )
        stage1_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage1_module)
        
        return {
            'process_stage1_style_transfer': stage1_module.process_stage1_style_transfer,
            'select_best_server': stage1_module.select_best_server,
            'cleanup_shared_memory_by_pattern': stage1_module.cleanup_shared_memory_by_pattern,
            'cleanup_temp_files': stage1_module.cleanup_temp_files
        }
    except Exception as e:
        raise ImportError(f"Failed to import Stage 1 functions: {e}")


def import_stage2_functions():
    """Import proven Stage 2 functions from test_stage2_video_processing.py"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "test_stage2",
            os.path.join(current_dir, "video_stylize_depends", "test_stage2_video_processing.py")
        )
        stage2_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage2_module)
        
        return {
            'process_stage2_video_processing': stage2_module.process_stage2_video_processing,
            'create_shared_memory_for_image': stage2_module.create_shared_memory_for_image,
            'cleanup_shared_memory_segments': stage2_module.cleanup_shared_memory_segments,
            'save_processed_frames': stage2_module.save_processed_frames
        }
    except Exception as e:
        raise ImportError(f"Failed to import Stage 2 functions: {e}")


# =============================================================================
# EXTRACTED CACHE AND SERVER FUNCTIONS
# =============================================================================

def clear_model_cache(server_address: str, quiet: bool = False):
    """
    Clear model and node cache on ComfyUI server

    Args:
        server_address: ComfyUI server address
        quiet: If True, suppress output messages
    """
    try:
        # Call the free memory endpoint with proper request data
        free_url = f"http://{server_address}/free"

        # Add correct request body data as expected by ComfyUI
        request_data = {
            "unload_models": True,
            "free_memory": True
        }

        data = json.dumps(request_data).encode('utf-8')
        req = urllib.request.Request(free_url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')

        with urllib.request.urlopen(req, timeout=10) as response:
            if response.getcode() == 200:
                if not quiet:
                    print(f"✓ Cleared model cache on {server_address}")
            else:
                if not quiet:
                    print(f"⚠ Cache clear returned status {response.getcode()} on {server_address}")

    except Exception as e:
        if not quiet:
            print(f"⚠ Failed to clear cache on {server_address}: {e}")
        # Don't raise exception - cache clearing is non-critical


def check_server_status(server_address):
    """
    Check ComfyUI server status
    """
    try:
        # Check queue status
        queue_url = f"http://{server_address}/queue"
        queue_req = urllib.request.Request(queue_url)
        queue_req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(queue_req, timeout=3) as response:
            queue_data = json.loads(response.read())
        
        # Check system status
        stats_url = f"http://{server_address}/system_stats"
        stats_req = urllib.request.Request(stats_url)
        
        with urllib.request.urlopen(stats_req, timeout=3) as response:
            system_data = json.loads(response.read())
        
        # Calculate server load
        queue_running = len(queue_data.get('queue_running', []))
        queue_pending = len(queue_data.get('queue_pending', []))
        total_load = queue_running + queue_pending
        
        # Get VRAM usage
        vram_free = 0
        vram_total = 0
        if 'devices' in system_data and len(system_data['devices']) > 0:
            device = system_data['devices'][0]
            vram_free = device.get('vram_free', 0)
            vram_total = device.get('vram_total', 1)
        
        vram_usage_ratio = 1 - (vram_free / vram_total) if vram_total > 0 else 1
        
        return {
            'server_address': server_address,
            'queue_running': queue_running,
            'queue_pending': queue_pending,
            'total_load': total_load,
            'vram_free': vram_free,
            'vram_total': vram_total,
            'vram_usage_ratio': vram_usage_ratio,
            'available': True
        }
        
    except Exception as e:
        print(f"Failed to check server {server_address}: {e}")
        return {
            'server_address': server_address,
            'available': False,
            'error': str(e)
        }


# Global variable to track server selection for round-robin (thread-safe)
_server_index = 0
_server_index_lock = threading.Lock()

# Server configuration for compatibility with other modules
COMFYUI_SERVERS = [
    "127.0.0.1:8281", "127.0.0.1:8282", "127.0.0.1:8283", "127.0.0.1:8284",
    "127.0.0.1:8285", "127.0.0.1:8286", "127.0.0.1:8287", "127.0.0.1:8288"
]


def select_best_server(servers=None, use_round_robin=False, quiet=False):
    """
    Select best ComfyUI server based on load and VRAM usage
    
    Args:
        servers: List of server addresses to check
        use_round_robin: If True, use round-robin for servers with same load
        quiet: If True, suppress detailed output to reduce terminal clutter
    """
    global _server_index
    
    if servers is None:
        servers = PipelineConfig.STYLE_TRANSFER_SERVERS
    
    if not quiet:
        print("=== Checking ComfyUI servers status (8281-8287) ===")
    available_servers = []
    
    for server in servers:
        status = check_server_status(server)
        if status['available']:
            available_servers.append(status)
            if not quiet:
                print(f"✓ {server}: load={status['total_load']}, vram_free={status['vram_free']/(1024**3):.1f}GB")
        else:
            if not quiet:
                print(f"✗ {server}: {status.get('error', 'unavailable')}")
    
    if not available_servers:
        print("No available ComfyUI servers found!")
        return None
    
    if use_round_robin:
        # Group servers by load level (prioritize low load)
        min_load = min(s['total_load'] for s in available_servers)
        low_load_servers = [s for s in available_servers if s['total_load'] == min_load]
        
        # Use round-robin among servers with same (lowest) load (thread-safe)
        if low_load_servers:
            with _server_index_lock:
                selected_server = low_load_servers[_server_index % len(low_load_servers)]
                _server_index += 1
        else:
            selected_server = available_servers[0]
    else:
        # Select server with lowest load, then lowest VRAM usage
        selected_server = min(available_servers, key=lambda x: (x['total_load'], x['vram_usage_ratio']))
    
    if not quiet:
        print(f"Selected server: {selected_server['server_address']} (load={selected_server['total_load']})")
        print("=" * 50)
    
    return selected_server['server_address']


def import_cache_clearing_functions():
    """Return local cache clearing functions"""
    return {
        'clear_model_cache': clear_model_cache,
        'select_best_server': select_best_server
    }


def import_combination_functions():
    """Import proven combination functions from test_combine_processed_segments_improved.py"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "test_combination",
            os.path.join(current_dir, "video_stylize_depends", "test_combine_processed_segments_improved.py")
        )
        combination_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(combination_module)
        
        return {
            'combine_processed_segments_improved': combination_module.combine_processed_segments_improved
        }
    except Exception as e:
        raise ImportError(f"Failed to import combination functions: {e}")


def validate_test_inputs(config: PipelineConfig) -> bool:
    """Validate that required test input files exist"""
    print("🔍 Validating test input files...")
    
    missing_files = []
    
    if not os.path.exists(config.TEST_VIDEO_PATH):
        missing_files.append(config.TEST_VIDEO_PATH)
    
    if not os.path.exists(config.TEST_SHOT_DETECTION_PATH):
        missing_files.append(config.TEST_SHOT_DETECTION_PATH)
    
    if missing_files:
        print("❌ Missing required input files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure all input files are available before running the test.")
        return False
    
    print("✅ All required input files found")
    return True


def extract_frames_from_video(segment_path: str, config: PipelineConfig) -> Tuple[str, str, bool]:
    """
    Extract first and last frames from video segment with fallback logic.
    
    Returns:
        Tuple of (first_frame_path, last_frame_path, are_temporary)
    """
    segment_basename = os.path.splitext(os.path.basename(segment_path))[0]

    # Check for pre-extracted frames from video segmentation stage
    video_segments_frames_dir = os.path.join(config.OUTPUT_DIR, "video_segments", "frames")
    pre_extracted_first = os.path.join(video_segments_frames_dir, f"{segment_basename}_first.jpg")
    pre_extracted_last = os.path.join(video_segments_frames_dir, f"{segment_basename}_last.jpg")

    if os.path.exists(pre_extracted_first) and os.path.exists(pre_extracted_last):
        return pre_extracted_first, pre_extracted_last, False

    # Check for manually extracted frames
    frames_dir = os.path.join(config.OUTPUT_DIR, "video_segments", "extracted_frames")
    os.makedirs(frames_dir, exist_ok=True)

    first_frame_path = os.path.join(frames_dir, f"{segment_basename}_first.jpg")
    last_frame_path = os.path.join(frames_dir, f"{segment_basename}_last.jpg")

    if os.path.exists(first_frame_path) and os.path.exists(last_frame_path):
        return first_frame_path, last_frame_path, False

    # Extract frames manually
    cap = cv2.VideoCapture(segment_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open segment video: {segment_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame")

        # Extract last frame
        last_frame = None
        for i in range(frame_count - 1, max(0, frame_count - 5), -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                last_frame = frame
                break

        if last_frame is None:
            raise RuntimeError("Cannot read last frame")

        # Convert BGR to RGB and save
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)

        Image.fromarray(first_frame_rgb).save(first_frame_path, quality=95)
        Image.fromarray(last_frame_rgb).save(last_frame_path, quality=95)

        return first_frame_path, last_frame_path, True

    finally:
        cap.release()


def save_stylized_frames_to_disk(stylized_first: np.ndarray, stylized_last: np.ndarray, 
                                segment_path: str, config: PipelineConfig) -> Tuple[str, str]:
    """
    Save stylized frames to disk for Stage 2 processing.
    
    Returns:
        Tuple of (stylized_first_path, stylized_last_path)
    """
    stylized_dir = os.path.join(config.OUTPUT_DIR, "video_segments", "stylized_frames")
    os.makedirs(stylized_dir, exist_ok=True)
    
    segment_basename = os.path.splitext(os.path.basename(segment_path))[0]
    unique_id = uuid.uuid4().hex[:8]
    
    first_path = os.path.join(stylized_dir, f"{segment_basename}_first_{unique_id}.jpg")
    last_path = os.path.join(stylized_dir, f"{segment_basename}_last_{unique_id}.jpg")
    
    # Normalize frames to [0-255] uint8
    def normalize_frame(frame):
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        return frame
    
    stylized_first = normalize_frame(stylized_first)
    stylized_last = normalize_frame(stylized_last)
    
    Image.fromarray(stylized_first).save(first_path, quality=95)
    Image.fromarray(stylized_last).save(last_path, quality=95)
    
    return first_path, last_path


# =============================================================================
# SERVER MANAGEMENT
# =============================================================================

class ServerManager:
    """Manages server selection, reservation, and monitoring."""
    
    def __init__(self, config: PipelineConfig, quiet_mode: bool = False):
        self.config = config
        self.quiet_mode = quiet_mode
        self._print_lock = threading.Lock()
        self._server_reservations = {}  # {server_address: worker_thread_name}
        self._server_reservation_lock = threading.Lock()
    
    def _safe_print(self, message: str):
        """Thread-safe printing with quiet mode support"""
        if not self.quiet_mode:
            with self._print_lock:
                print(message)
    
    def monitor_server_utilization(self) -> Dict:
        """Monitor utilization across all ComfyUI servers"""
        server_stats = {}
        
        for server_address in self.config.STYLE_TRANSFER_SERVERS:
            try:
                queue_url = f"http://{server_address}/queue"
                queue_req = urllib.request.Request(queue_url)
                queue_req.add_header('Content-Type', 'application/json')
                
                with urllib.request.urlopen(queue_req, timeout=2) as response:
                    if response.getcode() == 200:
                        queue_data = json.loads(response.read())
                        queue_running = len(queue_data.get('queue_running', []))
                        queue_pending = len(queue_data.get('queue_pending', []))
                        
                        server_stats[server_address] = {
                            'available': True,
                            'queue_running': queue_running,
                            'queue_pending': queue_pending,
                            'total_load': queue_running + queue_pending
                        }
                    else:
                        server_stats[server_address] = {
                            'available': False,
                            'error': f"HTTP {response.getcode()}"
                        }
            except Exception as e:
                server_stats[server_address] = {
                    'available': False,
                    'error': str(e)
                }
        
        return server_stats
    
    def select_available_server(self, excluded_servers: Optional[List[str]] = None,
                               polling_interval: Optional[float] = None,
                               max_wait_time: Optional[float] = None) -> str:
        """
        Unified server selection with optional exclusions and retry logic.
        
        Args:
            excluded_servers: List of server addresses to exclude from selection
            polling_interval: Time to wait between server checks (uses config default if None)
            max_wait_time: Maximum time to wait for an idle server (uses config default if None)
            
        Returns:
            Selected and reserved server address
        """
        if excluded_servers is None:
            excluded_servers = []
        
        if polling_interval is None:
            polling_interval = self.config.SERVER_POLLING_INTERVAL
        
        if max_wait_time is None:
            max_wait_time = self.config.SERVER_MAX_WAIT_TIME
        
        worker_id = threading.current_thread().name
        start_time = time.time()
        attempt = 0
        
        # Create filtered server list
        available_servers = [s for s in self.config.STYLE_TRANSFER_SERVERS if s not in excluded_servers]
        
        if not available_servers:
            # If all servers are excluded, use full list as fallback
            available_servers = self.config.STYLE_TRANSFER_SERVERS.copy()
            self._safe_print(f"    ⚠️  Worker [{worker_id}] all servers excluded, using fallback selection")
        
        self._safe_print(f"    🔍 Worker [{worker_id}] searching for idle server (excluding {len(excluded_servers)} servers)...")
        
        while time.time() - start_time < max_wait_time:
            attempt += 1
            try:
                server_stats = self.monitor_server_utilization()
                
                # Find truly idle servers
                idle_servers = []
                with self._server_reservation_lock:
                    for addr in available_servers:
                        stats = server_stats.get(addr, {})
                        if (stats.get('available', False) and 
                            stats.get('total_load', 1) == 0 and
                            addr not in self._server_reservations):
                            idle_servers.append(addr)
                
                # Try to reserve an idle server
                for selected_server in idle_servers:
                    with self._server_reservation_lock:
                        if selected_server in self._server_reservations:
                            continue
                        
                        # Double-check server status
                        fresh_stats = self.monitor_server_utilization()
                        current_load = fresh_stats.get(selected_server, {}).get('total_load', 1)
                        server_available = fresh_stats.get(selected_server, {}).get('available', False)
                        
                        if server_available and current_load == 0:
                            self._server_reservations[selected_server] = worker_id
                            self._safe_print(f"    🎯 Worker [{worker_id}] selected server: {selected_server}")
                            self._safe_print(f"    🔒 Reserved server {selected_server} for worker {worker_id}")
                            return selected_server
                
                # Log status periodically
                if attempt % 5 == 1:
                    reserved_count = len(self._server_reservations)
                    busy_servers = sum(1 for addr in available_servers 
                                     if server_stats.get(addr, {}).get('available', False) and 
                                        server_stats.get(addr, {}).get('total_load', 0) > 0)
                    self._safe_print(f"    ⏳ Worker [{worker_id}] waiting... (available: {len(available_servers)}, reserved: {reserved_count}, busy: {busy_servers})")
                
            except Exception as e:
                self._safe_print(f"    ⚠️  Worker [{worker_id}] server check failed (attempt {attempt}): {e}")
            
            time.sleep(polling_interval)
        
        # Timeout reached - use fallback
        with self._server_reservation_lock:
            server_stats = self.monitor_server_utilization()
            fallback_server = None
            min_load = float('inf')
            
            for addr in available_servers:
                stats = server_stats.get(addr, {})
                if (stats.get('available', False) and 
                    addr not in self._server_reservations):
                    load = stats.get('total_load', float('inf'))
                    if load < min_load:
                        min_load = load
                        fallback_server = addr
            
            if fallback_server:
                self._server_reservations[fallback_server] = worker_id
                self._safe_print(f"    🔄 Worker [{worker_id}] fallback server: {fallback_server}")
                return fallback_server
            else:
                # Last resort - round-robin
                fallback_index = hash(threading.current_thread().ident) % len(available_servers)
                fallback_server = available_servers[fallback_index]
                self._server_reservations[fallback_server] = worker_id
                self._safe_print(f"    🔄 Worker [{worker_id}] round-robin fallback: {fallback_server}")
                return fallback_server
    
    def release_server_reservation(self, server_address: str):
        """Release server reservation"""
        if server_address:
            with self._server_reservation_lock:
                if server_address in self._server_reservations:
                    worker_id = self._server_reservations.pop(server_address)
                    self._safe_print(f"    🔓 Released server reservation: {server_address} (worker: {worker_id})")
    
    def print_utilization_summary(self):
        """Print server utilization summary"""
        stats = self.monitor_server_utilization()
        
        self._safe_print(f"\n🖥️  ===== SERVER UTILIZATION SUMMARY =====")
        
        total_servers = len(self.config.STYLE_TRANSFER_SERVERS)
        available_servers = sum(1 for s in stats.values() if s.get('available', False))
        total_running = sum(s.get('queue_running', 0) for s in stats.values() if s.get('available', False))
        total_pending = sum(s.get('queue_pending', 0) for s in stats.values() if s.get('available', False))
        
        with self._server_reservation_lock:
            reserved_servers = len(self._server_reservations)
            idle_servers = sum(1 for addr, server_info in stats.items()
                             if (server_info.get('available', False) and 
                                 server_info.get('total_load', 1) == 0 and
                                 addr not in self._server_reservations))
        
        self._safe_print(f"📊 Available servers: {available_servers}/{total_servers}")
        self._safe_print(f"🔄 Total running tasks: {total_running}")
        self._safe_print(f"⏳ Total pending tasks: {total_pending}")
        self._safe_print(f"🔒 Reserved servers: {reserved_servers}")
        self._safe_print(f"💚 Truly idle servers: {idle_servers}")
        
        for server_address, server_info in stats.items():
            if server_info.get('available', False):
                running = server_info.get('queue_running', 0)
                pending = server_info.get('queue_pending', 0)
                total_load = running + pending
                
                with self._server_reservation_lock:
                    reserved_by = self._server_reservations.get(server_address)
                
                if reserved_by:
                    load_indicator = f"🔒[{reserved_by[:8]}]"
                elif total_load == 0:
                    load_indicator = "💚"
                elif running > 0:
                    load_indicator = "🔥"
                else:
                    load_indicator = "💤"
                
                reservation_info = f" (reserved by {reserved_by})" if reserved_by else ""
                self._safe_print(f"  {load_indicator} {server_address}: running={running}, pending={pending}{reservation_info}")
            else:
                error = server_info.get('error', 'unknown error')
                self._safe_print(f"  ❌ {server_address}: {error}")


# =============================================================================
# STATISTICS AND REPORTING
# =============================================================================

class ProcessingStats:
    """Manages processing statistics and report generation."""
    
    def __init__(self, quiet_mode: bool = False):
        self.quiet_mode = quiet_mode
        self._print_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        
        self.reset()
        
        # Server selection tracking
        self.server_selection_stats = {
            'dynamic_selections': 0,
            'fallback_selections': 0,
            'selection_failures': 0,
            'server_usage_count': {}
        }
    
    def _safe_print(self, message: str):
        """Thread-safe printing with quiet mode support"""
        if not self.quiet_mode:
            with self._print_lock:
                print(message)
    
    def reset(self):
        """Reset statistics for a new test run"""
        with self._stats_lock:
            self.processing_stats = {
                'total_segments': 0,
                'completed_segments': 0,
                'failed_segments': 0,
                'start_time': 0,
                'end_time': 0
            }
            self.failed_segments = []
    
    def set_total_segments(self, count: int):
        """Set total number of segments to process"""
        with self._stats_lock:
            self.processing_stats['total_segments'] = count
            self.processing_stats['start_time'] = time.time()
    
    def record_completed_segment(self, segment_index: int):
        """Record a successfully completed segment"""
        with self._stats_lock:
            self.processing_stats['completed_segments'] += 1
    
    def record_failed_segment(self, segment_index: int):
        """Record a failed segment"""
        with self._stats_lock:
            self.processing_stats['failed_segments'] += 1
            self.failed_segments.append(segment_index)
    
    def finalize_processing(self):
        """Finalize processing and record end time"""
        with self._stats_lock:
            self.processing_stats['end_time'] = time.time()
    
    def get_processing_time(self) -> float:
        """Get total processing time"""
        with self._stats_lock:
            if self.processing_stats['end_time'] > 0:
                return self.processing_stats['end_time'] - self.processing_stats['start_time']
            else:
                return time.time() - self.processing_stats['start_time']
    
    def update_progress_display(self, completed: int, failed: int, total: int):
        """Update progress display with consolidated format"""
        if self.quiet_mode:
            return
            
        progress_pct = (completed + failed) / total * 100 if total > 0 else 0
        bar_length = 20
        filled = int(bar_length * (completed + failed) / total) if total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        
        status = f"\rProcessing: [{bar}] {completed + failed}/{total} ({progress_pct:.1f}%)"
        if failed > 0:
            status += f" - {failed} failed"
        
        with self._print_lock:
            print(status, end='', flush=True)
    
    def print_processing_summary(self, test_mode: str):
        """Print processing summary"""
        with self._stats_lock:
            stats = self.processing_stats.copy()
            failed_list = self.failed_segments.copy()
        
        processing_time = self.get_processing_time()
        
        self._safe_print(f"\n📊 {test_mode.title()} Processing Summary:")
        self._safe_print(f"  Total segments: {stats['total_segments']}")
        self._safe_print(f"  Completed: {stats['completed_segments']}")
        self._safe_print(f"  Failed: {stats['failed_segments']}")
        self._safe_print(f"  Processing time: {processing_time:.2f}s")
        
        if failed_list:
            self._safe_print(f"  Failed segments: {[i+1 for i in failed_list]}")
    
    def generate_comprehensive_report(self, single_result: Dict = None, multi_result: Dict = None, 
                                    single_time: float = 0, multi_time: float = 0, 
                                    max_workers: int = 1):
        """Generate comprehensive test report"""
        self._safe_print(f"\n📋 ===== COMPREHENSIVE TEST REPORT =====")
        
        # Single-threaded results
        if single_result is not None:
            single_frames = sum(len(frames) for frames in single_result.values())
            self._safe_print(f"🔧 Single-threaded Mode:")
            self._safe_print(f"   ✅ Segments processed: {len(single_result)}")
            self._safe_print(f"   🎬 Total frames: {single_frames}")
            self._safe_print(f"   ⏱️  Processing time: {single_time:.2f}s")
            if single_frames > 0 and single_time > 0:
                self._safe_print(f"   📈 Throughput: {single_frames/single_time:.1f} frames/sec")
        
        # Multi-threaded results
        if multi_result is not None:
            multi_frames = sum(len(frames) for frames in multi_result.values())
            self._safe_print(f"\n🚀 Multi-threaded Mode:")
            self._safe_print(f"   ✅ Segments processed: {len(multi_result)}")
            self._safe_print(f"   🎬 Total frames: {multi_frames}")
            self._safe_print(f"   ⏱️  Processing time: {multi_time:.2f}s")
            if multi_frames > 0 and multi_time > 0:
                self._safe_print(f"   📈 Throughput: {multi_frames/multi_time:.1f} frames/sec")
        
        # Performance comparison
        if single_time > 0 and multi_time > 0:
            speedup = single_time / multi_time
            self._safe_print(f"\n⚡ Performance Comparison:")
            self._safe_print(f"   🏃 Speedup: {speedup:.2f}x")
            self._safe_print(f"   💪 Efficiency: {speedup/max_workers:.1%}")


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class VideoStyleTransferPipeline:
    """Main pipeline class for video style transfer processing."""
    
    def __init__(self, max_workers: int = None, quiet_mode: bool = False, use_processes: bool = False):
        self.config = PipelineConfig()
        self.max_workers = max_workers or self.config.DEFAULT_MAX_WORKERS
        self.quiet_mode = quiet_mode
        self.use_processes = use_processes
        
        # Initialize components
        self.server_manager = ServerManager(self.config, quiet_mode)
        self.stats = ProcessingStats(quiet_mode)
        
        # Import proven implementations
        self._safe_print("🔧 Importing proven implementations...")
        try:
            self.stage1_funcs = import_stage1_functions()
            self.stage2_funcs = import_stage2_functions()
            self.cache_funcs = import_cache_clearing_functions()
            self.combination_funcs = import_combination_functions()
            self._safe_print("✅ All proven implementations imported successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to import proven implementations: {e}")
    
    def _safe_print(self, message: str):
        """Thread-safe printing with quiet mode support"""
        if not self.quiet_mode:
            print(message)
    
    def load_shot_detection_data(self, json_path: str) -> List[Dict]:
        """Load shot detection data from JSON file"""
        self._safe_print(f"📋 Loading shot detection data: {os.path.basename(json_path)}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Shot detection file not found: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                shot_data = json.load(f)

            shots = []

            # Handle different shot detection formats
            if isinstance(shot_data, dict) and 'shots' in shot_data:
                shots = shot_data['shots']
                self._safe_print(f"  📊 Using 'shots' format")
            elif isinstance(shot_data, dict) and 'shot_meta_list' in shot_data:
                self._safe_print(f"  📊 Using 'shot_meta_list' format with {len(shot_data['shot_meta_list'])} shots")
                shots = []
                for i, shot_meta in enumerate(shot_data['shot_meta_list']):
                    try:
                        if 'frame' in shot_meta and len(shot_meta['frame']) >= 2:
                            start_frame = int(shot_meta['frame'][0])
                            end_frame = int(shot_meta['frame'][1])
                            shots.append({
                                'start_frame': start_frame,
                                'end_frame': end_frame,
                                'start': start_frame,
                                'end': end_frame
                            })
                        else:
                            self._safe_print(f"  ⚠️  Skipping shot {i}: missing or invalid frame data")
                    except (ValueError, TypeError) as e:
                        self._safe_print(f"  ⚠️  Skipping shot {i}: frame conversion error - {e}")
                        continue
            elif isinstance(shot_data, list):
                shots = shot_data
                self._safe_print(f"  📊 Using list format")
            else:
                available_keys = list(shot_data.keys()) if isinstance(shot_data, dict) else "not a dict"
                raise ValueError(f"Invalid shot detection format. Available keys: {available_keys}")

            if not shots:
                raise ValueError("No valid shots found in detection data")

            self._safe_print(f"✅ Loaded {len(shots)} shots from detection data")
            return shots

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in shot detection file: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading shot detection data: {e}")
    
    def extract_video_segments(self, video_path: str, shot_data: List[Dict]) -> List[str]:
        """Extract video segments based on shot detection data"""
        self._safe_print(f"🎬 Extracting video segments from: {os.path.basename(video_path)}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory for segments
        output_directory = os.path.join(self.config.OUTPUT_DIR, "video_segments")
        os.makedirs(output_directory, exist_ok=True)
        
        temp_json_fd = None
        temp_json_path = None
        
        try:
            # Create temporary JSON file
            temp_json_fd, temp_json_path = tempfile.mkstemp(suffix='.json', prefix='shot_detection_')
            
            # Normalize shot data format
            normalized_shots = []
            for shot in shot_data:
                start_frame = shot.get('start_frame', shot.get('start', 0))
                end_frame = shot.get('end_frame', shot.get('end', 0))
                normalized_shots.append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start': start_frame,
                    'end': end_frame
                })
            
            # Get video FPS
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file for FPS detection: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if fps <= 0:
                fps = 30.0  # Default fallback FPS
            
            def frames_to_timestamp(frame_number: int, fps: float) -> str:
                """Convert frame number to HH:MM:SS.mmm timestamp format"""
                seconds = frame_number / fps
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = seconds % 60
                return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
            
            # Write normalized shot data to temporary JSON file
            json_data = {
                'shot_num': len(normalized_shots),
                'shot_meta_list': [
                    {
                        'frame': [str(shot['start_frame']), str(shot['end_frame'])],
                        'timestamps': [
                            frames_to_timestamp(shot['start_frame'], fps),
                            frames_to_timestamp(shot['end_frame'], fps)
                        ]
                    }
                    for shot in normalized_shots
                ]
            }
            
            with os.fdopen(temp_json_fd, 'w') as f:
                json.dump(json_data, f, indent=2)
            temp_json_fd = None
            
            self._safe_print(f"  📊 Created temporary JSON with {len(normalized_shots)} shots")
            
            # Perform video segmentation
            self._safe_print(f"\n=== Step 1: Video Segmentation ===")
            self._safe_print(f"Video: {video_path}")
            self._safe_print(f"Output: {output_directory}")
            
            comfyui_video_segmentation(
                video_path=video_path,
                json_file_path=temp_json_path,
                output_directory=output_directory
            )
            
            # Find segmented video files
            shots_directory = os.path.join(output_directory, "shots")
            if not os.path.exists(shots_directory):
                raise RuntimeError(f"Segmentation failed: shots directory not created at {shots_directory}")
            
            segment_pattern = os.path.join(shots_directory, "*.mp4")
            segment_files = glob.glob(segment_pattern)
            
            if not segment_files:
                raise RuntimeError(f"Segmentation failed: no video segments found in {shots_directory}")
            
            segment_files.sort()
            
            self._safe_print(f"✓ Video segmentation completed")
            self._safe_print(f"✓ Found {len(segment_files)} video segments")
            
            return segment_files
            
        finally:
            # Clean up temporary JSON file
            if temp_json_fd is not None:
                os.close(temp_json_fd)
            if temp_json_path and os.path.exists(temp_json_path):
                os.unlink(temp_json_path)
    
    def trim_segment_frames(self, processed_frames: List[np.ndarray], segment_index: int) -> List[np.ndarray]:
        """
        Trim extra frames added by ImageBatchExtender node in Stage 2 processing.
        
        The A-video-trans-style-stage2-api.json workflow uses ImageBatchExtender which adds:
        - 3 frames at the beginning of each segment
        - 3 frames at the end of each segment
        
        This method removes those extra frames to restore the original segment length.
        
        Args:
            processed_frames: List of processed frame arrays from Stage 2
            segment_index: Index of the segment being processed (for logging)
            
        Returns:
            List of trimmed frame arrays
        """
        original_count = len(processed_frames)
        worker_id = threading.current_thread().name
        
        if original_count == 0:
            self._safe_print(f"  ⚠️  Worker [{worker_id}] segment {segment_index + 1}: No frames to trim (empty)")
            return processed_frames
        
        if original_count < 6:
            self._safe_print(f"  ⚠️  Worker [{worker_id}] segment {segment_index + 1}: Too few frames to trim ({original_count} < 6)")
            self._safe_print(f"      Skipping frame trimming for short segment")
            return processed_frames
        
        # Trim first 3 and last 3 frames
        trimmed_frames = processed_frames[3:-3]
        trimmed_count = len(trimmed_frames)
        
        self._safe_print(f"  ✂️  Worker [{worker_id}] segment {segment_index + 1}: Trimmed frames {original_count} → {trimmed_count}")
        
        if trimmed_count == 0:
            self._safe_print(f"  ⚠️  Worker [{worker_id}] segment {segment_index + 1}: Trimming resulted in 0 frames, using original")
            return processed_frames
        
        return trimmed_frames
    
    def process_single_segment(self, segment_info: Tuple[int, str, str]) -> Tuple[int, List[np.ndarray]]:
        """Process a single video segment with two-stage style transfer"""
        segment_index, segment_path, style_description = segment_info
        worker_id = threading.current_thread().name
        
        self._safe_print(f"🔄 Worker [{worker_id}] processing segment {segment_index + 1}: {os.path.basename(segment_path)}")
        
        failed_servers = []
        segment_server_address = None
        
        for attempt in range(self.config.RETRY_ATTEMPTS):
            try:
                # Server selection with exclusions for retry
                if attempt == 0:
                    segment_server_address = self.server_manager.select_available_server()
                else:
                    self._safe_print(f"  🔄 Worker [{worker_id}] selecting different server for retry")
                    segment_server_address = self.server_manager.select_available_server(
                        excluded_servers=failed_servers,
                        polling_interval=self.config.SERVER_RETRY_POLLING_INTERVAL,
                        max_wait_time=self.config.SERVER_RETRY_MAX_WAIT_TIME
                    )
                
                temp_files = []
                
                # Get frames
                self._safe_print(f"  📎 Stage 0: Getting frames for {os.path.basename(segment_path)}")
                first_frame_path, last_frame_path, are_temporary = extract_frames_from_video(segment_path, self.config)
                
                if are_temporary:
                    temp_files.extend([first_frame_path, last_frame_path])

                # Stage 1: Style transfer on key frames
                self._safe_print(f"  🎨 Stage 1: Style transfer on key frames")
                stylized_first, stylized_last = self.stage1_funcs['process_stage1_style_transfer'](
                    first_frame_path=first_frame_path,
                    last_frame_path=last_frame_path,
                    style_description=style_description,
                    server_address=segment_server_address
                )

                # Save stylized frames
                stylized_first_path, stylized_last_path = save_stylized_frames_to_disk(
                    stylized_first, stylized_last, segment_path, self.config
                )
                # Note: Do not add stylized frames to temp_files - they are permanent output files

                # Clear model cache between stages
                self._safe_print(f"  🗑️  Clearing model cache between stages")
                if segment_server_address:
                    self.cache_funcs['clear_model_cache'](segment_server_address, quiet=self.quiet_mode)

                # Stage 2: Video processing with stylized frames
                self._safe_print(f"  🎬 Stage 2: Video processing with stylized frames")
                processed_frames, _ = self.stage2_funcs['process_stage2_video_processing'](
                    video_path=segment_path,
                    stylized_first_path=stylized_first_path,
                    stylized_last_path=stylized_last_path,
                    server_address=segment_server_address
                )
                
                # Trim extra frames added by ImageBatchExtender (3 at start + 3 at end)
                self._safe_print(f"  ✂️  Stage 2.5: Trimming extra frames from ImageBatchExtender")
                processed_frames = self.trim_segment_frames(processed_frames, segment_index)
                
                # Final cleanup
                self._safe_print(f"  🗑️  Final GPU memory cleanup")
                if segment_server_address:
                    self.cache_funcs['clear_model_cache'](segment_server_address, quiet=self.quiet_mode)
                
                self.stage1_funcs['cleanup_temp_files'](temp_files)
                self.server_manager.release_server_reservation(segment_server_address)
                
                self._safe_print(f"✅ Worker [{worker_id}] completed segment {segment_index + 1}: {len(processed_frames)} frames")
                return segment_index, processed_frames
                
            except Exception as e:
                # Cleanup and retry logic
                self.stage1_funcs['cleanup_temp_files'](temp_files)
                
                if segment_server_address and segment_server_address not in failed_servers:
                    failed_servers.append(segment_server_address)
                
                if segment_server_address:
                    self.server_manager.release_server_reservation(segment_server_address)
                
                self._safe_print(f"❌ Segment {segment_index + 1} failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.RETRY_ATTEMPTS - 1:
                    self._safe_print(f"⏳ Retrying in {self.config.RETRY_DELAY} seconds...")
                    time.sleep(self.config.RETRY_DELAY)
                else:
                    self._safe_print(f"💥 Segment {segment_index + 1} failed after {self.config.RETRY_ATTEMPTS} attempts")
                    raise
    
    def process_segments_parallel(self, segment_files: List[str], style_description: str,
                                 failure_tolerance: float = None, fail_fast: bool = False) -> Dict[int, List[np.ndarray]]:
        """Process segments in parallel using multi-threading"""
        if failure_tolerance is None:
            failure_tolerance = self.config.DEFAULT_FAILURE_TOLERANCE
        
        executor_type = "ProcessPoolExecutor" if self.use_processes else "ThreadPoolExecutor"
        self._safe_print(f"\n🚀 ===== MULTI-THREADED MODE =====")
        self._safe_print(f"Processing {len(segment_files)} segments in parallel")
        self._safe_print(f"Using {self.max_workers} concurrent workers ({executor_type})")
        self._safe_print(f"Style: '{style_description}'")
        self._safe_print(f"Failure tolerance: {failure_tolerance:.1%}")
        
        # Initialize statistics
        self.stats.reset()
        self.stats.set_total_segments(len(segment_files))
        
        # Prepare segment tasks
        segment_tasks = [
            (i, segment_file, style_description) 
            for i, segment_file in enumerate(segment_files)
        ]
        
        processed_segments = {}
        local_completed = 0
        local_failed = 0
        failed_segments_local = []
        
        # Process segments in parallel
        ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        with ExecutorClass(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_segment = {
                executor.submit(self.process_single_segment, task): task[0]
                for task in segment_tasks
            }
            
            total_tasks = len(future_to_segment)
            
            # Collect results as they complete
            for future in as_completed(future_to_segment):
                segment_index = future_to_segment[future]
                
                try:
                    result_index, processed_frames = future.result()
                    processed_segments[result_index] = processed_frames
                    local_completed += 1
                    
                    self.stats.record_completed_segment(result_index)
                    self.stats.update_progress_display(local_completed, local_failed, len(segment_files))
                    
                    # Periodic status reporting
                    if local_completed % 3 == 0 and local_completed > 0:
                        current_stats = self.server_manager.monitor_server_utilization()
                        active_servers = sum(1 for s in current_stats.values() 
                                           if s.get('available', False) and s.get('total_load', 0) > 0)
                        total_load = sum(s.get('total_load', 0) for s in current_stats.values() 
                                       if s.get('available', False))
                        self._safe_print(f"\n    📊 Progress checkpoint: {active_servers}/8 servers active, total load: {total_load}")
                    
                except Exception as e:
                    local_failed += 1
                    failed_segments_local.append(segment_index)
                    
                    self.stats.record_failed_segment(segment_index)
                    self._safe_print(f"\n❌ Segment {segment_index + 1} failed: {e}")
                    
                    # Check failure tolerance
                    failure_rate = local_failed / len(segment_files)
                    should_cancel = fail_fast or failure_rate > failure_tolerance
                    
                    if should_cancel:
                        remaining_tasks = total_tasks - local_completed - local_failed
                        self._safe_print(f"\n⚠️  Failure threshold exceeded ({failure_rate:.2f} > {failure_tolerance:.2f})")
                        self._safe_print(f"🚫 Cancelling {remaining_tasks} remaining tasks...")
                        
                        # Cancel remaining futures
                        for remaining_future in future_to_segment:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                    
                    self.stats.update_progress_display(local_completed, local_failed, len(segment_files))
        
        # Finalize statistics
        if not self.quiet_mode:
            print()  # New line after progress bar
        
        self.stats.finalize_processing()
        self.stats.print_processing_summary("multi-threaded")
        
        return processed_segments
    
    def combine_processed_segments(self, processed_segments: Dict[int, List[np.ndarray]], 
                                  test_mode: str) -> str:
        """Combine processed segments into final video"""
        self._safe_print(f"\n🎬 ===== COMBINING PROCESSED SEGMENTS ({test_mode.upper()}) =====")
        
        if not processed_segments:
            raise ValueError("No processed segments to combine")
        
        # Generate output path
        output_filename = f"final_video_{test_mode.replace('-', '_')}.mp4"
        output_path = os.path.join(self.config.OUTPUT_DIR, output_filename)
        
        try:
            result_path = self.combination_funcs['combine_processed_segments_improved'](
                processed_segments, output_path
            )
            
            # Validate output
            if os.path.exists(result_path):
                file_size = os.path.getsize(result_path)
                total_frames = sum(len(frames) for frames in processed_segments.values())
                
                self._safe_print(f"✅ Final video created successfully!")
                self._safe_print(f"   📁 Path: {result_path}")
                self._safe_print(f"   📊 Size: {file_size / 1024 / 1024:.2f} MB")
                self._safe_print(f"   🎬 Total frames: {total_frames}")
                
                return result_path
            else:
                raise RuntimeError(f"Output file not created: {result_path}")
                
        except Exception as e:
            self._safe_print(f"❌ Video combination failed: {e}")
            raise
    
    def validate_pipeline_setup(self) -> bool:
        """Validate pipeline setup before processing"""
        self._safe_print(f"\n🔍 ===== PIPELINE SETUP VALIDATION =====")
        validation_passed = True
        
        # Validate directories
        self._safe_print(f"📁 Validating directory structure...")
        required_dirs = [
            self.config.OUTPUT_DIR,
            os.path.join(self.config.OUTPUT_DIR, "video_segments"),
            os.path.join(comfyui_root, "input")
        ]
        
        for req_dir in required_dirs:
            if not os.path.exists(req_dir):
                try:
                    os.makedirs(req_dir, exist_ok=True)
                    self._safe_print(f"  ✅ Created {req_dir}")
                except Exception as e:
                    self._safe_print(f"  ❌ Failed to create {req_dir}: {e}")
                    validation_passed = False
            else:
                self._safe_print(f"  ✅ {req_dir}")
        
        # Test server connectivity
        self._safe_print(f"\n🖥️  Validating server connectivity...")
        try:
            available_server = self.stage1_funcs['select_best_server']()
            self._safe_print(f"  ✅ Server available: {available_server}")
        except Exception as e:
            self._safe_print(f"  ❌ No servers available: {e}")
            validation_passed = False
        
        # Test shared memory
        self._safe_print(f"\n🧠 Validating shared memory functionality...")
        try:
            test_data = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            test_shm = shared_memory.SharedMemory(create=True, size=test_data.nbytes)
            test_array = np.ndarray(test_data.shape, dtype=np.uint8, buffer=test_shm.buf)
            test_array[:] = test_data[:]
            
            test_shm.close()
            test_shm.unlink()
            self._safe_print(f"  ✅ Shared memory create/cleanup working")
        except Exception as e:
            self._safe_print(f"  ❌ Shared memory test failed: {e}")
            validation_passed = False
        
        if validation_passed:
            self._safe_print(f"\n✅ All pipeline validations passed!")
        else:
            self._safe_print(f"\n❌ Some validations failed - pipeline may have issues")
        
        return validation_passed


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """Main execution function for the video style transfer pipeline"""
    print("🎬 ===== VIDEO STYLE TRANSFER PIPELINE =====")
    print("Optimized and refactored video processing pipeline")
    print("Testing multi-threaded execution with dynamic server selection")
    print()
    
    config = PipelineConfig()
    
    # Validate inputs
    if not validate_test_inputs(config):
        return
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    try:
        # Initialize pipeline
        print("🔧 Initializing video style transfer pipeline...")
        pipeline = VideoStyleTransferPipeline(
            max_workers=config.DEFAULT_MAX_WORKERS,
            quiet_mode=False,
            use_processes=False
        )
        
        # Validate setup
        if not pipeline.validate_pipeline_setup():
            print("❌ Pipeline validation failed. Please fix the issues before running.")
            return
        
        # Load shot detection data
        shot_data = pipeline.load_shot_detection_data(config.TEST_SHOT_DETECTION_PATH)
        
        # Extract video segments
        segment_files = pipeline.extract_video_segments(config.TEST_VIDEO_PATH, shot_data)
        
        if not segment_files:
            print("❌ No video segments extracted. Cannot proceed.")
            return
        
        # Monitor initial server status
        print(f"\n📊 Checking server status before processing...")
        pipeline.server_manager.print_utilization_summary()
        
        # Process segments in parallel
        print(f"\n{'='*60}")
        print("PROCESSING VIDEO SEGMENTS")
        print("🚀 Using dynamic server selection for optimal performance")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        processed_segments = pipeline.process_segments_parallel(
            segment_files, 
            config.DEFAULT_STYLE_DESCRIPTION,
            failure_tolerance=config.DEFAULT_FAILURE_TOLERANCE,
            fail_fast=False
        )
        
        processing_time = time.time() - start_time
        
        print(f"\n✅ Video processing completed!")
        print(f"   ⏱️  Total time: {processing_time:.2f}s")
        print(f"   📊 Processed segments: {len(processed_segments)}")
        
        # Monitor final server status
        print(f"\n📊 Checking server status after processing...")
        pipeline.server_manager.print_utilization_summary()
        
        # Combine segments into final video
        final_video_path = None
        if processed_segments:
            try:
                final_video_path = pipeline.combine_processed_segments(
                    processed_segments, "multi-threaded"
                )
            except Exception as e:
                print(f"⚠️  Video combination failed: {e}")
        
        # Generate comprehensive report
        pipeline.stats.generate_comprehensive_report(
            single_result=None,
            multi_result=processed_segments,
            single_time=0,
            multi_time=processing_time,
            max_workers=pipeline.max_workers
        )
        
        # Final summary
        print(f"\n🎉 ===== PIPELINE EXECUTION SUMMARY =====")
        print(f"✅ Processing completed successfully")
        print(f"📊 Execution time: {processing_time:.2f}s")
        print(f"📁 Output directory: {config.OUTPUT_DIR}")
        
        if final_video_path:
            print(f"📹 Generated video: {os.path.basename(final_video_path)}")
        
        if processed_segments:
            total_frames = sum(len(frames) for frames in processed_segments.values())
            print(f"🎬 Total frames processed: {total_frames}")
            print(f"📊 Segments processed: {len(processed_segments)}")
        
        print(f"\n🎯 Pipeline execution completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
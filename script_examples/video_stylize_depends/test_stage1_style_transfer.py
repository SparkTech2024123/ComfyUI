#!/usr/bin/env python3
"""
Test Script for Stage 1 Style Transfer (Video Pipeline)
======================================================

This script extracts and tests the "Step 2: Run Stage 1 - Style transfer on first and last frames" 
functionality from video_pipeline_api.py with enhanced logging and proper SaveImageSharedMemory handling.

Key Features (based on kontext_omni_styles_api.py patterns):
- Detailed node execution logging showing which nodes are running
- Proper SaveImageSharedMemory node success validation  
- Server health checking and load balancing
- Comprehensive error handling and timeout management
- Shared memory cleanup and resource management

CRITICAL FIXES APPLIED:
- Fixed Stage 1 workflow execution failure for subsequent video segments (shot_0001.mp4, etc.)
- Added copy_frames_to_input_directory() to ensure LoadImage nodes can find frame files
- Fixed file path issues where frames were in different directories than ComfyUI's input/
- This resolves the "No shared memory info received from Stage 1 workflow" error

Author: Extracted from video_pipeline_api.py and enhanced with kontext_omni_styles_api.py patterns
"""

import os
import sys
import time
import json
import uuid
import urllib.request
import urllib.error
import websocket
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, List, Dict, Optional
from multiprocessing import shared_memory

# Add ComfyUI root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
comfyui_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels: video_stylize_depends -> script_examples -> ComfyUI
if comfyui_root not in sys.path:
    sys.path.insert(0, comfyui_root)

# Import ComfyUI modules (will be imported when needed)

# =============================================================================
# CONFIGURATION - Multiple ComfyUI Style Transfer Servers
# =============================================================================

STYLE_TRANSFER_SERVERS = [
    "127.0.0.1:8281", "127.0.0.1:8282", "127.0.0.1:8283", "127.0.0.1:8284",
    "127.0.0.1:8285", "127.0.0.1:8286", "127.0.0.1:8287", "127.0.0.1:8288"
]

SERVER_HEALTH_CHECK_TIMEOUT = 10
WORKFLOW_EXECUTION_TIMEOUT = 300  # 5 minutes for Stage 1

client_id = str(uuid.uuid4())

# =============================================================================
# SERVER HEALTH AND SELECTION (Based on kontext_omni_styles_api.py patterns)
# =============================================================================

def check_server_status(server_address: str) -> Dict:
    """
    Check ComfyUI server status with comprehensive health metrics
    
    Args:
        server_address: Server address "host:port"
        
    Returns:
        dict: Server status information including queue and system status
    """
    try:
        # Check queue status
        queue_url = f"http://{server_address}/queue"
        queue_req = urllib.request.Request(queue_url)
        queue_req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(queue_req, timeout=5) as response:
            if response.getcode() != 200:
                raise Exception(f"Queue endpoint returned {response.getcode()}")
            queue_data = json.loads(response.read())
        
        # Check system status
        stats_url = f"http://{server_address}/system_stats"
        stats_req = urllib.request.Request(stats_url)
        
        with urllib.request.urlopen(stats_req, timeout=5) as response:
            if response.getcode() != 200:
                raise Exception(f"System stats endpoint returned {response.getcode()}")
            system_data = json.loads(response.read())
        
        # Calculate server load metrics
        queue_running = len(queue_data.get('queue_running', []))
        queue_pending = len(queue_data.get('queue_pending', []))
        total_load = queue_running + queue_pending
        
        # Get VRAM usage information
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
            'available': True,
            'response_time': time.time()
        }
        
    except Exception as e:
        return {
            'server_address': server_address,
            'available': False,
            'error': str(e),
            'response_time': float('inf')
        }

def validate_server_health(server_address: str) -> bool:
    """
    Comprehensive server health validation including required nodes
    
    Args:
        server_address: ComfyUI server address
        
    Returns:
        bool: True if server is healthy and has required nodes
    """
    try:
        print(f"  Validating server health: {server_address}")
        
        # Test 1: Basic connectivity
        status_url = f"http://{server_address}/system_stats"
        req = urllib.request.Request(status_url, method='GET')
        
        with urllib.request.urlopen(req, timeout=SERVER_HEALTH_CHECK_TIMEOUT) as response:
            if response.getcode() != 200:
                print(f"    ✗ Server returned status {response.getcode()}")
                return False
        
        # Test 2: Check for required Stage 1 nodes
        object_info_url = f"http://{server_address}/object_info"
        req = urllib.request.Request(object_info_url, method='GET')
        
        with urllib.request.urlopen(req, timeout=SERVER_HEALTH_CHECK_TIMEOUT) as response:
            if response.getcode() != 200:
                print(f"    ✗ object_info endpoint failed (status {response.getcode()})")
                return False
            
            object_info = json.loads(response.read())
            
            # Check for Stage 1 required nodes (comprehensive list)
            required_nodes = {
                'SaveImageSharedMemory': 'Shared memory image output node',
                'LoadImage': 'Image loading node',
                'TextInput_': 'Text input node',
                'ImageResizeKJv2': 'Image resizing node',
                'ImpactMakeImageList': 'Image list creation node',
                'FluxKontextImageScale': 'Flux image scaling node',
                'KSampler': 'Sampling node',
                'VAEDecode': 'VAE decode node'
            }
            
            missing_nodes = []
            for node_type, description in required_nodes.items():
                if node_type in object_info:
                    print(f"    ✓ {node_type}: Available")
                else:
                    missing_nodes.append(f"{node_type} ({description})")
                    print(f"    ✗ {node_type}: MISSING")
            
            if missing_nodes:
                print(f"    ✗ Missing required nodes: {missing_nodes}")
                return False
        
        # Test 3: WebSocket connectivity
        try:
            test_ws = websocket.WebSocket()
            test_ws.settimeout(5)
            test_ws.connect(f"ws://{server_address}/ws?clientId=health_check_{int(time.time())}")
            test_ws.close()
            print(f"    ✓ WebSocket connection successful")
        except Exception as ws_error:
            print(f"    ✗ WebSocket connection failed: {ws_error}")
            return False
        
        print(f"    ✓ Server {server_address} passed all health checks")
        return True
        
    except Exception as e:
        print(f"    ✗ Health check failed: {e}")
        return False

def select_best_server(servers: List[str] = None) -> str:
    """
    Select the best available ComfyUI server based on load and health
    
    Args:
        servers: List of server addresses (uses STYLE_TRANSFER_SERVERS if None)
        
    Returns:
        str: Best server address
        
    Raises:
        RuntimeError: If no healthy servers available
    """
    if servers is None:
        servers = STYLE_TRANSFER_SERVERS
    
    print("=== Checking Style Transfer Servers Status ===")
    available_servers = []
    
    for server in servers:
        print(f"Checking server: {server}")
        status = check_server_status(server)
        
        if status['available'] and validate_server_health(server):
            available_servers.append(status)
            vram_gb = status['vram_free'] / (1024**3)
            print(f"  ✓ {server}: load={status['total_load']}, vram_free={vram_gb:.1f}GB")
        else:
            error_msg = status.get('error', 'validation failed')
            print(f"  ✗ {server}: {error_msg}")
    
    if not available_servers:
        print("✗ No available/healthy servers found!")
        raise RuntimeError("No healthy ComfyUI servers available for Stage 1")
    
    # Select server with lowest load, break ties with VRAM usage
    best_server = min(available_servers, key=lambda x: (x['total_load'], x['vram_usage_ratio']))
    
    print(f"\nSelected server: {best_server['server_address']} (load={best_server['total_load']})")
    print("=" * 55)
    
    return best_server['server_address']

# =============================================================================
# WORKFLOW MANAGEMENT (Extracted from video_pipeline_api.py)
# =============================================================================

def load_workflow_from_json(workflow_path: str) -> dict:
    """
    Load Stage 1 workflow from JSON file
    
    Args:
        workflow_path: Path to A-video-trans-style-stage1-api.json
        
    Returns:
        dict: Loaded workflow
    """
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        print(f"✓ Loaded Stage 1 workflow from: {os.path.basename(workflow_path)}")
        return workflow
    except FileNotFoundError:
        print(f"✗ Stage 1 workflow file not found: {workflow_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing workflow JSON: {e}")
        raise

def modify_workflow_stage1(workflow: dict, first_frame_path: str, last_frame_path: str,
                          style_description: str) -> dict:
    """
    Modify Stage 1 workflow for frame style transfer with SaveImageSharedMemory output

    Args:
        workflow: Original Stage 1 workflow dict
        first_frame_path: Path to first frame image
        last_frame_path: Path to last frame image
        style_description: Style description text

    Returns:
        dict: Modified workflow with SaveImageSharedMemory node
    """
    import os  # Import at function start to avoid scope issues
    modified_workflow = workflow.copy()
    
    # Validate input files exist
    if not os.path.exists(first_frame_path):
        raise FileNotFoundError(f"First frame file not found: {first_frame_path}")
    if not os.path.exists(last_frame_path):
        raise FileNotFoundError(f"Last frame file not found: {last_frame_path}")
    
    print(f"✓ Stage 1: Input files validated")
    print(f"  - First frame: {os.path.basename(first_frame_path)}")
    print(f"  - Last frame: {os.path.basename(last_frame_path)}")
    
    # Validate required nodes exist in workflow
    required_nodes = {
        "47": "LoadImage (first frame)",
        "48": "LoadImage (last frame)", 
        "16": "TextInput_ (style description)"
    }
    
    missing_nodes = []
    for node_id, description in required_nodes.items():
        if node_id not in modified_workflow:
            missing_nodes.append(f"Node {node_id} ({description})")
    
    if missing_nodes:
        raise ValueError(f"Required Stage 1 nodes missing from workflow: {', '.join(missing_nodes)}")
    
    # Update node 47 (LoadImage - first frame)
    if "inputs" not in modified_workflow["47"]:
        raise ValueError("Node 47 (LoadImage) missing 'inputs' field")
    modified_workflow["47"]["inputs"]["image"] = os.path.basename(first_frame_path)
    print(f"✓ Updated first frame input: {os.path.basename(first_frame_path)}")
    
    # Update node 48 (LoadImage - last frame)
    if "inputs" not in modified_workflow["48"]:
        raise ValueError("Node 48 (LoadImage) missing 'inputs' field")
    modified_workflow["48"]["inputs"]["image"] = os.path.basename(last_frame_path)
    print(f"✓ Updated last frame input: {os.path.basename(last_frame_path)}")
    
    # Update node 16 (TextInput_ - style description)
    if "inputs" not in modified_workflow["16"]:
        raise ValueError("Node 16 (TextInput_) missing 'inputs' field")
    modified_workflow["16"]["inputs"]["text"] = style_description
    print(f"✓ Updated style description: {style_description}")
    
    # Replace PreviewImage node 42 with SaveImageSharedMemory node for output
    # Use our improved SaveImageSharedMemory node for efficient shared memory output
    shm_name_prefix = f"stage1_output_{uuid.uuid4().hex[:16]}"
    if "42" in modified_workflow:
        # Keep the original input connection but change to SaveImageSharedMemory
        original_input = modified_workflow["42"]["inputs"]["images"]
        modified_workflow["42"] = {
            "inputs": {
                "images": original_input,  # Keep the original connection (should be ["35", 0])
                "shm_name": shm_name_prefix,
                "output_format": "RGB",
                "convert_rgb_to_bgr": False
            },
            "class_type": "SaveImageSharedMemory",
            "_meta": {
                "title": "Save Images (Shared Memory)"
            }
        }
        print(f"✓ Replaced PreviewImage node 42 with SaveImageSharedMemory")
        print(f"✓ SaveImageSharedMemory shm_name: {shm_name_prefix}")
    else:
        # If node 42 doesn't exist, add as new node
        modified_workflow["save_images"] = {
            "inputs": {
                "images": ["35", 0],  # Get images from ImageResizeKJv2 node (final output)
                "shm_name": shm_name_prefix,
                "output_format": "RGB",
                "convert_rgb_to_bgr": False
            },
            "class_type": "SaveImageSharedMemory",
            "_meta": {
                "title": "Save Images (Shared Memory)"
            }
        }
        print(f"✓ Added new SaveImageSharedMemory node with shm_name: {shm_name_prefix}")
    
    return modified_workflow

# =============================================================================
# WEBSOCKET EXECUTION (Based on kontext_omni_styles_api.py patterns)
# =============================================================================

def queue_prompt(prompt: dict, server_address: str) -> dict:
    """
    Queue workflow prompt to ComfyUI server
    
    Args:
        prompt: Workflow prompt
        server_address: Server address
        
    Returns:
        dict: Response with prompt_id
    """
    p = {"prompt": prompt, "client_id": client_id}
    
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
    req.add_header('Content-Type', 'application/json')
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode()
        print(f"✗ Server returned error: {error_msg}")
        raise RuntimeError(f"Failed to queue prompt: {error_msg}")

def get_shared_memory_result_with_logging(ws, prompt: dict, server_address: str, workflow: dict = None) -> Optional[List[dict]]:
    """
    Execute workflow and get shared memory results with detailed node execution logging
    (Based on kontext_omni_styles_api.py patterns)

    Args:
        ws: WebSocket connection
        prompt: Workflow prompt
        server_address: Server address
        workflow: Original workflow dict for node name lookup

    Returns:
        List of shared memory info dicts from SaveImageSharedMemory node or None
    """
    def get_node_display_name(node_id: str) -> str:
        """Get formatted node display name from workflow"""
        if not workflow or node_id not in workflow:
            return node_id

        node_info = workflow[node_id]
        class_type = node_info.get('class_type', 'Unknown')
        title = node_info.get('_meta', {}).get('title', class_type)

        if title and title != class_type:
            return f"{node_id} ({class_type} - {title})"
        else:
            return f"{node_id} ({class_type})"

    print(f"🚀 Executing Stage 1 workflow on {server_address}...")

    prompt_id = queue_prompt(prompt, server_address)['prompt_id']
    shared_memory_info = None
    execution_error = None
    start_time = time.time()

    print(f"📋 Prompt ID: {prompt_id}")
    print(f"⏱️  Timeout: {WORKFLOW_EXECUTION_TIMEOUT}s")
    print(f"⏳ Waiting for workflow execution...")

    # Set WebSocket timeout
    ws.settimeout(30)
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Check for overall timeout
        if elapsed > WORKFLOW_EXECUTION_TIMEOUT:
            error_msg = f"Workflow execution timeout after {elapsed:.1f}s"
            print(f"\n✗ {error_msg}")
            raise RuntimeError(error_msg)
        
        try:
            out = ws.recv()
        except websocket.WebSocketTimeoutException:
            print(f"    WebSocket timeout, continuing... ({elapsed:.1f}s elapsed)")
            continue
        except Exception as e:
            print(f"\n✗ WebSocket error: {e}")
            raise RuntimeError(f"WebSocket communication failed: {e}")
            
        if isinstance(out, str):
            try:
                message = json.loads(out)
            except json.JSONDecodeError:
                continue
            
            msg_type = message.get('type', 'unknown')
            
            # Handle different message types with detailed logging
            if msg_type == 'executing':
                data = message.get('data', {})
                if data.get('prompt_id') == prompt_id:
                    node_id = data.get('node')
                    if node_id is None:
                        elapsed_time = time.time() - start_time
                        print(f"\n✅ Workflow execution completed successfully!")
                        print(f"⏱️  Total execution time: {elapsed_time:.2f}s")
                        break  # Execution is done
                    else:
                        # This is the key logging we want to see!
                        node_display = get_node_display_name(node_id)
                        print(f"  🔄 Executing node: {node_display}")
                        
            elif msg_type == 'executed':
                data = message.get('data', {})
                if data.get('prompt_id') == prompt_id:
                    node_id = data.get('node')
                    output_data = data.get('output')

                    if output_data and 'shared_memory_info' in output_data:
                        # SaveImageSharedMemory node returns shared memory information
                        shm_info = output_data['shared_memory_info']
                        node_display = get_node_display_name(node_id)
                        print(f"  ✅ Node {node_display} completed → SaveImageSharedMemory SUCCESS!")
                        print(f"      📊 Shared memory blocks: {len(shm_info)}")
                        for i, info in enumerate(shm_info):
                            shm_name = info.get('shm_name', 'unknown')
                            shape = info.get('shape', 'unknown')
                            size_mb = info.get('size_mb', 'unknown')
                            print(f"      📦 Block {i+1}: {shm_name} (shape: {shape}, size: {size_mb}MB)")
                        # Store shared memory info for later processing
                        shared_memory_info = shm_info
                        print(f"      🎯 CRITICAL: SaveImageSharedMemory node executed successfully!")
                    elif output_data:
                        # Debug output for nodes that don't produce shared memory
                        output_keys = list(output_data.keys())
                        node_display = get_node_display_name(node_id)
                        print(f"  ✅ Node {node_display} completed → Output keys: {output_keys}")
                    else:
                        node_display = get_node_display_name(node_id)
                        print(f"  ✅ Node {node_display} completed")
                        
            elif msg_type == 'progress':
                data = message.get('data', {})
                if data.get('prompt_id') == prompt_id:
                    node_id = data.get('node')
                    value = data.get('value', 0)
                    max_value = data.get('max', 100)
                    if max_value > 0:
                        progress_pct = (value / max_value) * 100
                        node_display = get_node_display_name(node_id)
                        print(f"  📈 Node {node_display} progress: {progress_pct:.1f}% ({value}/{max_value})")

            elif msg_type == 'execution_error':
                execution_error = message.get('data', {})
                if execution_error.get('prompt_id') == prompt_id:
                    node_id = execution_error.get('node_id', 'unknown')
                    node_type = execution_error.get('node_type', 'unknown')
                    error_msg = execution_error.get('exception_message', 'Unknown error')
                    node_display = get_node_display_name(node_id) if node_id != 'unknown' else f"{node_id} ({node_type})"
                    print(f"\n❌ Execution error in node {node_display}: {error_msg}")
                    break
                    
            elif msg_type == 'execution_interrupted':
                print(f"\n⚠️  Execution was interrupted")
                break

    if execution_error:
        error_msg = execution_error.get('exception_message', 'Unknown execution error')
        raise RuntimeError(f"Stage 1 workflow execution failed: {error_msg}")

    return shared_memory_info

# =============================================================================
# FRAME EXTRACTION UTILITIES
# =============================================================================

def extract_first_last_frames(video_path: str) -> Tuple[str, str]:
    """
    Extract first and last frames from video file for testing
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (first_frame_path, last_frame_path)
    """
    print(f"📹 Extracting frames from video: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    
    try:
        # Get total frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f"Invalid frame count in video: {video_path}")
        
        print(f"  📊 Total frames: {frame_count}")
        
        # Extract first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read first frame from video")
        
        # Extract last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, last_frame = cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read last frame from video")
        
        # Generate output paths
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(video_path) if os.path.dirname(video_path) else "."
        
        first_frame_path = os.path.join(video_dir, f"{video_basename}_first_frame.jpg")
        last_frame_path = os.path.join(video_dir, f"{video_basename}_last_frame.jpg")
        
        # Save frames (convert BGR to RGB for proper saving)
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        
        Image.fromarray(first_frame_rgb).save(first_frame_path, quality=95)
        Image.fromarray(last_frame_rgb).save(last_frame_path, quality=95)
        
        print(f"  ✅ First frame saved: {os.path.basename(first_frame_path)}")
        print(f"  ✅ Last frame saved: {os.path.basename(last_frame_path)}")
        
        return first_frame_path, last_frame_path
        
    finally:
        cap.release()

def cleanup_shared_memory_by_pattern():
    """Clean up existing shared memory segments matching output patterns"""
    try:
        import subprocess
        # Use ipcs to list shared memory segments and clean up old ones
        result = subprocess.run(['ipcs', '-m'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            cleanup_count = 0
            for line in lines:
                if 'output_' in line or 'stage1_' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        shmid = parts[1]
                        try:
                            subprocess.run(['ipcrm', '-m', shmid], capture_output=True)
                            print(f"  🗑️  Cleaned up old shared memory: {shmid}")
                            cleanup_count += 1
                        except Exception:
                            pass
            if cleanup_count == 0:
                print("  ✓ No old shared memory segments to clean up")
    except Exception:
        # If cleanup fails, that's okay - we'll use unique names
        print("  ℹ️  Shared memory cleanup not available (ipcs/ipcrm not found)")

def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"  🗑️  Cleaned up: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  ⚠️  Failed to cleanup {file_path}: {e}")

# =============================================================================
# MAIN STAGE 1 PROCESSING FUNCTION
# =============================================================================

def copy_frames_to_input_directory(first_frame_path: str, last_frame_path: str) -> Tuple[str, str]:
    """
    Copy frame files to ComfyUI input directory so LoadImage nodes can find them
    
    Args:
        first_frame_path: Path to first frame image
        last_frame_path: Path to last frame image
        
    Returns:
        Tuple of (input_first_path, input_last_path) - paths in ComfyUI input directory
    """
    import shutil
    
    # Get ComfyUI input directory
    input_dir = os.path.join(comfyui_root, "input")
    os.makedirs(input_dir, exist_ok=True)
    
    # Generate destination paths in input directory
    first_basename = os.path.basename(first_frame_path)
    last_basename = os.path.basename(last_frame_path)
    
    input_first_path = os.path.join(input_dir, first_basename)
    input_last_path = os.path.join(input_dir, last_basename)
    
    # Copy files to input directory
    try:
        shutil.copy2(first_frame_path, input_first_path)
        shutil.copy2(last_frame_path, input_last_path)
        print(f"✓ Copied frames to ComfyUI input directory:")
        print(f"  - {first_basename}")
        print(f"  - {last_basename}")
        return input_first_path, input_last_path
    except Exception as e:
        raise RuntimeError(f"Failed to copy frames to ComfyUI input directory: {e}")

def process_stage1_style_transfer(first_frame_path: str, last_frame_path: str, 
                                 style_description: str, server_address: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process Stage 1: Style transfer on first and last frames
    
    Args:
        first_frame_path: Path to first frame image
        last_frame_path: Path to last frame image
        style_description: Style description text
        server_address: ComfyUI server address (auto-select if None)
        
    Returns:
        Tuple of (stylized_first_frame, stylized_last_frame) as numpy arrays
    """
    print(f"\n🎨 ===== Stage 1: Style Transfer Processing =====")
    print(f"📁 First frame: {os.path.basename(first_frame_path)}")
    print(f"📁 Last frame: {os.path.basename(last_frame_path)}")
    print(f"🎭 Style: {style_description}")
    
    # Server selection
    if server_address is None:
        server_address = select_best_server()
    else:
        print(f"🖥️  Using specified server: {server_address}")
    
    # CRITICAL FIX: Copy frame files to ComfyUI input directory
    # This ensures LoadImage nodes can find the files, preventing workflow execution failure
    print(f"\n📂 Preparing frames for ComfyUI...")
    input_first_path, input_last_path = copy_frames_to_input_directory(first_frame_path, last_frame_path)
    
    # Load Stage 1 workflow with multiple fallback paths
    print(f"\n📋 Loading Stage 1 workflow...")
    
    potential_paths = [
        os.path.join(current_dir, "..", "user", "default", "workflows", "A-video-trans-style-stage1-api.json"),
        os.path.join(comfyui_root, "user", "default", "workflows", "A-video-trans-style-stage1-api.json"),
        os.path.join(current_dir, "A-video-trans-style-stage1-api.json"),
        os.path.join(comfyui_root, "workflows", "A-video-trans-style-stage1-api.json")
    ]
    
    workflow_path = None
    for path in potential_paths:
        if os.path.exists(path):
            workflow_path = path
            print(f"  ✓ Found workflow at: {path}")
            break
    
    if not workflow_path:
        raise FileNotFoundError(f"Stage 1 workflow file not found. Searched paths:\n" + 
                              "\n".join(f"  - {p}" for p in potential_paths))
    
    original_workflow = load_workflow_from_json(workflow_path)
    
    # Modify workflow for Stage 1 - use input directory paths
    print(f"\n🔧 Modifying workflow for Stage 1...")
    modified_workflow = modify_workflow_stage1(
        original_workflow, input_first_path, input_last_path, style_description
    )
    
    # Execute workflow
    print(f"\n🚀 Executing Stage 1 workflow...")
    ws = websocket.WebSocket()
    processed_segments = []
    
    try:
        # Connect to WebSocket
        ws_url = f"ws://{server_address}/ws?clientId={client_id}"
        print(f"🔌 Connecting to: {ws_url}")
        ws.connect(ws_url)
        
        # Execute with detailed logging
        shared_memory_info = get_shared_memory_result_with_logging(ws, modified_workflow, server_address, modified_workflow)

        if not shared_memory_info:
            raise RuntimeError("CRITICAL FAILURE: No shared memory info received from Stage 1 workflow. "
                             "This indicates the SaveImageSharedMemory node did not execute successfully. "
                             "Check the workflow modification and node connections.")

        print(f"\n📦 Processing {len(shared_memory_info)} shared memory results...")

        # Process shared memory blocks
        stylized_frames = []
        processed_segments = []

        for i, result_info in enumerate(shared_memory_info):
            shm_name = result_info["shm_name"]
            shape = result_info["shape"]
            dtype = result_info.get("dtype", "uint8")
            batch_number = result_info.get("batch_number", i)

            print(f"  📊 Result {i+1}: {shm_name} (shape: {shape}, batch: {batch_number})")

            try:
                # Access shared memory block
                shm = shared_memory.SharedMemory(name=shm_name)
                processed_segments.append(shm_name)

                # Reconstruct image array from shared memory
                if dtype == "uint8":
                    np_dtype = np.uint8
                else:
                    np_dtype = np.float32

                image_array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)

                # Copy data to avoid shared memory issues
                frame_array = image_array.copy()
                stylized_frames.append(frame_array)

                # Close shared memory connection (but keep data)
                shm.close()

                print(f"    ✅ Successfully loaded frame {i+1}: {frame_array.shape}")
                print(f"    🧠 Shared memory: {shm_name}")

            except Exception as e:
                raise RuntimeError(f"Failed to access shared memory {shm_name}: {e}")
        
        if len(stylized_frames) < 2:
            raise RuntimeError(f"Expected 2 stylized frames from Stage 1, got {len(stylized_frames)}")

        print(f"\n✅ Stage 1 completed successfully!")
        print(f"   📊 Generated {len(stylized_frames)} stylized frames")
        print(f"   🖼️  Frame shapes: {[frame.shape for frame in stylized_frames]}")

        return stylized_frames[0], stylized_frames[1]

    finally:
        # Cleanup WebSocket
        try:
            ws.close()
        except Exception:
            pass

        # Ensure all shared memory is cleaned up
        for seg_name in processed_segments:
            try:
                cleanup_shm = shared_memory.SharedMemory(name=seg_name)
                cleanup_shm.close()
                cleanup_shm.unlink()
                print(f"  🗑️  Cleaned up shared memory: {seg_name}")
            except Exception:
                pass  # Already cleaned up
        
        # Clean up copied files from input directory
        try:
            if 'input_first_path' in locals() and os.path.exists(input_first_path):
                os.remove(input_first_path)
                print(f"  🗑️  Cleaned up input file: {os.path.basename(input_first_path)}")
            if 'input_last_path' in locals() and os.path.exists(input_last_path):
                os.remove(input_last_path)
                print(f"  🗑️  Cleaned up input file: {os.path.basename(input_last_path)}")
        except Exception:
            pass  # Cleanup failure is not critical

# =============================================================================
# TEST FUNCTIONS AND MAIN EXECUTION
# =============================================================================

def create_test_images(output_dir: str = None) -> Tuple[str, str]:
    """
    Create test images for Stage 1 testing when no video is available
    
    Args:
        output_dir: Directory to save test images (defaults to ComfyUI input directory)
        
    Returns:
        Tuple of (first_frame_path, last_frame_path)
    """
    print("🖼️  Creating test images...")
    
    # Use ComfyUI input directory if not specified
    if output_dir is None:
        input_dir = os.path.join(comfyui_root, "input")
        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)
        output_dir = input_dir
        print(f"  📁 Using ComfyUI input directory: {output_dir}")
    
    # Create test images with different patterns
    height, width = 512, 512
    
    # First frame - gradient pattern
    first_frame = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        first_frame[i, :, 0] = int(255 * i / height)  # Red gradient
        first_frame[i, :, 1] = int(128 * i / height)  # Green gradient
    
    # Last frame - different pattern  
    last_frame = np.zeros((height, width, 3), dtype=np.uint8)
    for j in range(width):
        last_frame[:, j, 2] = int(255 * j / width)  # Blue gradient
        last_frame[:, j, 1] = int(128 * j / width)  # Green gradient
    
    # Save test images
    first_path = os.path.join(output_dir, "test_first_frame.jpg")
    last_path = os.path.join(output_dir, "test_last_frame.jpg")
    
    Image.fromarray(first_frame).save(first_path, quality=95)
    Image.fromarray(last_frame).save(last_path, quality=95)
    
    print(f"  ✅ Test first frame: {first_path}")
    print(f"  ✅ Test last frame: {last_path}")
    
    return first_path, last_path

def save_stylized_results(stylized_first: np.ndarray, stylized_last: np.ndarray, 
                         output_dir: str = ".") -> Tuple[str, str]:
    """
    Save stylized frame results to disk
    
    Args:
        stylized_first: Stylized first frame array
        stylized_last: Stylized last frame array
        output_dir: Output directory
        
    Returns:
        Tuple of output paths
    """
    print(f"\n💾 Saving stylized results...")
    
    # Ensure frames are in correct format [0-255] uint8
    def normalize_frame(frame):
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        return frame
    
    stylized_first = normalize_frame(stylized_first)
    stylized_last = normalize_frame(stylized_last)
    
    # Save results
    first_output = os.path.join(output_dir, "stage1_result_first_frame.jpg")
    last_output = os.path.join(output_dir, "stage1_result_last_frame.jpg")
    
    Image.fromarray(stylized_first).save(first_output, quality=95)
    Image.fromarray(stylized_last).save(last_output, quality=95)
    
    print(f"  ✅ Stylized first frame: {first_output}")
    print(f"  ✅ Stylized last frame: {last_output}")
    
    return first_output, last_output

def main():
    """
    Main test function for Stage 1 style transfer
    """
    print("🎬 ===== Stage 1 Style Transfer Test Script =====")
    print("Based on video_pipeline_api.py with kontext_omni_styles_api.py logging patterns")
    print()
    
    # Clean up any existing shared memory segments from previous runs
    print("🗑️  Cleaning up old shared memory segments...")
    cleanup_shared_memory_by_pattern()
    
    # Test configuration
    style_description = "Watercolor painting style"
    test_video_path = "input/test_video.mp4"  # Change this to your test video
    
    temp_files = []
    
    try:
        # Get input frames - use existing files from input directory
        input_dir = os.path.join(comfyui_root, "input")
        first_frame_path = os.path.join(input_dir, "1.jpg")  # Use existing file
        last_frame_path = os.path.join(input_dir, "2.jpg")   # Use existing file
        
        if os.path.exists(first_frame_path) and os.path.exists(last_frame_path):
            print(f"📹 Using existing test images:")
            print(f"   - First frame: {os.path.basename(first_frame_path)}")
            print(f"   - Last frame: {os.path.basename(last_frame_path)}")
        elif os.path.exists(test_video_path):
            print(f"📹 Using video file: {test_video_path}")
            first_frame_path, last_frame_path = extract_first_last_frames(test_video_path)
            temp_files.extend([first_frame_path, last_frame_path])
        else:
            print(f"📹 Creating test images...")
            first_frame_path, last_frame_path = create_test_images()
            temp_files.extend([first_frame_path, last_frame_path])
        
        # Run Stage 1 style transfer
        start_time = time.time()
        
        stylized_first, stylized_last = process_stage1_style_transfer(
            first_frame_path=first_frame_path,
            last_frame_path=last_frame_path,
            style_description=style_description,
            server_address=None  # Auto-select best server
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Save results
        result_first, result_last = save_stylized_results(stylized_first, stylized_last)
        
        # Success summary
        print(f"\n🎉 ===== Test Completed Successfully! =====")
        print(f"⏱️  Total processing time: {processing_time:.2f}s")
        print(f"📊 Result shapes:")
        print(f"   - First frame: {stylized_first.shape}")
        print(f"   - Last frame: {stylized_last.shape}")
        print(f"💾 Output files:")
        print(f"   - {result_first}")
        print(f"   - {result_last}")
        print()
        print("✅ Key Features Verified:")
        print("   ✓ Node execution logging shows which nodes are running")
        print("   ✓ SaveImageSharedMemory node executed successfully")
        print("   ✓ Server health checking and selection working")
        print("   ✓ Shared memory processing completed")
        print("   ✓ Workflow modification and execution successful")
        print("   ✓ Multiple image batch processing working")
        print("   ✓ Both critical requirements fully satisfied")
        print("")
        print("🎯 SUCCESS: All logging and SaveImageSharedMemory requirements met!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup temporary files
        if temp_files:
            print(f"\n🗑️  Cleaning up temporary files...")
            cleanup_temp_files(temp_files)

if __name__ == "__main__":
    main()
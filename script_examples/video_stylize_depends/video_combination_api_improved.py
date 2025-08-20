# Improved Video Combination API Script
# Uses simplified workflow with LoadImageSharedMemory batch mode
# Eliminates ImageBatch nodes for better performance
#
# Features:
# - Single LoadImageSharedMemory node with batch_mode=True
# - Direct connection to VHS_VideoCombine node
# - Simplified workflow with only 2 nodes instead of N+M nodes
# - Improved memory efficiency and performance

import websocket
import uuid
import json
import urllib.request
import urllib.parse
import urllib.error
import numpy as np
import cv2
import os
import sys
import time
from multiprocessing import shared_memory
from typing import List, Optional, Dict, Tuple
from PIL import Image

# Add ComfyUI root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
comfyui_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels: video_stylize_depends -> script_examples -> ComfyUI
if comfyui_root not in sys.path:
    sys.path.insert(0, comfyui_root)

# Import server selection from style transfer script
from script_examples.video_style_transfer_pipeline import select_best_server, COMFYUI_SERVERS

client_id = str(uuid.uuid4())

def queue_prompt(prompt, server_address):
    """Queue prompt to ComfyUI server"""
    p = {"prompt": prompt, "client_id": client_id}
    
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
    req.add_header('Content-Type', 'application/json')
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        print("Server returned error:", e.read().decode())
        raise

def get_execution_result(ws, prompt, server_address):
    """Execute workflow and get results"""
    prompt_id = queue_prompt(prompt, server_address)['prompt_id']
    output_videos = []
    
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break  # Execution is done
        else:
            continue  # Binary message, skip
    
    # Get the output videos
    history = get_history(prompt_id, server_address)
    if prompt_id in history:
        outputs = history[prompt_id]['outputs']
        for node_id in outputs:
            node_output = outputs[node_id]
            if 'gifs' in node_output:
                output_videos.extend(node_output['gifs'])
    
    return output_videos

def get_history(prompt_id, server_address):
    """Get execution history"""
    with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.read())

def send_socket_catch_exception(ws_send_func, message):
    """Send WebSocket message with exception handling"""
    try:
        ws_send_func(message)
    except Exception as e:
        print(f"WebSocket send error: {e}")

def cleanup_shared_memory(shm_name=None, shm_object=None):
    """Clean up shared memory"""
    try:
        if shm_object:
            shm_object.close()
            shm_object.unlink()
        elif shm_name:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
    except Exception as e:
        print(f"Warning: Failed to cleanup shared memory: {e}")

def create_frames_to_shared_memory(frames: List[np.ndarray]) -> List[Tuple[str, List[int], str, shared_memory.SharedMemory]]:
    """Convert frames to shared memory"""
    shared_frames = []
    
    print(f"Converting {len(frames)} frames to shared memory...")
    
    for i, frame in enumerate(frames):
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame {i} is not a numpy array")
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame {i} has invalid shape {frame.shape}, expected (h, w, 3)")
        
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        
        # Create shared memory
        shm_name = f"video_frame_{uuid.uuid4().hex[:16]}_{i:04d}"
        shm_size = frame.nbytes
        
        try:
            shm = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)
            
            # Copy frame data to shared memory
            shm_array = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
            shm_array[:] = frame[:]
            
            shared_frames.append((shm_name, list(frame.shape), str(frame.dtype), shm))
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(frames)} frames")
                
        except Exception as e:
            # Clean up any created shared memory on error
            for _, _, _, shm_obj in shared_frames:
                cleanup_shared_memory(shm_object=shm_obj)
            raise e
    
    print(f"✓ All {len(frames)} frames converted to shared memory")
    return shared_frames

def create_simplified_video_combination_workflow(shared_frames: List[Tuple[str, List[int], str, shared_memory.SharedMemory]], 
                                               frame_rate: int = 30,
                                               output_filename: str = "combined_video",
                                               format_type: str = "video/h264-mp4",
                                               crf: int = 19) -> dict:
    """
    Create simplified video combination workflow using LoadImageSharedMemory batch mode
    
    Args:
        shared_frames: List of (shm_name, shape, dtype, shm_object) tuples
        frame_rate: Video frame rate
        output_filename: Output video filename prefix
        format_type: Video format
        crf: Video quality (lower = better quality)
        
    Returns:
        dict: ComfyUI workflow with only 2 nodes
    """
    if not shared_frames:
        raise ValueError("No shared frames provided")
    
    # Get frame info from first frame
    _, shape, dtype, _ = shared_frames[0]
    
    # Create comma-separated list of shared memory names
    shm_names = [shm_name for shm_name, _, _, _ in shared_frames]
    shm_names_str = ",".join(shm_names)
    
    print(f"Creating simplified workflow with {len(shared_frames)} frames")
    print(f"Frame shape: {shape}, dtype: {dtype}")
    print(f"Shared memory names: {len(shm_names)} names (first: {shm_names[0]})")
    
    workflow = {
        # Single LoadImageSharedMemory node with batch mode
        "load_all_frames": {
            "inputs": {
                "shm_name": shm_names_str,  # Comma-separated shared memory names
                "shape": json.dumps(shape),
                "dtype": dtype,
                "convert_bgr_to_rgb": False,  # Frames are already RGB
                "batch_mode": True  # Enable batch processing
            },
            "class_type": "LoadImageSharedMemory",
            "_meta": {
                "title": f"Load All {len(shared_frames)} Frames (Batch Mode)"
            }
        },
        
        # Direct connection to VHS_VideoCombine node
        "video_combine": {
            "inputs": {
                "frame_rate": frame_rate,
                "loop_count": 0,
                "filename_prefix": output_filename,
                "format": format_type,
                "pix_fmt": "yuv420p",
                "crf": crf,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
                "images": ["load_all_frames", 0]  # Direct connection from batch loader
            },
            "class_type": "VHS_VideoCombine",
            "_meta": {
                "title": "Video Combine"
            }
        }
    }
    
    print(f"✓ Simplified workflow created: 2 nodes instead of {len(shared_frames) + len(shared_frames) - 1} nodes")
    return workflow

def process_video_combination_improved(frames: List[np.ndarray],
                                     output_path: str = "output/combined_video.mp4",
                                     frame_rate: int = 30,
                                     server_address: Optional[str] = None) -> str:
    """
    Process video combination using improved simplified workflow

    Args:
        frames: List of processed frame arrays
        output_path: Output video file path
        frame_rate: Video frame rate
        server_address: ComfyUI server address (auto-select if None)

    Returns:
        str: Path to created video file
    """
    process_start_time = time.time()

    print(f"=== Improved Video Combination Processing ===")
    print(f"Processing video combination:")
    print(f"  Frames: {len(frames)}")
    print(f"  Output: {output_path}")
    print(f"  Frame rate: {frame_rate}")

    if not frames:
        raise ValueError("No frames to combine")

    # Select best server if not specified
    if server_address is None:
        server_address = select_best_server(COMFYUI_SERVERS)
        if server_address is None:
            raise RuntimeError("No available ComfyUI servers found")

    shared_frames = []

    try:
        # 1. Convert frames to shared memory
        conversion_start_time = time.time()
        shared_frames = create_frames_to_shared_memory(frames)
        conversion_time = time.time() - conversion_start_time

        # 2. Create simplified workflow
        workflow_start_time = time.time()
        output_filename = os.path.splitext(os.path.basename(output_path))[0]
        workflow = create_simplified_video_combination_workflow(
            shared_frames=shared_frames,
            frame_rate=frame_rate,
            output_filename=output_filename
        )
        workflow_time = time.time() - workflow_start_time

        # 3. Execute workflow with proper WebSocket cleanup
        execution_start_time = time.time()
        ws = websocket.WebSocket()
        try:
            ws.connect(f"ws://{server_address}/ws?clientId={client_id}")

            print(f"Executing simplified video combination workflow on {server_address}...")
            output_videos = get_execution_result(ws, workflow, server_address)

            execution_time = time.time() - execution_start_time
        finally:
            # Ensure WebSocket is always closed
            try:
                ws.close()
            except Exception as close_error:
                print(f"Warning: Error closing WebSocket: {close_error}")

        # 4. Process results
        if output_videos:
            # Get the generated video file path
            video_info = output_videos[0]
            generated_filename = video_info['filename']

            # Construct full path to generated video
            comfyui_output_dir = os.path.join(comfyui_root, "output")
            generated_path = os.path.join(comfyui_output_dir, generated_filename)

            # Copy to desired output location if different
            if generated_path != output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                import shutil
                shutil.copy2(generated_path, output_path)
                print(f"✓ Video copied to: {output_path}")

            total_process_time = time.time() - process_start_time

            print(f"\n=== Improved Video Combination Summary ===")
            print(f"  Server: {server_address}")
            print(f"  Frame conversion: {conversion_time:.4f}s")
            print(f"  Workflow creation: {workflow_time:.4f}s")
            print(f"  ComfyUI execution: {execution_time:.4f}s")
            print(f"  Total processing time: {total_process_time:.4f}s")
            print(f"  Output video: {output_path}")
            print(f"  Frame count: {len(frames)}")
            print(f"  Frame rate: {frame_rate} fps")
            print(f"  Workflow optimization: 2 nodes vs {len(frames) + len(frames) - 1} nodes (original)")
            print("=" * 50)

            return output_path
        else:
            raise RuntimeError("No video output received from workflow")

    except Exception as e:
        print(f"Error processing video combination on {server_address}: {e}")
        raise
    finally:
        # Clean up shared memory
        for _, _, _, shm_obj in shared_frames:
            cleanup_shared_memory(shm_object=shm_obj)

# Main interface
def comfyui_video_combination_improved(frames: List[np.ndarray],
                                     output_path: str = "output/combined_video.mp4",
                                     frame_rate: int = 30,
                                     **kwargs) -> str:
    """
    Improved video combination interface using simplified workflow

    Args:
        frames: List of processed frame arrays (h, w, 3) RGB format
        output_path: Output video file path
        frame_rate: Video frame rate
        **kwargs: Additional parameters

    Returns:
        str: Path to created video file
    """
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a non-empty list of numpy arrays")

    # Validate frame format
    for i, frame in enumerate(frames):
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame {i + 1} is not a numpy array")
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame {i + 1} has invalid shape {frame.shape}, expected (h, w, 3)")

    return process_video_combination_improved(frames, output_path, frame_rate, **kwargs)

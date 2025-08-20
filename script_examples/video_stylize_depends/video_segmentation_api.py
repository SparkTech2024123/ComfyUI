# Video Segmentation API Script
# Processes videos for shot segmentation and frame extraction using pre-existing JSON metadata
# Port: 8261 (Video Segmentation Service)
#
# Features:
# - Hardcoded workflow for video shot splitting and frame extraction
# - Uses existing JSON shot detection metadata (JSON-only mode)
# - Outputs shot videos and frame images to specified directory
# - Server selection and load balancing for ComfyUI service
# - Comprehensive error handling and progress monitoring

import websocket
import uuid
import json
import urllib.request
import urllib.error
import os
import sys
from typing import Dict, Optional, Any

# Add ComfyUI root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
comfyui_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels: video_stylize_depends -> script_examples -> ComfyUI
if comfyui_root not in sys.path:
    sys.path.insert(0, comfyui_root)

# Configure ComfyUI video segmentation server
COMFYUI_SEGMENTATION_SERVERS = [
    "127.0.0.1:8261"  # Video Segmentation Service (GPU 4)
]

client_id = str(uuid.uuid4())

def check_server_status(server_address: str) -> Dict[str, Any]:
    """
    Check ComfyUI server status
    
    Args:
        server_address: Server address "host:port"
    
    Returns:
        dict: Server status information
    """
    try:
        # Check queue status
        queue_url = f"http://{server_address}/queue"
        queue_req = urllib.request.Request(queue_url)
        queue_req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(queue_req, timeout=5) as response:
            queue_data = json.loads(response.read())
        
        # Check system status
        stats_url = f"http://{server_address}/system_stats"
        stats_req = urllib.request.Request(stats_url)
        
        with urllib.request.urlopen(stats_req, timeout=5) as response:
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

def select_best_server(servers=None, quiet=False) -> Optional[str]:
    """
    Select best ComfyUI server for video segmentation
    
    Args:
        servers: List of server addresses to check
        quiet: If True, suppress detailed output
    
    Returns:
        str: Best server address, None if no servers available
    """
    if servers is None:
        servers = COMFYUI_SEGMENTATION_SERVERS
    
    if not quiet:
        print("=== Checking Video Segmentation Server Status (8261) ===")
    
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
        print("No available video segmentation servers found!")
        return None
    
    # Select server with lowest load, then lowest VRAM usage
    selected_server = min(available_servers, key=lambda x: (x['total_load'], x['vram_usage_ratio']))
    
    if not quiet:
        print(f"Selected server: {selected_server['server_address']} (load={selected_server['total_load']})")
        print("=" * 50)
    
    return selected_server['server_address']

def queue_prompt(prompt: Dict, server_address: str) -> Dict:
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

def get_execution_result(ws, prompt: Dict, server_address: str) -> Dict:
    """
    Execute workflow and get results
    
    Args:
        ws: WebSocket connection
        prompt: Workflow prompt
        server_address: Server address
    
    Returns:
        dict: Execution results
    """
    prompt_id = queue_prompt(prompt, server_address)['prompt_id']
    execution_error = None
    results = {}
    
    print(f"Executing video segmentation workflow {prompt_id} on {server_address}...")
    
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            
            if message['type'] == 'executing':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    if data['node'] is None:
                        print("✓ Video segmentation completed")
                        break
                    else:
                        node_id = data['node']
                        print(f"Processing node: {node_id}")
            
            elif message['type'] == 'executed':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    if 'output' in data:
                        results.update(data['output'])
            
            elif message['type'] == 'execution_error':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    execution_error = data
                    print(f"Execution error: {execution_error}")
                    break
    
    if execution_error:
        raise Exception(f"Workflow execution failed: {execution_error}")
    
    return results

def create_video_segmentation_workflow(video_path: str, json_file_path: str, output_directory: str) -> Dict:
    """
    Create hardcoded video segmentation workflow for JSON-only mode
    
    Args:
        video_path: Path to input video
        json_file_path: Path to existing JSON file with shot detection metadata
        output_directory: Output directory for results
    
    Returns:
        dict: Hardcoded workflow prompt in API format
    """
    # Hardcoded workflow with 2 nodes: VideoShotJSONLoader -> VideoShotSplitter
    workflow = {
        "1": {
            "inputs": {
                "json_file_path": json_file_path
            },
            "class_type": "VideoShotJSONLoader",
            "_meta": {
                "title": "Video Shot JSON Loader"
            }
        },
        "2": {
            "inputs": {
                "video_path": video_path,
                "shot_json": ["1", 0],  # Connect to VideoShotJSONLoader output
                "output_directory": output_directory,
                "enable_shot_splitting": True,
                "enable_frame_extraction": True,
                "max_workers": 0,  # Auto-detect
                "force_cpu": False
            },
            "class_type": "VideoShotSplitter",
            "_meta": {
                "title": "Video Shot Splitter"
            }
        }
    }
    
    return workflow

def comfyui_video_segmentation(video_path: str, json_file_path: str, output_directory: str) -> str:
    """
    Video segmentation interface for ComfyUI (JSON-only mode)
    
    Args:
        video_path: Path to input video file
        json_file_path: Path to existing shot detection JSON file (required)
        output_directory: Output directory for shot videos and frames
    
    Returns:
        str: Path to output directory containing results
    """
    print(f"🎬 Starting video segmentation (JSON mode)...")
    print(f"   Video: {video_path}")
    print(f"   JSON: {json_file_path}")
    print(f"   Output: {output_directory}")
    
    # Validate inputs
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not json_file_path or not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Select best server
    server_address = select_best_server()
    if not server_address:
        raise Exception("No available video segmentation servers")
    
    try:
        # Create hardcoded workflow
        prompt = create_video_segmentation_workflow(video_path, json_file_path, output_directory)
        
        # Execute workflow
        ws_url = f"ws://{server_address}/ws?clientId={client_id}"
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        
        try:
            get_execution_result(ws, prompt, server_address)
            
            # Process results
            print("✅ Video segmentation completed successfully")
            
            # Log results summary
            shots_dir = os.path.join(output_directory, "shots")
            frames_dir = os.path.join(output_directory, "frames")
            
            if os.path.exists(shots_dir):
                shot_files = [f for f in os.listdir(shots_dir) if f.endswith('.mp4')]
                print(f"📹 Generated {len(shot_files)} shot videos in: {shots_dir}")
            
            if os.path.exists(frames_dir):
                frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
                print(f"🖼️ Extracted {len(frame_files)} frame images in: {frames_dir}")
            
            return output_directory
            
        finally:
            ws.close()
            
    except Exception as e:
        print(f"❌ Video segmentation failed: {e}")
        raise

if __name__ == "__main__":
    # Test the video segmentation API
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python video_segmentation_api.py <video_path> <json_file_path> <output_directory>")
        print("Note: JSON file with shot detection metadata is required for JSON-only mode")
        sys.exit(1)
    
    video_path = sys.argv[1]
    json_file_path = sys.argv[2]
    output_directory = sys.argv[3]
    
    try:
        result_dir = comfyui_video_segmentation(video_path, json_file_path, output_directory)
        print(f"✅ Success! Results saved to: {result_dir}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
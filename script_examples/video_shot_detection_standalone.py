#!/usr/bin/env python3
"""
Video Shot Detection Standalone Script
=====================================

This script performs video shot detection using ComfyUI's video segmentation nodes.
It generates a JSON file containing shot metadata (frame ranges and timestamps) 
for the input video using ModelScope's cv_resnet50-bert video segmentation model.

Features:
- Standalone script for video shot detection
- Uses ComfyUI VideoSegmentationModelLoader and VideoShotDetector nodes
- Connects to ComfyUI server on port 8261
- Generates standard format JSON output
- Automatic model cache cleanup after processing
- Comprehensive error handling and logging

Usage:
    python video_shot_detection_standalone.py <video_path> [output_json_path]

Arguments:
    video_path: Path to input video file
    output_json_path: Optional output JSON file path (auto-generated if not provided)

Example:
    python video_shot_detection_standalone.py ./input/video.mp4
    python video_shot_detection_standalone.py ./input/video.mp4 ./output/video_shots.json

Author: Generated for ComfyUI Video Processing Pipeline
"""

import os
import sys
import json
import uuid
import time
import urllib.request
import urllib.error
import websocket
from typing import Dict, Optional, Any


# =============================================================================
# CONFIGURATION
# =============================================================================

# ComfyUI server configuration
COMFYUI_SERVER_ADDRESS = "127.0.0.1:8261"
CLIENT_ID = str(uuid.uuid4())

# Default device mode for model loading
DEFAULT_DEVICE_MODE = "auto"  # auto, cpu, gpu


# =============================================================================
# UTILITY FUNCTIONS
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


def check_server_availability(server_address: str) -> bool:
    """Check if ComfyUI server is available"""
    try:
        queue_url = f"http://{server_address}/queue"
        req = urllib.request.Request(queue_url)
        req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.getcode() == 200
    except Exception:
        return False


def queue_prompt(prompt: Dict, server_address: str) -> Dict:
    """Queue prompt to ComfyUI server"""
    p = {"prompt": prompt, "client_id": CLIENT_ID}
    
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
        Dict containing execution results
    """
    prompt_id = queue_prompt(prompt, server_address)['prompt_id']
    results = {}
    execution_error = None
    
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            
            if message['type'] == 'executing':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    if data['node'] is None:
                        print("✓ Shot detection completed")
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


# =============================================================================
# WORKFLOW CREATION
# =============================================================================

def create_shot_detection_workflow(video_path: str, device_mode: str = "auto") -> Dict:
    """
    Create hardcoded shot detection workflow using VideoSegmentationModelLoader and VideoShotDetector

    Args:
        video_path: Path to input video file
        device_mode: Device mode for model loading (auto, cpu, gpu)

    Returns:
        dict: Hardcoded workflow prompt in API format
    """
    # Convert to absolute path to ensure ComfyUI can find the file
    abs_video_path = os.path.abspath(video_path)

    workflow = {
        "1": {
            "inputs": {
                "device_mode": device_mode
            },
            "class_type": "VideoSegmentationModelLoader",
            "_meta": {
                "title": "Video Segmentation Model Loader"
            }
        },
        "2": {
            "inputs": {
                "video_path": abs_video_path,
                "segmentation_model": ["1", 0],  # Connect to VideoSegmentationModelLoader output
                "save_json_to_file": False,  # We'll handle JSON output manually
                "output_json_path": ""
            },
            "class_type": "VideoShotDetector",
            "_meta": {
                "title": "Video Shot Detector"
            }
        },
        "3": {
            "inputs": {
                "source": ["2", 0]  # Connect to VideoShotDetector shot_json output
            },
            "class_type": "PreviewAny",
            "_meta": {
                "title": "Preview Shot Detection JSON"
            }
        }
    }

    return workflow


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def detect_video_shots(video_path: str, output_json_path: Optional[str] = None,
                      device_mode: str = "auto") -> str:
    """
    Detect shots in video and generate JSON metadata file

    Args:
        video_path: Path to input video file
        output_json_path: Optional output JSON file path
        device_mode: Device mode for model loading

    Returns:
        str: Path to generated JSON file
    """
    # Validate input video
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Generate output JSON path if not provided
    if not output_json_path:
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(os.path.abspath(video_path))
        output_json_path = os.path.join(video_dir, f"{video_basename}_shot_detection.json")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"🎬 Starting video shot detection...")
    print(f"   Input video: {video_path}")
    print(f"   Output JSON: {output_json_path}")
    print(f"   Device mode: {device_mode}")
    print(f"   Server: {COMFYUI_SERVER_ADDRESS}")

    # Check server availability
    print("🔍 Checking server availability...")
    if not check_server_availability(COMFYUI_SERVER_ADDRESS):
        raise Exception(f"ComfyUI server not available at {COMFYUI_SERVER_ADDRESS}")
    print(f"✅ Server {COMFYUI_SERVER_ADDRESS} is available")

    ws = None
    try:
        # Create workflow
        print("🔧 Creating shot detection workflow...")
        prompt = create_shot_detection_workflow(video_path, device_mode)
        print(f"   Workflow nodes: VideoSegmentationModelLoader → VideoShotDetector → PreviewAny")

        # Execute workflow
        print("🚀 Executing workflow...")
        ws_url = f"ws://{COMFYUI_SERVER_ADDRESS}/ws?clientId={CLIENT_ID}"
        ws = websocket.WebSocket()

        print(f"   Connecting to WebSocket: {ws_url}")
        ws.connect(ws_url)
        print("   WebSocket connection established")

        # Execute and get results
        print("   Starting workflow execution...")
        results = get_execution_result(ws, prompt, COMFYUI_SERVER_ADDRESS)

        # Extract shot detection results from PreviewAny node
        print("📊 Processing results...")
        print(f"   Available result keys: {list(results.keys())}")

        # Check if results contain 'text' directly (PreviewAny UI output format)
        if "text" in results:
            shot_json_string = results["text"][0]
            print("   Found shot detection results in 'text' key")
        elif "3" in results:
            # Check if node 3 (PreviewAny) has the output
            preview_output = results["3"]
            print(f"   Node 3 output keys: {list(preview_output.keys())}")
            if "text" in preview_output:
                shot_json_string = preview_output["text"][0]
                print("   Found shot detection results in node 3 'text' key")
            else:
                available_outputs = list(preview_output.keys())
                raise Exception(f"No 'text' output found in PreviewAny node. Available outputs: {available_outputs}")
        else:
            available_keys = list(results.keys())
            raise Exception(f"No shot detection results found in workflow output. Available keys: {available_keys}")

        # Validate JSON format
        try:
            shot_data = json.loads(shot_json_string)
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format in shot detection results: {e}")

        # Validate shot data structure
        if not isinstance(shot_data, dict):
            raise Exception("Shot detection results must be a dictionary")

        if "shot_meta_list" not in shot_data:
            raise Exception("Shot detection results missing 'shot_meta_list' key")

        shot_count = shot_data.get('shot_num', 0)
        shot_list = shot_data.get('shot_meta_list', [])

        if shot_count != len(shot_list):
            print(f"⚠️ Warning: shot_num ({shot_count}) doesn't match shot_meta_list length ({len(shot_list)})")

        # Save JSON file
        print(f"💾 Saving results to: {output_json_path}")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(shot_data, f, indent=2, ensure_ascii=False)

        # Verify file was saved correctly
        if not os.path.exists(output_json_path):
            raise Exception(f"Failed to save JSON file: {output_json_path}")

        file_size = os.path.getsize(output_json_path)
        print(f"   File size: {file_size} bytes")

        print("✅ Shot detection completed successfully!")
        print(f"📊 Detected {shot_count} shots")
        print(f"💾 Results saved to: {output_json_path}")

        return output_json_path

    except websocket.WebSocketException as e:
        raise Exception(f"WebSocket connection failed: {e}")
    except urllib.error.URLError as e:
        raise Exception(f"Network error: {e}")
    except Exception as e:
        print(f"❌ Shot detection failed: {e}")
        raise

    finally:
        # Close WebSocket connection
        if ws:
            try:
                ws.close()
                print("   WebSocket connection closed")
            except Exception as e:
                print(f"⚠️ Warning: Failed to close WebSocket: {e}")

        # Clean up model cache
        print("🗑️ Cleaning up model cache...")
        clear_model_cache(COMFYUI_SERVER_ADDRESS, quiet=False)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def validate_video_file(video_path: str) -> bool:
    """
    Validate video file format and accessibility

    Args:
        video_path: Path to video file

    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False

    # Check file extension
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    file_ext = os.path.splitext(video_path)[1].lower()

    if file_ext not in valid_extensions:
        print(f"⚠️ Warning: Unsupported video format '{file_ext}'. Supported formats: {', '.join(valid_extensions)}")
        print("   Proceeding anyway - ModelScope may still be able to process it.")

    # Check file size
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        print(f"❌ Video file is empty: {video_path}")
        return False

    file_size_mb = file_size / (1024 * 1024)
    print(f"📊 Video file size: {file_size_mb:.1f} MB")

    if file_size_mb > 1000:  # 1GB
        print(f"⚠️ Warning: Large video file ({file_size_mb:.1f} MB). Processing may take a long time.")

    return True


def print_usage():
    """Print detailed usage information"""
    print("Video Shot Detection Standalone Script")
    print("=" * 50)
    print("This script performs video shot detection using ComfyUI's video segmentation nodes.")
    print("It generates a JSON file containing shot metadata for the input video.")
    print()
    print("Usage:")
    print("  python video_shot_detection_standalone.py <video_path> [output_json_path] [options]")
    print()
    print("Arguments:")
    print("  video_path       Path to input video file (required)")
    print("  output_json_path Optional output JSON file path (auto-generated if not provided)")
    print()
    print("Options:")
    print("  --device-mode    Device mode: auto, cpu, gpu (default: auto)")
    print("  --server         ComfyUI server address (default: 127.0.0.1:8261)")
    print("  --help, -h       Show this help message")
    print()
    print("Examples:")
    print("  python video_shot_detection_standalone.py ./input/video.mp4")
    print("  python video_shot_detection_standalone.py ./input/video.mp4 ./output/shots.json")
    print("  python video_shot_detection_standalone.py ./input/video.mp4 --device-mode cpu")
    print()
    print("Output JSON Format:")
    print("  {")
    print('    "shot_num": 5,')
    print('    "shot_meta_list": [')
    print('      {')
    print('        "frame": ["0", "150"],')
    print('        "timestamps": ["00:00:00.000", "00:00:05.000"]')
    print('      },')
    print('      ...')
    print('    ]')
    print('  }')


def parse_arguments():
    """Parse command line arguments"""
    args = sys.argv[1:]

    if not args or '--help' in args or '-h' in args:
        print_usage()
        sys.exit(0 if '--help' in args or '-h' in args else 1)

    # Parse positional arguments
    video_path = None
    output_json_path = None
    device_mode = DEFAULT_DEVICE_MODE
    server_address = COMFYUI_SERVER_ADDRESS

    # Extract positional arguments (non-option arguments)
    positional_args = [arg for arg in args if not arg.startswith('--')]

    if len(positional_args) < 1:
        print("❌ Error: Video path is required")
        print_usage()
        sys.exit(1)

    video_path = positional_args[0]
    if len(positional_args) > 1:
        output_json_path = positional_args[1]

    # Parse options
    i = 0
    while i < len(args):
        if args[i] == '--device-mode' and i + 1 < len(args):
            device_mode = args[i + 1]
            if device_mode not in ['auto', 'cpu', 'gpu']:
                print(f"❌ Error: Invalid device mode '{device_mode}'. Must be: auto, cpu, gpu")
                sys.exit(1)
            i += 2
        elif args[i] == '--server' and i + 1 < len(args):
            server_address = args[i + 1]
            i += 2
        else:
            i += 1

    return video_path, output_json_path, device_mode, server_address


def main():
    """Main function for command line interface"""
    try:
        # Parse arguments
        video_path, output_json_path, device_mode, server_address = parse_arguments()

        # Update global server address if specified
        global COMFYUI_SERVER_ADDRESS
        COMFYUI_SERVER_ADDRESS = server_address

        # Validate video file
        if not validate_video_file(video_path):
            sys.exit(1)

        # Start processing
        start_time = time.time()
        result_path = detect_video_shots(video_path, output_json_path, device_mode)
        processing_time = time.time() - start_time

        print(f"\n🎉 Success! Shot detection completed in {processing_time:.1f} seconds")
        print(f"📄 JSON file saved to: {result_path}")

        # Display summary of results
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                shot_data = json.load(f)

            shot_count = shot_data.get('shot_num', 0)
            print(f"📊 Summary: {shot_count} shots detected")

            if shot_count > 0:
                shot_list = shot_data.get('shot_meta_list', [])
                if shot_list:
                    first_shot = shot_list[0]
                    last_shot = shot_list[-1]
                    print(f"   First shot: frames {first_shot['frame'][0]}-{first_shot['frame'][1]} ({first_shot['timestamps'][0]} - {first_shot['timestamps'][1]})")
                    print(f"   Last shot:  frames {last_shot['frame'][0]}-{last_shot['frame'][1]} ({last_shot['timestamps'][0]} - {last_shot['timestamps'][1]})")

        except Exception as e:
            print(f"⚠️ Warning: Could not read result summary: {e}")

    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Video Shot Segmentation ComfyUI Plugin

This plugin provides three nodes for intelligent video shot segmentation:
1. VideoSegmentationModelLoader - Loads the ModelScope segmentation model with GPU detection
2. VideoShotDetector - Detects shots in video and outputs JSON metadata  
3. VideoShotSplitter - Splits video into shots and extracts frames based on JSON metadata

Based on the cv_resnet50-bert_video-scene-segmentation_movienet model.
"""

import os
import sys
import json
import urllib.request
import subprocess
import time
import multiprocessing
from datetime import datetime
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
import folder_paths


# ===================== UTILITY FUNCTIONS =====================

def is_url(path: str) -> bool:
    """Check if path is URL"""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False


def download_video(url: str, output_path: str = "temp_video.mp4") -> Optional[str]:
    """Download network video to local"""
    print(f"📥 Downloading video: {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Video downloaded successfully: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Video download failed: {e}")
        return None


def timestamp_to_seconds(timestamp_str: str) -> float:
    """Convert timestamp string (HH:MM:SS.mmm) to seconds"""
    try:
        parts = timestamp_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        print(f"⚠️ Warning: Cannot parse timestamp {timestamp_str}")
        return 0.0


def create_output_dirs(base_dir: str = 'output') -> None:
    """Create necessary output directories"""
    dirs = [base_dir, f'{base_dir}/shots', f'{base_dir}/frames']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f"📁 Output directories created: {base_dir}")


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


def check_nvidia_gpu_available() -> Tuple[bool, bool, Dict[str, str]]:
    """Check if NVIDIA GPU and NVENC encoder are available"""
    try:
        # Check nvidia-smi
        gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=5)
        gpu_available = gpu_result.returncode == 0
        gpu_name = gpu_result.stdout.strip() if gpu_available else "Unknown"
        
        # Check NVENC encoder
        if gpu_available:
            enc_result = subprocess.run(['ffmpeg', '-encoders'], 
                                      capture_output=True, text=True, timeout=5)
            encoder_available = enc_result.returncode == 0 and 'h264_nvenc' in enc_result.stdout
            
            if encoder_available:
                print(f"🚀 GPU acceleration available: {gpu_name}")
            else:
                print(f"⚠️ GPU detected but NVENC unavailable: {gpu_name}")
        else:
            encoder_available = False
            
        return gpu_available, encoder_available, {'name': gpu_name}
        
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False, False, {}


def get_optimal_worker_count(total_shots: int, gpu_available: bool = False) -> int:
    """Calculate optimal worker count based on system resources and task count"""
    cpu_count = multiprocessing.cpu_count()
    
    # GPU encoding can support more concurrency
    if gpu_available:
        base_workers = min(cpu_count, 8)  # Max 8 concurrent
    else:
        base_workers = min(cpu_count // 2, 4)  # Half CPU cores, max 4
    
    # Adjust based on actual task count
    optimal_workers = min(base_workers, total_shots)
    
    # At least 1 worker
    return max(1, optimal_workers)


def get_video_info_ffprobe(video_path: str) -> Optional[Dict[str, Any]]:
    """Get video information using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ ffprobe execution failed: {result.stderr}")
            return None
            
        probe_data = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in probe_data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
                
        if not video_stream:
            print("❌ No video stream found")
            return None
            
        # Extract video info
        info = {
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'fps': eval(video_stream.get('r_frame_rate', '0/1')),  # Calculate framerate
            'frame_count': int(video_stream.get('nb_frames', 0)),
            'duration': float(probe_data.get('format', {}).get('duration', 0)),
            'codec': video_stream.get('codec_name', 'unknown'),
            'format': probe_data.get('format', {}).get('format_name', 'unknown')
        }
        
        # If nb_frames not available, calculate from duration and fps
        if info['frame_count'] == 0 and info['duration'] > 0 and info['fps'] > 0:
            info['frame_count'] = int(info['duration'] * info['fps'])
            
        print(f"📊 Video: {info['width']}x{info['height']}, {info['duration']:.1f}s")
        return info
        
    except subprocess.TimeoutExpired:
        print("❌ ffprobe execution timeout")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ ffprobe output parsing failed: {e}")
        return None
    except Exception as e:
        print(f"❌ ffprobe execution error: {e}")
        return None


def process_single_shot(args: Tuple[str, Dict, int, str, bool, Dict]) -> Optional[Dict]:
    """Process single shot function"""
    video_path, metadata, shot_index, output_dir, use_gpu, encoder_settings = args
    
    shot_id = f"{shot_index:04d}"
    start_seconds = timestamp_to_seconds(metadata['timestamps'][0])
    end_seconds = timestamp_to_seconds(metadata['timestamps'][1])
    duration = end_seconds - start_seconds
    
    if duration <= 0:
        return None
    
    output_file = f"{output_dir}/shots/shot_{shot_id}.mp4"
    start_time = time.time()
    
    # Build FFmpeg command
    cmd = ['ffmpeg', '-y', '-ss', f"{start_seconds:.3f}", '-i', video_path, '-t', f"{duration:.3f}"]
    
    if use_gpu and encoder_settings.get('gpu_available', False):
        cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '18', '-b:v', '0'])
        method = 'GPU'
    else:
        cmd.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '18'])
        method = 'CPU'
    
    cmd.extend(['-c:a', 'aac', '-movflags', '+faststart', output_file])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        process_time = time.time() - start_time
        
        if result.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            # Verify duration
            video_info = get_video_info_ffprobe(output_file)
            actual_duration = video_info['duration'] if video_info else duration
            
            print(f"✅ Shot {shot_id}: {method} {process_time:.1f}s")
            
            return {
                'id': shot_id,
                'frame_range': metadata['frame'],
                'timestamps': metadata['timestamps'],
                'start_seconds': start_seconds,
                'end_seconds': end_seconds,
                'video_file': output_file,
                'file_size': os.path.getsize(output_file),
                'expected_duration': duration,
                'actual_duration': actual_duration,
                'method': method.lower(),
                'process_time': process_time
            }
        else:
            print(f"❌ Shot {shot_id}: Encoding failed")
            return None
            
    except Exception as e:
        print(f"❌ Shot {shot_id}: {str(e)[:50]}...")
        return None


def segment_video_by_shots(video_path: str, shot_meta_list: List[Dict], 
                          output_dir: str = 'output', max_workers: Optional[int] = None, 
                          force_cpu: bool = False) -> List[Dict]:
    """Use parallel processing and GPU acceleration for video segmentation"""
    if not os.path.exists(video_path) or not check_ffmpeg_available():
        return []
    
    # Detect GPU and encoder
    gpu_available, encoder_available, gpu_info = check_nvidia_gpu_available()
    use_gpu = gpu_available and encoder_available and not force_cpu
    
    if force_cpu and use_gpu:
        print("⚠️ Force CPU mode")
        use_gpu = False
    
    # Setup parameters
    os.makedirs(f"{output_dir}/shots", exist_ok=True)
    total_shots = len(shot_meta_list)
    if max_workers is None:
        max_workers = get_optimal_worker_count(total_shots, use_gpu)
    
    print(f"📊 Processing {total_shots} shots, {max_workers} workers, {'GPU' if use_gpu else 'CPU'}")
    
    # Parallel processing
    processing_args = [(video_path, metadata, i, output_dir, use_gpu, {'gpu_available': encoder_available}) 
                      for i, metadata in enumerate(shot_meta_list)]
    
    shot_results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_shot, args): args for args in processing_args}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                shot_results.append(result)
    
    total_time = time.time() - start_time
    print(f"🎉 Complete: {len(shot_results)}/{total_shots} success, {total_time:.1f}s")
    
    return sorted(shot_results, key=lambda x: x['id'])


def extract_shot_frames_ffmpeg(original_video_path: str, shot_info: Dict, 
                              output_dir: str = 'output') -> Dict[str, str]:
    """Extract shot frames using FFmpeg directly from original video"""
    if not os.path.exists(original_video_path):
        print(f"❌ Original video not found: {original_video_path}")
        return {}
    
    if not check_ffmpeg_available():
        print("⚠️ FFmpeg unavailable, cannot extract frames")
        return {}
    
    shot_id = shot_info['id']
    frames_dir = f"{output_dir}/frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    first_frame_path = f"{frames_dir}/shot_{shot_id}_first.jpg"
    last_frame_path = f"{frames_dir}/shot_{shot_id}_last.jpg"
    
    frame_paths = {}
    
    # Get timestamp info
    start_seconds = shot_info.get('start_seconds')
    end_seconds = shot_info.get('end_seconds')
    
    # If no precise seconds, calculate from timestamps
    if start_seconds is None or end_seconds is None:
        start_time = shot_info['timestamps'][0]
        end_time = shot_info['timestamps'][1]
        start_seconds = timestamp_to_seconds(start_time)
        end_seconds = timestamp_to_seconds(end_time)
    
    # Adjust timestamps to avoid boundary issues
    first_time = start_seconds + 0.001
    last_time = max(end_seconds - 0.1, start_seconds + 0.001)
    
    print(f"📸 Extract shot {shot_id} frames: first@{first_time:.3f}s, last@{last_time:.3f}s")
    
    # Extract first frame
    try:
        cmd_first = [
            'ffmpeg', '-y',
            '-ss', f"{first_time:.3f}",
            '-i', original_video_path,
            '-vframes', '1',
            '-q:v', '2',
            '-f', 'image2',
            first_frame_path
        ]
        
        result = subprocess.run(cmd_first, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(first_frame_path):
            frame_paths['first_frame'] = first_frame_path
            print(f"✅ First frame saved: {first_frame_path}")
        else:
            print(f"❌ First frame extraction failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ First frame extraction error: {e}")
    
    # Extract last frame (if sufficient time)
    if last_time > first_time + 0.01:  # At least 0.01s difference
        try:
            cmd_last = [
                'ffmpeg', '-y',
                '-ss', f"{last_time:.3f}",
                '-i', original_video_path,
                '-vframes', '1',
                '-q:v', '2',
                '-f', 'image2',
                last_frame_path
            ]
            
            result = subprocess.run(cmd_last, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(last_frame_path):
                frame_paths['last_frame'] = last_frame_path
                print(f"✅ Last frame saved: {last_frame_path}")
            else:
                print(f"❌ Last frame extraction failed: {result.stderr}")
                # Use first frame as last frame if failed
                if 'first_frame' in frame_paths:
                    frame_paths['last_frame'] = first_frame_path
                    print(f"⚠️ Using first frame as last frame")
                
        except Exception as e:
            print(f"❌ Last frame extraction error: {e}")
            # Use first frame as last frame if failed
            if 'first_frame' in frame_paths:
                frame_paths['last_frame'] = first_frame_path
                print(f"⚠️ Using first frame as last frame")
    else:
        # Shot too short, use first frame as last frame
        if 'first_frame' in frame_paths:
            frame_paths['last_frame'] = first_frame_path
            print(f"⚠️ Shot too short, using first frame as last frame")
    
    return frame_paths


# ===================== COMFYUI NODE CLASSES =====================

class VideoSegmentationModelLoader:
    """ComfyUI Node for loading ModelScope video segmentation model with GPU detection"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device_mode": (["auto", "cpu", "gpu"], {
                    "default": "auto",
                    "tooltip": "Device selection: auto (detect), cpu (force CPU), gpu (force GPU)"
                }),
            },
        }

    RETURN_TYPES = ("MODEL_SEGMENTATION",)
    RETURN_NAMES = ("segmentation_model",)
    FUNCTION = "load_model"
    CATEGORY = "video/segmentation"
    
    def load_model(self, device_mode: str) -> Tuple[Dict]:
        """Load the ModelScope video segmentation model"""
        try:
            # Import ModelScope modules
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            
            # Check GPU availability
            gpu_available, encoder_available, gpu_info = check_nvidia_gpu_available()
            
            print("🔧 Loading ModelScope segmentation model...")
            
            # Load model
            video_scene_seg = pipeline(
                Tasks.movie_scene_segmentation, 
                model='damo/cv_resnet50-bert_video-scene-segmentation_movienet', 
                model_revision='v1.0.2'
            )
            
            model_info = {
                'pipeline': video_scene_seg,
                'gpu_available': gpu_available,
                'encoder_available': encoder_available,
                'gpu_info': gpu_info,
                'device_mode': device_mode,
                'model_name': 'cv_resnet50-bert_video-scene-segmentation_movienet',
                'model_version': 'v1.0.2'
            }
            
            print("✅ Model loaded successfully")
            if gpu_available:
                print(f"🚀 GPU detected: {gpu_info.get('name', 'Unknown')}")
            else:
                print("💻 CPU mode")
                
            return (model_info,)
            
        except ImportError as e:
            raise Exception(f"ModelScope import failed: {e}\nPlease install: pip install modelscope")
        except Exception as e:
            raise Exception(f"Model loading failed: {e}")


class VideoShotDetector:
    """ComfyUI Node for detecting shots in video and outputting JSON metadata"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to video file or URL"
                }),
                "segmentation_model": ("MODEL_SEGMENTATION", {
                    "tooltip": "Loaded segmentation model from VideoSegmentationModelLoader"
                }),
                "save_json_to_file": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save shot detection results to JSON file"
                }),
                "output_json_path": ("STRING", {
                    "default": "",
                    "tooltip": "Output JSON file path (optional, auto-generated if empty)"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("shot_json",)
    FUNCTION = "detect_shots"
    CATEGORY = "video/segmentation"
    
    def detect_shots(self, video_path: str, segmentation_model: Dict, save_json_to_file: bool, output_json_path: str) -> Tuple[str]:
        """Detect shots in video and return JSON metadata"""
        if not video_path.strip():
            raise ValueError("Video path cannot be empty")
            
        # Handle URL or local file
        temp_video_path = None
        if is_url(video_path):
            temp_video_path = download_video(video_path)
            if not temp_video_path:
                raise Exception("Failed to download video from URL")
            actual_video_path = temp_video_path
        else:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            actual_video_path = video_path
        
        try:
            # Get model pipeline
            pipeline = segmentation_model['pipeline']
            
            print("🔄 Executing video shot detection...")
            result = pipeline(actual_video_path)
            print("✅ Shot detection complete")
            
            # Create shot-only results (excluding scene information)
            shot_results = {
                "shot_num": result.get("shot_num", 0),
                "shot_meta_list": result.get("shot_meta_list", [])
            }
            
            # Save JSON to file if requested
            if save_json_to_file:
                try:
                    # Generate filename if not provided
                    if not output_json_path.strip():
                        # Handle URL vs local file path differently
                        if is_url(video_path):
                            # For URLs, extract filename from URL and save in current directory
                            video_basename = os.path.splitext(os.path.basename(video_path))[0]
                            output_json_path = f"{video_basename}_shot_detection.json"
                        else:
                            # For local files, save in the same directory as the video
                            video_dir = os.path.dirname(os.path.abspath(video_path))
                            video_basename = os.path.splitext(os.path.basename(video_path))[0]
                            output_json_path = os.path.join(video_dir, f"{video_basename}_shot_detection.json")

                    # Ensure directory exists
                    json_dir = os.path.dirname(output_json_path)
                    if json_dir and not os.path.exists(json_dir):
                        os.makedirs(json_dir, exist_ok=True)
                    
                    # Save JSON file
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(shot_results, f, indent=2, ensure_ascii=False)
                    
                    print(f"💾 Shot detection results saved to: {output_json_path}")
                    
                except Exception as e:
                    print(f"⚠️ Warning: Failed to save JSON file: {e}")
                    # Continue execution - don't fail the whole operation
            
            # Return as JSON string
            return (json.dumps(shot_results, indent=2, ensure_ascii=False),)
            
        except Exception as e:
            raise Exception(f"Shot detection failed: {e}")
            
        finally:
            # Cleanup temporary file
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                print("🗑️ Temporary video file cleaned up")


class VideoShotSplitter:
    """ComfyUI Node for splitting video into shots and extracting frames"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to video file or URL"
                }),
                "shot_json": ("STRING", {
                    "default": "",
                    "tooltip": "JSON string from VideoShotDetector containing shot metadata"
                }),
                "output_directory": ("STRING", {
                    "default": "output",
                    "tooltip": "Output directory for shot videos and frames"
                }),
                "enable_shot_splitting": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable shot video splitting"
                }),
                "enable_frame_extraction": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable extraction of first and last frame images for each shot"
                }),
                "max_workers": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16,
                    "tooltip": "Maximum worker threads (0 = auto-detect)"
                }),
                "force_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force CPU encoding (disable GPU acceleration)"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_directory", "processing_results")
    FUNCTION = "split_video"
    OUTPUT_NODE = True
    CATEGORY = "video/segmentation"
    
    def split_video(self, video_path: str, shot_json: str, output_directory: str,
                   enable_shot_splitting: bool, enable_frame_extraction: bool, max_workers: int, force_cpu: bool) -> Tuple[str, str]:
        """Split video into shots and extract frames based on JSON metadata"""
        
        if not video_path.strip():
            raise ValueError("Video path cannot be empty")
            
        if not shot_json.strip():
            raise ValueError("Shot JSON cannot be empty")
        
        # Parse shot metadata
        try:
            shot_data = json.loads(shot_json)
            shot_meta_list = shot_data.get("shot_meta_list", [])
            if not shot_meta_list:
                raise ValueError("No shot metadata found in JSON")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        
        # Handle URL or local file
        temp_video_path = None
        if is_url(video_path):
            temp_video_path = download_video(video_path)
            if not temp_video_path:
                raise Exception("Failed to download video from URL")
            actual_video_path = temp_video_path
        else:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            actual_video_path = video_path
        
        try:
            # Create output directories
            create_output_dirs(output_directory)
            
            # Set worker count
            actual_max_workers = max_workers if max_workers > 0 else None

            # Initialize shot segments list
            shot_segments = []

            if enable_shot_splitting:
                print("📹 Starting video splitting...")
                # Split video into shots
                shot_segments = segment_video_by_shots(
                    actual_video_path,
                    shot_meta_list,
                    output_directory,
                    actual_max_workers,
                    force_cpu
                )

                if not shot_segments:
                    print("❌ Segmentation failed, trying single-thread mode...")
                    shot_segments = segment_video_by_shots(
                        actual_video_path,
                        shot_meta_list,
                        output_directory,
                        1,
                        True
                    )

                    if not shot_segments:
                        raise Exception("Video segmentation failed")
            else:
                print("⏭️ Shot splitting disabled, preparing metadata for frame extraction...")
                # Create shot metadata without actual video splitting
                for i, metadata in enumerate(shot_meta_list):
                    shot_id = f"{i:04d}"
                    start_seconds = timestamp_to_seconds(metadata['timestamps'][0])
                    end_seconds = timestamp_to_seconds(metadata['timestamps'][1])

                    shot_segments.append({
                        'id': shot_id,
                        'frame_range': metadata['frame'],
                        'timestamps': metadata['timestamps'],
                        'start_seconds': start_seconds,
                        'end_seconds': end_seconds,
                        'video_file': None,  # No video file created
                        'file_size': 0,
                        'expected_duration': end_seconds - start_seconds,
                        'actual_duration': end_seconds - start_seconds,
                        'method': 'frame_only',
                        'process_time': 0
                    })

            # Extract frames for each shot (if enabled)
            if enable_frame_extraction:
                print("🖼️ Extracting frames...")
                for segment_info in shot_segments:
                    frame_paths = extract_shot_frames_ffmpeg(
                        actual_video_path,  # Use original video path
                        segment_info,
                        output_directory
                    )
                    segment_info.update(frame_paths)
            else:
                print("⏭️ Frame extraction disabled")

            # Create processing results
            processing_results = {
                'original_video': video_path,
                'processed_at': datetime.now().isoformat(),
                'output_directory': output_directory,
                'shot_data': shot_data,
                'segmentation_results': {
                    'shots': shot_segments
                },
                'summary': {
                    'total_shots': len(shot_segments),
                    'successful_shots': len([s for s in shot_segments if s.get('video_file')]),
                    'shot_splitting_enabled': enable_shot_splitting,
                    'frame_extraction_enabled': enable_frame_extraction,
                    'force_cpu': force_cpu,
                    'max_workers': actual_max_workers or 'auto'
                }
            }
            
            # Save processing results
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            results_path = f"{output_directory}/{base_name}_processing_results.json"
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(processing_results, f, indent=2, ensure_ascii=False)
            
            # Status message based on what was enabled
            if enable_shot_splitting and enable_frame_extraction:
                print("✅ Video splitting and frame extraction complete!")
            elif enable_shot_splitting:
                print("✅ Video splitting complete (frame extraction disabled)!")
            elif enable_frame_extraction:
                print("✅ Frame extraction complete (video splitting disabled)!")
            else:
                print("✅ Processing complete (only metadata generated)!")
            print(f"📊 Summary:")
            print(f"  - Original video: {video_path}")
            print(f"  - Processed shots: {len(shot_segments)}")
            print(f"  - Shot splitting: {'Enabled' if enable_shot_splitting else 'Disabled'}")
            print(f"  - Frame extraction: {'Enabled' if enable_frame_extraction else 'Disabled'}")
            print(f"  - Output directory: {output_directory}")
            
            return (output_directory, json.dumps(processing_results, indent=2, ensure_ascii=False))
            
        except Exception as e:
            raise Exception(f"Video splitting failed: {e}")
            
        finally:
            # Cleanup temporary file
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                print("🗑️ Temporary video file cleaned up")


class VideoShotJSONLoader:
    """ComfyUI Node for loading shot detection JSON files"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_file_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to JSON file containing shot detection results"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("shot_json",)
    FUNCTION = "load_shot_json"
    CATEGORY = "video/segmentation"
    
    def load_shot_json(self, json_file_path: str) -> Tuple[str]:
        """Load shot detection JSON file and return as string"""
        if not json_file_path.strip():
            raise ValueError("JSON file path cannot be empty")
            
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        
        try:
            # Load JSON file
            with open(json_file_path, 'r', encoding='utf-8') as f:
                shot_data = json.load(f)
            
            # Validate JSON structure
            if not isinstance(shot_data, dict):
                raise ValueError("JSON file must contain a dictionary object")
                
            if "shot_meta_list" not in shot_data:
                raise ValueError("JSON file must contain 'shot_meta_list' key")
                
            if not isinstance(shot_data["shot_meta_list"], list):
                raise ValueError("'shot_meta_list' must be a list")
            
            print(f"📂 Loaded shot detection JSON: {json_file_path}")
            print(f"   - Found {len(shot_data['shot_meta_list'])} shots")
            
            # Return as JSON string (same format as VideoShotDetector)
            return (json.dumps(shot_data, indent=2, ensure_ascii=False),)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file {json_file_path}: {e}")
        except Exception as e:
            raise Exception(f"Failed to load JSON file: {e}")


# ===================== COMFYUI NODE REGISTRATION =====================

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VideoSegmentationModelLoader": VideoSegmentationModelLoader,
    "VideoShotDetector": VideoShotDetector,
    "VideoShotJSONLoader": VideoShotJSONLoader,
    "VideoShotSplitter": VideoShotSplitter
}

# Human-readable display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSegmentationModelLoader": "Video Segmentation Model Loader",
    "VideoShotDetector": "Video Shot Detector", 
    "VideoShotJSONLoader": "Video Shot JSON Loader",
    "VideoShotSplitter": "Video Shot Splitter"
}
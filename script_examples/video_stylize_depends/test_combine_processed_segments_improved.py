#!/usr/bin/env python3
"""
Test script for improved combine_processed_segments function
Tests the simplified workflow using LoadImageSharedMemory batch mode

Test data: 285 stage2_processed_frame_*.jpg images in ComfyUI root directory
- Segment 1: frames 0-145 (146 frames)
- Segment 2: frames 146-284 (139 frames)

ComfyUI server: http://127.0.0.1:8281
"""

import os
import sys
import numpy as np
import time
import glob
from PIL import Image
from typing import Dict, List

# Add ComfyUI root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
comfyui_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels: video_stylize_depends -> script_examples -> ComfyUI
if comfyui_root not in sys.path:
    sys.path.insert(0, comfyui_root)

# Import the improved video combination API
from script_examples.video_stylize_depends.video_combination_api_improved import comfyui_video_combination_improved

def load_test_images(comfyui_root: str) -> List[np.ndarray]:
    """
    Load all stage2_processed_frame_*.jpg images from ComfyUI root directory
    
    Args:
        comfyui_root: Path to ComfyUI root directory
        
    Returns:
        List of image arrays in RGB format
    """
    print("=== Loading Test Images ===")
    
    # Find all stage2_processed_frame_*.jpg files
    pattern = os.path.join(comfyui_root, "stage2_processed_frame_*.jpg")
    image_files = sorted(glob.glob(pattern))
    
    print(f"Found {len(image_files)} test images")
    
    if len(image_files) == 0:
        raise FileNotFoundError(f"No stage2_processed_frame_*.jpg files found in {comfyui_root}")
    
    # Load images
    images = []
    for i, image_path in enumerate(image_files):
        try:
            # Load image and convert to RGB
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.uint8)
            images.append(image_array)
            
            if (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            raise
    
    print(f"✓ Successfully loaded {len(images)} images")
    
    # Print image info
    if images:
        sample_shape = images[0].shape
        print(f"  Image shape: {sample_shape}")
        print(f"  Image dtype: {images[0].dtype}")
        print(f"  Value range: {images[0].min()} - {images[0].max()}")
    
    return images

def create_test_segments(images: List[np.ndarray]) -> Dict[int, List[np.ndarray]]:
    """
    Create test segments from loaded images
    
    Args:
        images: List of image arrays
        
    Returns:
        Dict mapping segment_index to list of frames
    """
    print("\n=== Creating Test Segments ===")
    
    if len(images) != 285:
        print(f"Warning: Expected 285 images, got {len(images)}")
    
    # Split into segments
    # Segment 0: frames 0-145 (146 frames)
    # Segment 1: frames 146-284 (139 frames)
    
    segment_0_frames = images[0:146]  # 0-145 inclusive
    segment_1_frames = images[146:285]  # 146-284 inclusive
    
    processed_segments = {
        0: segment_0_frames,
        1: segment_1_frames
    }
    
    print(f"Segment 0: {len(segment_0_frames)} frames (indices 0-145)")
    print(f"Segment 1: {len(segment_1_frames)} frames (indices 146-284)")
    print(f"Total frames: {len(segment_0_frames) + len(segment_1_frames)}")
    
    return processed_segments

def combine_processed_segments_improved(processed_segments: Dict[int, List[np.ndarray]], 
                                      output_path: str = "output/test_final_video.mp4") -> str:
    """
    Improved version of combine_processed_segments using simplified workflow
    
    Args:
        processed_segments: Dict mapping segment_index to processed_frames  
        output_path: Output path for final combined video
        
    Returns:
        str: Path to final combined video
    """
    print(f"\n=== Improved Video Combination (Simplified Workflow) ===")
    
    # Count total frames without creating new list (memory efficient)
    total_frames = 0
    for segment_index in sorted(processed_segments.keys()):
        frames = processed_segments[segment_index]
        frame_count = len(frames)
        total_frames += frame_count
        print(f"Segment {segment_index}: {frame_count} frames")
    
    print(f"Total frames to combine: {total_frames}")
    
    if total_frames == 0:
        raise ValueError("No frames to combine")
    
    # Create frame generator to stream frames without loading all into memory
    def frame_generator():
        segment_keys = sorted(processed_segments.keys())
        for segment_index in segment_keys:
            frames = processed_segments[segment_index]
            for frame in frames:
                yield frame
    
    # Convert generator to list (for this test, we'll load all frames)
    print("Converting frame generator to list...")
    all_frames = list(frame_generator())
    
    print(f"✓ Collected {len(all_frames)} frames for processing")
    
    # Use improved video combination API
    try:
        result_path = comfyui_video_combination_improved(
            frames=all_frames,
            output_path=output_path,
            frame_rate=30,
            server_address="127.0.0.1:8281"  # Use specified server
        )
        
        print(f"✓ Final video saved: {result_path}")
        return result_path
        
    except Exception as e:
        print(f"✗ Video combination failed: {e}")
        raise

def test_server_connection():
    """Test connection to ComfyUI server"""
    print("=== Testing Server Connection ===")
    
    import urllib.request
    import urllib.error
    
    server_address = "127.0.0.1:8281"
    
    try:
        # Test basic connection
        response = urllib.request.urlopen(f"http://{server_address}/", timeout=5)
        print(f"✓ Server {server_address} is accessible")
        
        # Test object_info endpoint
        response = urllib.request.urlopen(f"http://{server_address}/object_info", timeout=5)
        object_info = response.read()
        print(f"✓ Object info endpoint accessible (response size: {len(object_info)} bytes)")
        
        return True
        
    except urllib.error.URLError as e:
        print(f"✗ Server {server_address} is not accessible: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing server connection: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING IMPROVED COMBINE_PROCESSED_SEGMENTS FUNCTION")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Test server connection first
        if not test_server_connection():
            print("Server connection failed. Please ensure ComfyUI is running on http://127.0.0.1:8281")
            return
        
        # Load test images
        images = load_test_images(comfyui_root)
        
        # Create test segments
        processed_segments = create_test_segments(images)
        
        # Test improved combine function
        output_path = os.path.join(comfyui_root, "output", "test_improved_final_video.mp4")
        result_path = combine_processed_segments_improved(processed_segments, output_path)
        
        # Verify output
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path)
            print(f"\n✓ Test completed successfully!")
            print(f"  Output file: {result_path}")
            print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
        else:
            print(f"\n✗ Test failed: Output file not found at {result_path}")
        
        total_time = time.time() - start_time
        print(f"\nTotal test time: {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

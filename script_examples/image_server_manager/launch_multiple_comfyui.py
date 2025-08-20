#!/usr/bin/env python3
"""
ComfyUI Multi-Instance Launcher

This script launches multiple ComfyUI instances across different GPUs and ports.
It should be run from the conda comfyui environment.

Usage:
    python script_examples/launch_multiple_comfyui.py

GPU allocation:
- GPU 2: ports 8241, 8250, 8251
- GPU 3: port 8266
- GPU 4: port 8261
- GPU 5: ports 8221, 8211
- GPU 6: ports 8202, 8212
- GPU 7: ports 8201, 8203, 8204, 8231
"""

import os
import sys
import time
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comfyui_launcher.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ComfyUI instance configurations
# Format: (gpu_id, port)
COMFYUI_INSTANCES = [
    (7, 8201),
    (6, 8202),
    (5, 8221),
    (7, 8203),
    (7, 8204),
    (5, 8211),
    (6, 8212),
    (7, 8231),
    (2, 8241),
    (2, 8250),
    (2, 8251),
    (4, 8261),
    (3, 8266),
]

# Working directory for ComfyUI - calculate dynamically
# Since we're now in script_examples/image_server_manager/, we need to go up two levels to reach ComfyUI root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMFYUI_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def check_conda_environment():
    """Check if we're running in the correct conda environment."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env != 'comfyui':
        logger.warning(f"Current conda environment: {conda_env}")
        logger.warning("This script should be run in the 'comfyui' conda environment")
        logger.warning("Please run: conda activate comfyui")
        return False
    return True

def check_port_availability(port):
    """Check if a port is available."""
    try:
        result = subprocess.run(
            ['netstat', '-tuln'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        return f":{port}" not in result.stdout
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        logger.warning(f"Could not check port {port} availability")
        return True

def check_gpu_availability(gpu_id):
    """Check if GPU is available using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '-i', str(gpu_id), '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip()
            logger.info(f"GPU {gpu_id}: {gpu_name}")
            return True
        else:
            logger.error(f"GPU {gpu_id} not available")
            return False
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.error(f"Error checking GPU {gpu_id}: {e}")
        return False

def launch_comfyui_instance(gpu_id, port, delay=2):
    """Launch a single ComfyUI instance."""
    logger.info(f"Launching ComfyUI on GPU {gpu_id}, port {port}")
    
    # Check port availability
    if not check_port_availability(port):
        logger.error(f"Port {port} is already in use, skipping...")
        return False
    
    # Check GPU availability
    if not check_gpu_availability(gpu_id):
        logger.error(f"GPU {gpu_id} not available, skipping...")
        return False
    
    try:
        # Construct the command
        cmd = [
            'nohup', 'python', 'main.py', '--port', str(port)
        ]
        
        # Set environment variables
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Launch the process
        process = subprocess.Popen(
            cmd,
            cwd=COMFYUI_DIR,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        logger.info(f"✓ Started ComfyUI instance: GPU {gpu_id}, port {port}, PID {process.pid}")
        
        # Add delay to avoid resource conflicts
        if delay > 0:
            time.sleep(delay)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to launch ComfyUI on GPU {gpu_id}, port {port}: {e}")
        return False

def main():
    """Main function to launch all ComfyUI instances."""
    logger.info("=" * 60)
    logger.info("ComfyUI Multi-Instance Launcher Starting")
    logger.info("=" * 60)
    
    # Check conda environment
    if not check_conda_environment():
        logger.error("Please activate the 'comfyui' conda environment first")
        sys.exit(1)
    
    # Check if we're in the correct directory
    if not os.path.exists('main.py'):
        logger.error(f"main.py not found in current directory: {os.getcwd()}")
        logger.error(f"Please run this script from: {COMFYUI_DIR}")
        sys.exit(1)
    
    # Launch all instances
    successful_launches = 0
    failed_launches = 0
    
    logger.info(f"Launching {len(COMFYUI_INSTANCES)} ComfyUI instances...")
    
    for i, (gpu_id, port) in enumerate(COMFYUI_INSTANCES, 1):
        logger.info(f"[{i}/{len(COMFYUI_INSTANCES)}] Launching instance...")
        
        if launch_comfyui_instance(gpu_id, port):
            successful_launches += 1
        else:
            failed_launches += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info("Launch Summary:")
    logger.info(f"✓ Successful launches: {successful_launches}")
    logger.info(f"✗ Failed launches: {failed_launches}")
    logger.info(f"Total instances: {len(COMFYUI_INSTANCES)}")
    logger.info("=" * 60)
    
    if successful_launches > 0:
        logger.info("ComfyUI instances are starting up...")
        logger.info("Check the logs for any startup errors.")
        logger.info("Use 'python script_examples/stop_all_comfyui.py' to stop all instances.")
    
    if failed_launches > 0:
        logger.warning(f"{failed_launches} instances failed to launch. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

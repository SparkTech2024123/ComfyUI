#!/usr/bin/env python3
"""
ComfyUI Multi-Instance Status Checker

This script checks the status of all ComfyUI instances by checking
if processes are running on the expected ports and testing basic connectivity.

Usage:
    python script_examples/check_comfyui_status.py
"""

import os
import sys
import subprocess
import logging
import requests
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
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

def find_process_by_port(port):
    """Find process ID using a specific port."""
    try:
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip()]
            return pids
        return []
        
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
        return []

def check_http_endpoint(port, timeout=5):
    """Check if ComfyUI HTTP endpoint is responding."""
    try:
        response = requests.get(f'http://localhost:{port}/', timeout=timeout)
        return response.status_code == 200
    except:
        return False

def get_process_info(pid):
    """Get process information."""
    try:
        result = subprocess.run(
            ['ps', '-p', str(pid), '-o', 'pid,ppid,etime,cmd', '--no-headers'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        return None
        
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return None

def check_instance_status(gpu_id, port):
    """Check status of a single ComfyUI instance."""
    status = {
        'gpu_id': gpu_id,
        'port': port,
        'process_running': False,
        'http_responding': False,
        'pids': [],
        'process_info': None
    }
    
    # Check if process is running on port
    pids = find_process_by_port(port)
    if pids:
        status['process_running'] = True
        status['pids'] = pids
        
        # Get process info for the first PID
        if pids:
            status['process_info'] = get_process_info(pids[0])
    
    # Check HTTP endpoint
    status['http_responding'] = check_http_endpoint(port)
    
    return status

def print_status_summary(statuses):
    """Print a summary of all instance statuses."""
    logger.info("=" * 80)
    logger.info("ComfyUI Multi-Instance Status Report")
    logger.info("=" * 80)
    
    # Group by GPU
    gpu_groups = {}
    for status in statuses:
        gpu_id = status['gpu_id']
        if gpu_id not in gpu_groups:
            gpu_groups[gpu_id] = []
        gpu_groups[gpu_id].append(status)
    
    running_count = 0
    responding_count = 0
    total_count = len(statuses)
    
    for gpu_id in sorted(gpu_groups.keys()):
        logger.info(f"\nGPU {gpu_id}:")
        logger.info("-" * 40)
        
        for status in sorted(gpu_groups[gpu_id], key=lambda x: x['port']):
            port = status['port']
            process_status = "✓" if status['process_running'] else "✗"
            http_status = "✓" if status['http_responding'] else "✗"
            
            if status['process_running']:
                running_count += 1
            if status['http_responding']:
                responding_count += 1
            
            logger.info(f"  Port {port:4d}: Process {process_status} | HTTP {http_status}")
            
            if status['pids']:
                logger.info(f"             PIDs: {status['pids']}")
            
            if status['process_info']:
                # Truncate long command lines
                cmd_info = status['process_info']
                if len(cmd_info) > 100:
                    cmd_info = cmd_info[:97] + "..."
                logger.info(f"             Info: {cmd_info}")
    
    # Overall summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary:")
    logger.info(f"  Total instances: {total_count}")
    logger.info(f"  Processes running: {running_count}")
    logger.info(f"  HTTP responding: {responding_count}")
    logger.info(f"  Health: {responding_count}/{total_count} instances responding")
    logger.info("=" * 80)
    
    return running_count, responding_count, total_count

def main():
    """Main function to check all ComfyUI instances."""
    logger.info("Checking ComfyUI instance statuses...")
    
    # Check all instances concurrently
    statuses = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_instance = {
            executor.submit(check_instance_status, gpu_id, port): (gpu_id, port)
            for gpu_id, port in COMFYUI_INSTANCES
        }
        
        for future in as_completed(future_to_instance):
            gpu_id, port = future_to_instance[future]
            try:
                status = future.result()
                statuses.append(status)
            except Exception as e:
                logger.error(f"Error checking GPU {gpu_id}, port {port}: {e}")
                # Add failed status
                statuses.append({
                    'gpu_id': gpu_id,
                    'port': port,
                    'process_running': False,
                    'http_responding': False,
                    'pids': [],
                    'process_info': None
                })
    
    # Print summary
    running_count, responding_count, total_count = print_status_summary(statuses)
    
    # Exit with appropriate code
    if responding_count == total_count:
        logger.info("All instances are healthy!")
        sys.exit(0)
    elif running_count == 0:
        logger.warning("No instances are running!")
        sys.exit(2)
    else:
        logger.warning("Some instances are not responding properly!")
        sys.exit(1)

if __name__ == "__main__":
    main()

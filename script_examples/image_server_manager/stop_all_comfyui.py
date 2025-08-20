#!/usr/bin/env python3
"""
ComfyUI Multi-Instance Stopper

This script stops all running ComfyUI instances by finding processes
running on the specified ports and terminating them.

Usage:
    python script_examples/stop_all_comfyui.py
"""

import os
import sys
import subprocess
import logging
import signal
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comfyui_stopper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ComfyUI ports to check and stop
COMFYUI_PORTS = [8201, 8202, 8221, 8203, 8204, 8211, 8212, 8231, 8241, 8250, 8251, 8261, 8266]

def find_process_by_port(port):
    """Find process ID using a specific port."""
    try:
        # Use lsof to find process using the port
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip()]
            return pids
        return []
        
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Error finding process on port {port}: {e}")
        return []

def find_comfyui_processes():
    """Find all ComfyUI processes by searching for main.py processes."""
    try:
        # Find all python processes running main.py
        result = subprocess.run(
            ['pgrep', '-f', 'python.*main.py'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip()]
            return pids
        return []
        
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Error finding ComfyUI processes: {e}")
        return []

def terminate_process(pid, process_name="process"):
    """Terminate a process by PID."""
    try:
        # Check if process exists
        os.kill(pid, 0)
        
        # Try graceful termination first
        logger.info(f"Sending SIGTERM to {process_name} (PID: {pid})")
        os.kill(pid, signal.SIGTERM)
        
        # Wait a bit for graceful shutdown
        time.sleep(2)
        
        # Check if process is still running
        try:
            os.kill(pid, 0)
            # Process still running, force kill
            logger.warning(f"Process {pid} still running, sending SIGKILL")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
        except ProcessLookupError:
            # Process terminated successfully
            pass
            
        logger.info(f"✓ Terminated {process_name} (PID: {pid})")
        return True
        
    except ProcessLookupError:
        logger.info(f"Process {pid} not found (already terminated)")
        return True
    except PermissionError:
        logger.error(f"✗ Permission denied to terminate process {pid}")
        return False
    except Exception as e:
        logger.error(f"✗ Error terminating process {pid}: {e}")
        return False

def stop_all_comfyui():
    """Stop all ComfyUI instances."""
    logger.info("=" * 60)
    logger.info("ComfyUI Multi-Instance Stopper Starting")
    logger.info("=" * 60)
    
    all_pids = set()
    port_processes = {}
    
    # Find processes by port
    logger.info("Searching for ComfyUI processes by port...")
    for port in COMFYUI_PORTS:
        pids = find_process_by_port(port)
        if pids:
            port_processes[port] = pids
            all_pids.update(pids)
            logger.info(f"Port {port}: Found PIDs {pids}")
        else:
            logger.info(f"Port {port}: No process found")
    
    # Find additional ComfyUI processes by name
    logger.info("Searching for additional ComfyUI processes...")
    comfyui_pids = find_comfyui_processes()
    if comfyui_pids:
        logger.info(f"Found additional ComfyUI processes: {comfyui_pids}")
        all_pids.update(comfyui_pids)
    
    if not all_pids:
        logger.info("No ComfyUI processes found running")
        return
    
    # Terminate all found processes
    logger.info(f"Found {len(all_pids)} ComfyUI processes to terminate")
    successful_stops = 0
    failed_stops = 0
    
    for pid in sorted(all_pids):
        # Find which port this PID is associated with
        associated_ports = [port for port, pids in port_processes.items() if pid in pids]
        port_info = f" (ports: {associated_ports})" if associated_ports else ""
        
        if terminate_process(pid, f"ComfyUI{port_info}"):
            successful_stops += 1
        else:
            failed_stops += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info("Stop Summary:")
    logger.info(f"✓ Successfully stopped: {successful_stops}")
    logger.info(f"✗ Failed to stop: {failed_stops}")
    logger.info(f"Total processes: {len(all_pids)}")
    logger.info("=" * 60)
    
    if failed_stops > 0:
        logger.warning(f"{failed_stops} processes could not be stopped. You may need to stop them manually.")
        sys.exit(1)
    else:
        logger.info("All ComfyUI instances have been stopped successfully.")

def main():
    """Main function."""
    try:
        stop_all_comfyui()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

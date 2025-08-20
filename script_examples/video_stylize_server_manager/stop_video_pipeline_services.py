#!/usr/bin/env python3
# Stop Video Pipeline Services
# Gracefully stops all ComfyUI services for the video style transfer pipeline
#
# Features:
# - Graceful shutdown with SIGTERM first, then SIGKILL if needed
# - PID file cleanup
# - Process verification
# - Detailed status reporting

import os
import sys
import time
import signal
import subprocess
import psutil
from typing import List, Dict, Optional

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from video_pipeline_config import (
    VIDEO_PIPELINE_SERVICES, SERVICE_SHUTDOWN_TIMEOUT,
    get_pid_file_path, get_log_file_path
)

class ServiceStopper:
    """Manages stopping of video pipeline services"""
    
    def __init__(self):
        self.stopped_services = []
        self.failed_services = []
    
    def find_process_by_port(self, port: int) -> List[int]:
        """Find process IDs using a specific port"""
        pids = []
        try:
            # Use lsof to find processes using the port
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = [int(pid) for pid in result.stdout.strip().split('\n')]
                
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        
        return pids
    
    def get_service_pid(self, service) -> Optional[int]:
        """Get PID for a service from PID file or port detection"""
        pid_file_path = get_pid_file_path(service)
        
        # Try to get PID from PID file first
        if os.path.exists(pid_file_path):
            try:
                with open(pid_file_path, 'r') as f:
                    pid = int(f.read().strip())
                
                # Verify the process is still running
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                        return pid
                        
            except (ValueError, OSError, psutil.NoSuchProcess):
                pass
        
        # If PID file method fails, try to find by port
        port_pids = self.find_process_by_port(service.port)
        if port_pids:
            return port_pids[0]  # Return the first PID found
        
        return None
    
    def kill_process_gracefully(self, pid: int, timeout: int = SERVICE_SHUTDOWN_TIMEOUT) -> bool:
        """Kill a process gracefully with timeout"""
        try:
            process = psutil.Process(pid)
            
            # Send SIGTERM first
            print(f"   Sending SIGTERM to process {pid}...")
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=timeout)
                print(f"   ✅ Process {pid} terminated gracefully")
                return True
            except psutil.TimeoutExpired:
                # Force kill if graceful shutdown failed
                print(f"   ⚠️  Process {pid} didn't terminate gracefully, sending SIGKILL...")
                process.kill()
                try:
                    process.wait(timeout=5)
                    print(f"   ✅ Process {pid} killed forcefully")
                    return True
                except psutil.TimeoutExpired:
                    print(f"   ❌ Failed to kill process {pid}")
                    return False
                    
        except psutil.NoSuchProcess:
            print(f"   ✅ Process {pid} already terminated")
            return True
        except Exception as e:
            print(f"   ❌ Error killing process {pid}: {e}")
            return False
    
    def cleanup_service_files(self, service):
        """Clean up PID and log files for a service"""
        pid_file_path = get_pid_file_path(service)
        
        # Remove PID file
        if os.path.exists(pid_file_path):
            try:
                os.remove(pid_file_path)
                print(f"   🧹 Removed PID file: {pid_file_path}")
            except OSError as e:
                print(f"   ⚠️  Failed to remove PID file: {e}")
        
        # Optionally rotate log file (keep it but mark as from previous run)
        log_file_path = get_log_file_path(service)
        if os.path.exists(log_file_path):
            try:
                # Rename log file with timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                rotated_log_path = f"{log_file_path}.{timestamp}"
                os.rename(log_file_path, rotated_log_path)
                print(f"   📄 Rotated log file to: {rotated_log_path}")
            except OSError as e:
                print(f"   ⚠️  Failed to rotate log file: {e}")
    
    def stop_service(self, service) -> bool:
        """Stop a single service"""
        print(f"🛑 Stopping {service.description} (port {service.port})...")
        
        pid = self.get_service_pid(service)
        if pid is None:
            print(f"   ℹ️  No running process found for {service.description}")
            self.cleanup_service_files(service)
            return True
        
        print(f"   Found process PID: {pid}")
        
        # Kill the process
        success = self.kill_process_gracefully(pid)
        
        if success:
            self.stopped_services.append(service)
            print(f"   ✅ Successfully stopped {service.description}")
        else:
            self.failed_services.append(service)
            print(f"   ❌ Failed to stop {service.description}")
        
        # Clean up files regardless of kill success
        self.cleanup_service_files(service)
        
        return success
    
    def stop_all_services(self) -> bool:
        """Stop all video pipeline services"""
        print("🛑 Stopping Video Pipeline Services")
        print("=" * 50)
        
        success_count = 0
        total_services = len(VIDEO_PIPELINE_SERVICES)
        
        # Stop each service
        for service in VIDEO_PIPELINE_SERVICES:
            if self.stop_service(service):
                success_count += 1
            print()  # Empty line between services
        
        # Print summary
        print("=" * 50)
        print(f"🎯 Service Stop Summary:")
        print(f"   Total services: {total_services}")
        print(f"   Successfully stopped: {success_count}")
        print(f"   Failed to stop: {len(self.failed_services)}")
        
        if self.failed_services:
            print(f"\n❌ Failed to stop services:")
            for service in self.failed_services:
                print(f"   - {service.description} (port {service.port})")
                
                # Try to find any remaining processes
                remaining_pids = self.find_process_by_port(service.port)
                if remaining_pids:
                    print(f"     Remaining PIDs on port {service.port}: {remaining_pids}")
        
        if success_count == total_services:
            print(f"\n🎉 All services stopped successfully!")
            return True
        else:
            print(f"\n⚠️  Some services failed to stop. You may need to kill them manually.")
            return False
    
    def stop_service_by_port(self, port: int) -> bool:
        """Stop a specific service by port number"""
        # Find the service configuration
        service = None
        for s in VIDEO_PIPELINE_SERVICES:
            if s.port == port:
                service = s
                break
        
        if service is None:
            print(f"❌ No service configured for port {port}")
            return False
        
        return self.stop_service(service)
    
    def force_kill_all_processes(self):
        """Force kill all processes on video pipeline ports"""
        print("⚠️  Force killing all processes on video pipeline ports...")
        
        killed_any = False
        for service in VIDEO_PIPELINE_SERVICES:
            pids = self.find_process_by_port(service.port)
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"   🔪 Force killed process {pid} on port {service.port}")
                    killed_any = True
                except OSError:
                    pass
        
        if not killed_any:
            print("   ℹ️  No processes found to kill")
        
        # Clean up all PID files
        print("🧹 Cleaning up all PID files...")
        for service in VIDEO_PIPELINE_SERVICES:
            self.cleanup_service_files(service)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stop Video Pipeline Services")
    parser.add_argument("--port", type=int, help="Stop only the service on specified port")
    parser.add_argument("--force", action="store_true", help="Force kill all processes")
    parser.add_argument("--timeout", type=int, default=SERVICE_SHUTDOWN_TIMEOUT,
                        help=f"Shutdown timeout in seconds (default: {SERVICE_SHUTDOWN_TIMEOUT})")
    
    args = parser.parse_args()
    
    stopper = ServiceStopper()
    
    try:
        if args.force:
            stopper.force_kill_all_processes()
            return 0
        elif args.port:
            success = stopper.stop_service_by_port(args.port)
            return 0 if success else 1
        else:
            success = stopper.stop_all_services()
            return 0 if success else 1
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
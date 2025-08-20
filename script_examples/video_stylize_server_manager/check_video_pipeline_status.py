#!/usr/bin/env python3
# Check Video Pipeline Services Status
# Monitors all ComfyUI services for the video style transfer pipeline
#
# Checks:
# - Process status (running/stopped)
# - HTTP connectivity
# - Service health and queue status
# - GPU memory usage
# - Performance metrics

import os
import sys
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import subprocess
import psutil
from typing import Dict, List, Optional, NamedTuple

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from video_pipeline_config import (
    VIDEO_PIPELINE_SERVICES, STATUS_CHECK_TIMEOUT,
    get_pid_file_path, get_log_file_path
)

class ServiceStatus(NamedTuple):
    """Status information for a service"""
    service: object
    process_running: bool
    http_accessible: bool
    queue_running: int
    queue_pending: int
    gpu_memory_used: float
    gpu_memory_total: float
    uptime_seconds: Optional[float]
    error_message: Optional[str]

class ServiceStatusChecker:
    """Checks and reports status of video pipeline services"""
    
    def __init__(self):
        self.service_statuses = []
    
    def check_process_status(self, service) -> tuple[bool, Optional[float]]:
        """Check if service process is running and get uptime"""
        pid_file_path = get_pid_file_path(service)
        
        if not os.path.exists(pid_file_path):
            return False, None
        
        try:
            with open(pid_file_path, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process exists and is running
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    # Calculate uptime
                    create_time = process.create_time()
                    uptime = time.time() - create_time
                    return True, uptime
            
            return False, None
            
        except (ValueError, OSError, psutil.NoSuchProcess):
            return False, None
    
    def check_http_status(self, service) -> tuple[bool, int, int, Optional[str]]:
        """Check HTTP accessibility and get queue information"""
        try:
            # Check basic connectivity
            url = f"http://127.0.0.1:{service.port}/"
            response = urllib.request.urlopen(url, timeout=STATUS_CHECK_TIMEOUT)
            
            if response.getcode() != 200:
                return False, 0, 0, f"HTTP {response.getcode()}"
            
            # Get queue status
            queue_url = f"http://127.0.0.1:{service.port}/queue"
            queue_req = urllib.request.Request(queue_url)
            queue_req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(queue_req, timeout=STATUS_CHECK_TIMEOUT) as queue_response:
                queue_data = json.loads(queue_response.read())
                
                queue_running = len(queue_data.get('queue_running', []))
                queue_pending = len(queue_data.get('queue_pending', []))
                
                return True, queue_running, queue_pending, None
                
        except urllib.error.URLError as e:
            return False, 0, 0, f"Connection failed: {e.reason}"
        except urllib.error.HTTPError as e:
            return False, 0, 0, f"HTTP {e.code}: {e.reason}"
        except json.JSONDecodeError:
            return False, 0, 0, "Invalid JSON response"
        except Exception as e:
            return False, 0, 0, f"Error: {str(e)}"
    
    def check_gpu_memory(self, gpu_id: int) -> tuple[float, float]:
        """Check GPU memory usage"""
        try:
            # Use nvidia-smi to get GPU memory info
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=memory.used,memory.total',
                '--format=csv,noheader,nounits', f'--id={gpu_id}'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                memory_info = result.stdout.strip().split(', ')
                memory_used = float(memory_info[0])  # MB
                memory_total = float(memory_info[1])  # MB
                return memory_used, memory_total
            else:
                return 0.0, 0.0
                
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
            return 0.0, 0.0
    
    def check_service_status(self, service) -> ServiceStatus:
        """Check comprehensive status of a single service"""
        # Check process status
        process_running, uptime = self.check_process_status(service)
        
        # Check HTTP status
        http_accessible, queue_running, queue_pending, error_msg = self.check_http_status(service)
        
        # Check GPU memory
        gpu_memory_used, gpu_memory_total = self.check_gpu_memory(service.gpu)
        
        return ServiceStatus(
            service=service,
            process_running=process_running,
            http_accessible=http_accessible,
            queue_running=queue_running,
            queue_pending=queue_pending,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            uptime_seconds=uptime,
            error_message=error_msg
        )
    
    def check_all_services(self) -> List[ServiceStatus]:
        """Check status of all video pipeline services"""
        print("🔍 Checking Video Pipeline Services Status...")
        print("=" * 70)
        
        self.service_statuses = []
        
        for service in VIDEO_PIPELINE_SERVICES:
            status = self.check_service_status(service)
            self.service_statuses.append(status)
        
        return self.service_statuses
    
    def format_uptime(self, uptime_seconds: Optional[float]) -> str:
        """Format uptime in human-readable format"""
        if uptime_seconds is None:
            return "N/A"
        
        hours, remainder = divmod(int(uptime_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def format_memory(self, used_mb: float, total_mb: float) -> str:
        """Format GPU memory usage"""
        if total_mb == 0:
            return "N/A"
        
        used_gb = used_mb / 1024
        total_gb = total_mb / 1024
        percentage = (used_mb / total_mb) * 100
        
        return f"{used_gb:.1f}GB/{total_gb:.1f}GB ({percentage:.1f}%)"
    
    def print_detailed_status(self):
        """Print detailed status report"""
        if not self.service_statuses:
            print("❌ No status information available")
            return
        
        # Service details
        for status in self.service_statuses:
            service = status.service
            
            # Determine overall status
            if status.process_running and status.http_accessible:
                status_emoji = "✅"
                status_text = "RUNNING"
            elif status.process_running:
                status_emoji = "⚠️"
                status_text = "PROCESS RUNNING (HTTP FAILED)"
            else:
                status_emoji = "❌"
                status_text = "STOPPED"
            
            print(f"{status_emoji} {service.description}")
            print(f"   Port: {service.port} | GPU: {service.gpu} | Status: {status_text}")
            
            if status.process_running:
                print(f"   Uptime: {self.format_uptime(status.uptime_seconds)}")
            
            if status.http_accessible:
                print(f"   Queue: {status.queue_running} running, {status.queue_pending} pending")
            
            if status.gpu_memory_total > 0:
                print(f"   GPU Memory: {self.format_memory(status.gpu_memory_used, status.gpu_memory_total)}")
            
            if status.error_message:
                print(f"   Error: {status.error_message}")
            
            print()  # Empty line between services
    
    def print_summary(self):
        """Print summary statistics"""
        if not self.service_statuses:
            return
        
        total_services = len(self.service_statuses)
        running_services = sum(1 for s in self.service_statuses if s.process_running and s.http_accessible)
        process_only = sum(1 for s in self.service_statuses if s.process_running and not s.http_accessible)
        stopped_services = sum(1 for s in self.service_statuses if not s.process_running)
        
        total_queue_running = sum(s.queue_running for s in self.service_statuses)
        total_queue_pending = sum(s.queue_pending for s in self.service_statuses)
        
        print("=" * 70)
        print("📊 Summary:")
        print(f"   Total Services: {total_services}")
        print(f"   ✅ Fully Operational: {running_services}")
        if process_only > 0:
            print(f"   ⚠️  Process Running (HTTP Issues): {process_only}")
        if stopped_services > 0:
            print(f"   ❌ Stopped: {stopped_services}")
        print(f"   📋 Total Queue: {total_queue_running} running, {total_queue_pending} pending")
        
        # GPU summary
        gpu_usage = {}
        for status in self.service_statuses:
            if status.gpu_memory_total > 0:
                gpu_id = status.service.gpu
                if gpu_id not in gpu_usage:
                    gpu_usage[gpu_id] = {
                        'used': status.gpu_memory_used,
                        'total': status.gpu_memory_total
                    }
        
        if gpu_usage:
            print(f"   🎮 GPU Memory Usage:")
            for gpu_id, memory in gpu_usage.items():
                memory_str = self.format_memory(memory['used'], memory['total'])
                print(f"      GPU {gpu_id}: {memory_str}")
        
        print("=" * 70)
    
    def get_failed_services(self) -> List[object]:
        """Get list of services that are not fully operational"""
        return [
            status.service for status in self.service_statuses
            if not (status.process_running and status.http_accessible)
        ]
    
    def is_pipeline_ready(self) -> bool:
        """Check if the complete pipeline is ready for use"""
        if not self.service_statuses:
            return False
        
        # Check if all services are running and accessible
        all_running = all(
            status.process_running and status.http_accessible
            for status in self.service_statuses
        )
        
        return all_running

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check Video Pipeline Services Status")
    parser.add_argument("--json", action="store_true", help="Output status in JSON format")
    parser.add_argument("--brief", action="store_true", help="Show only summary")
    parser.add_argument("--watch", type=int, metavar="SECONDS", 
                        help="Continuously monitor with specified interval")
    
    args = parser.parse_args()
    
    checker = ServiceStatusChecker()
    
    def check_and_report():
        checker.check_all_services()
        
        if args.json:
            # Output JSON format
            status_data = {
                'timestamp': time.time(),
                'services': [
                    {
                        'description': status.service.description,
                        'port': status.service.port,
                        'gpu': status.service.gpu,
                        'process_running': status.process_running,
                        'http_accessible': status.http_accessible,
                        'queue_running': status.queue_running,
                        'queue_pending': status.queue_pending,
                        'gpu_memory_used_mb': status.gpu_memory_used,
                        'gpu_memory_total_mb': status.gpu_memory_total,
                        'uptime_seconds': status.uptime_seconds,
                        'error_message': status.error_message
                    }
                    for status in checker.service_statuses
                ],
                'pipeline_ready': checker.is_pipeline_ready()
            }
            print(json.dumps(status_data, indent=2))
        else:
            if not args.brief:
                checker.print_detailed_status()
            checker.print_summary()
            
            # Pipeline readiness
            if checker.is_pipeline_ready():
                print("🎉 Video Pipeline is ready for processing!")
            else:
                failed_services = checker.get_failed_services()
                print(f"⚠️  Pipeline not ready. {len(failed_services)} service(s) need attention.")
                return 1
        
        return 0
    
    try:
        if args.watch:
            print(f"👀 Watching services every {args.watch} seconds (Press Ctrl+C to stop)...")
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
                check_and_report()
                time.sleep(args.watch)
        else:
            return check_and_report()
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
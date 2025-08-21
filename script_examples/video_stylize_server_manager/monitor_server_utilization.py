#!/usr/bin/env python3
"""
Server Utilization Monitor
=========================

Real-time monitoring script to verify that all 8 ComfyUI servers are being
utilized during multi-threaded video processing. Run this script in parallel
with video_style_transfer_pipeline.py to validate true concurrency.

Usage:
    python monitor_server_utilization.py

Author: Multi-threading optimization tool
"""

import time
import json
import urllib.request
import urllib.error
from datetime import datetime

# Server Configuration (matches video_style_transfer_pipeline.py)
STYLE_TRANSFER_SERVERS = [
    "127.0.0.1:8281", "127.0.0.1:8282", "127.0.0.1:8283", "127.0.0.1:8284",
    "127.0.0.1:8285", "127.0.0.1:8286", "127.0.0.1:8287", "127.0.0.1:8288"
]

def check_server_utilization():
    """Check utilization across all servers"""
    server_stats = {}
    
    for server_address in STYLE_TRANSFER_SERVERS:
        try:
            # Check queue status
            queue_url = f"http://{server_address}/queue"
            queue_req = urllib.request.Request(queue_url)
            queue_req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(queue_req, timeout=2) as response:
                if response.getcode() == 200:
                    queue_data = json.loads(response.read())
                    queue_running = len(queue_data.get('queue_running', []))
                    queue_pending = len(queue_data.get('queue_pending', []))
                    
                    server_stats[server_address] = {
                        'available': True,
                        'queue_running': queue_running,
                        'queue_pending': queue_pending,
                        'total_load': queue_running + queue_pending
                    }
                else:
                    server_stats[server_address] = {
                        'available': False,
                        'error': f"HTTP {response.getcode()}"
                    }
        except Exception as e:
            server_stats[server_address] = {
                'available': False,
                'error': str(e)
            }
    
    return server_stats

def print_utilization_summary(stats, timestamp):
    """Print server utilization summary"""
    print(f"\n🖥️  ===== SERVER UTILIZATION @ {timestamp} =====")
    
    total_servers = len(STYLE_TRANSFER_SERVERS)
    available_servers = sum(1 for s in stats.values() if s.get('available', False))
    total_running = sum(s.get('queue_running', 0) for s in stats.values() if s.get('available', False))
    total_pending = sum(s.get('queue_pending', 0) for s in stats.values() if s.get('available', False))
    
    print(f"📊 Available servers: {available_servers}/{total_servers}")
    print(f"🔄 Total running tasks: {total_running}")
    print(f"⏳ Total pending tasks: {total_pending}")
    
    # Count servers with active workload
    active_servers = sum(1 for s in stats.values() 
                        if s.get('available', False) and s.get('total_load', 0) > 0)
    print(f"🚀 Active servers: {active_servers}/{available_servers}")
    
    print(f"\n📋 Individual Server Status:")
    for i, (server_address, server_info) in enumerate(stats.items()):
        port = server_address.split(':')[1]
        if server_info.get('available', False):
            running = server_info.get('queue_running', 0)
            pending = server_info.get('queue_pending', 0)
            load_indicator = "🔥" if running > 0 else "💤" if pending > 0 else "🟢"
            print(f"  {load_indicator} Port {port}: running={running}, pending={pending}")
        else:
            error = server_info.get('error', 'unknown error')
            print(f"  ❌ Port {port}: {error}")

def monitor_continuously(interval=5, duration=300):
    """
    Monitor server utilization continuously
    
    Args:
        interval: Seconds between checks
        duration: Total monitoring duration in seconds
    """
    print(f"🔍 Starting continuous server monitoring...")
    print(f"⏱️  Interval: {interval}s, Duration: {duration}s")
    print(f"📊 Monitoring {len(STYLE_TRANSFER_SERVERS)} servers...")
    print(f"📝 Press Ctrl+C to stop monitoring early")
    
    start_time = time.time()
    check_count = 0
    max_concurrent_active = 0
    
    try:
        while time.time() - start_time < duration:
            timestamp = datetime.now().strftime("%H:%M:%S")
            stats = check_server_utilization()
            
            # Track maximum concurrent active servers
            active_count = sum(1 for s in stats.values() 
                             if s.get('available', False) and s.get('total_load', 0) > 0)
            max_concurrent_active = max(max_concurrent_active, active_count)
            
            print_utilization_summary(stats, timestamp)
            
            check_count += 1
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Monitoring stopped by user")
    
    elapsed = time.time() - start_time
    print(f"\n📋 ===== MONITORING SUMMARY =====")
    print(f"⏱️  Total monitoring time: {elapsed:.1f}s")
    print(f"📊 Total checks performed: {check_count}")
    print(f"🚀 Maximum concurrent active servers: {max_concurrent_active}/{len(STYLE_TRANSFER_SERVERS)}")
    
    if max_concurrent_active == len(STYLE_TRANSFER_SERVERS):
        print(f"✅ SUCCESS: All {len(STYLE_TRANSFER_SERVERS)} servers were utilized concurrently!")
    elif max_concurrent_active >= 6:
        print(f"🟡 GOOD: {max_concurrent_active} servers utilized (good parallelism)")
    elif max_concurrent_active >= 3:
        print(f"🟠 MODERATE: {max_concurrent_active} servers utilized (some parallelism)")
    else:
        print(f"🔴 LIMITED: Only {max_concurrent_active} servers utilized (limited parallelism)")

def main():
    """Main monitoring function"""
    print("🖥️  ComfyUI Server Utilization Monitor")
    print("=====================================")
    print(f"Monitoring {len(STYLE_TRANSFER_SERVERS)} servers for utilization patterns")
    print()
    
    # Initial status check
    print("📊 Initial server status check...")
    initial_stats = check_server_utilization()
    initial_timestamp = datetime.now().strftime("%H:%M:%S")
    print_utilization_summary(initial_stats, initial_timestamp)
    
    # Ask user for monitoring duration
    try:
        print(f"\n🔍 Starting continuous monitoring...")
        print(f"💡 Recommended: Run this while video_style_transfer_pipeline.py is executing")
        print(f"⏱️  Default monitoring: 300 seconds (5 minutes)")
        
        # Start continuous monitoring
        monitor_continuously(interval=3, duration=300)
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")

if __name__ == "__main__":
    main()
# Video Pipeline Service Configuration
# Centralized configuration for video style transfer pipeline services
#
# Service Layout:
# - Video Segmentation: 1 service (port 8261, GPU 4)
# - Style Transfer: 8 services (ports 8281-8288, GPUs 0,1,2,3,4,5,6,7)
# - Total: 9 services across 8 different GPUs (GPU 4 shared between segmentation and style transfer)

import os
from typing import Dict, List, NamedTuple

class ServiceConfig(NamedTuple):
    """Configuration for a single ComfyUI service"""
    port: int
    gpu: int
    description: str
    service_type: str

# Video Pipeline Service Configuration
VIDEO_PIPELINE_SERVICES = [
    # Video Segmentation Service
    ServiceConfig(
        port=8261,
        gpu=4,
        description="Video Segmentation Service",
        service_type="segmentation"
    ),
    
    # Style Transfer Services
    ServiceConfig(
        port=8281,
        gpu=0,
        description="Style Transfer Service 1",
        service_type="style_transfer"
    ),
    ServiceConfig(
        port=8282,
        gpu=1,
        description="Style Transfer Service 2",
        service_type="style_transfer"
    ),
    ServiceConfig(
        port=8283,
        gpu=2,
        description="Style Transfer Service 3",
        service_type="style_transfer"
    ),
    ServiceConfig(
        port=8284,
        gpu=3,
        description="Style Transfer Service 4",
        service_type="style_transfer"
    ),
    ServiceConfig(
        port=8285,
        gpu=5,
        description="Style Transfer Service 5",
        service_type="style_transfer"
    ),
    ServiceConfig(
        port=8286,
        gpu=6,
        description="Style Transfer Service 6",
        service_type="style_transfer"
    ),
    ServiceConfig(
        port=8287,
        gpu=7,
        description="Style Transfer Service 7",
        service_type="style_transfer"
    ),
    ServiceConfig(
        port=8288,
        gpu=4,
        description="Style Transfer Service 8",
        service_type="style_transfer"
    ),
]

# Directory Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Since we're now in script_examples/video_stylize_server_manager/, we need to go up two levels to reach ComfyUI root
COMFYUI_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
LOGS_DIR = os.path.join(COMFYUI_ROOT, "logs", "video_pipeline")
PIDS_DIR = os.path.join(LOGS_DIR, "pids")

# Service Management Configuration
SERVICE_STARTUP_TIMEOUT = 60  # seconds
SERVICE_SHUTDOWN_TIMEOUT = 30  # seconds
STATUS_CHECK_TIMEOUT = 5  # seconds
STARTUP_WAIT_INTERVAL = 5  # seconds between startup checks

def get_service_by_port(port: int) -> ServiceConfig:
    """Get service configuration by port number"""
    for service in VIDEO_PIPELINE_SERVICES:
        if service.port == port:
            return service
    raise ValueError(f"No service configured for port {port}")

def get_services_by_type(service_type: str) -> List[ServiceConfig]:
    """Get all services of a specific type"""
    return [service for service in VIDEO_PIPELINE_SERVICES if service.service_type == service_type]

def get_segmentation_service() -> ServiceConfig:
    """Get the video segmentation service configuration"""
    services = get_services_by_type("segmentation")
    if not services:
        raise ValueError("No segmentation service configured")
    return services[0]

def get_style_transfer_services() -> List[ServiceConfig]:
    """Get all style transfer service configurations"""
    return get_services_by_type("style_transfer")

def get_log_file_path(service: ServiceConfig) -> str:
    """Get log file path for a service"""
    return os.path.join(LOGS_DIR, f"service_{service.port}.log")

def get_pid_file_path(service: ServiceConfig) -> str:
    """Get PID file path for a service"""
    return os.path.join(PIDS_DIR, f"service_{service.port}.pid")

def ensure_directories():
    """Ensure required directories exist"""
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PIDS_DIR, exist_ok=True)

def get_service_command(service: ServiceConfig) -> List[str]:
    """Get the command to start a service"""
    return [
        "python", "main.py",
        "--port", str(service.port)
    ]

def get_service_env(service: ServiceConfig) -> Dict[str, str]:
    """Get environment variables for a service"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(service.gpu)
    return env

def print_service_summary():
    """Print a summary of all configured services"""
    print(f"Video Pipeline Service Configuration:")
    print(f"=" * 50)
    
    # Segmentation service
    seg_service = get_segmentation_service()
    print(f"📹 {seg_service.description}")
    print(f"   Port: {seg_service.port}, GPU: {seg_service.gpu}")
    
    # Style transfer services
    style_services = get_style_transfer_services()
    print(f"\n🎨 Style Transfer Services ({len(style_services)} services):")
    for service in style_services:
        print(f"   Port: {service.port}, GPU: {service.gpu} - {service.description}")
    
    print(f"\n📊 Total: {len(VIDEO_PIPELINE_SERVICES)} services across {len(set(s.gpu for s in VIDEO_PIPELINE_SERVICES))} GPUs")
    print(f"=" * 50)

if __name__ == "__main__":
    print_service_summary()
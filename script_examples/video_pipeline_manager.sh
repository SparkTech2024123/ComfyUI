#!/bin/bash
#
# Video Pipeline Manager Shell Script
#
# Manages ComfyUI services for the video style transfer pipeline.
# Usage:
#     ./script_examples/video_pipeline_manager.sh [start|stop|status|restart|watch]
#
# Services managed:
# - Video Segmentation: port 8261, GPU 4
# - Style Transfer: ports 8281-8288, GPUs 0,1,2,3,4,5,6,7
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_DIR="$(dirname "$SCRIPT_DIR")"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}[VIDEO PIPELINE]${NC} $1"
}

# Function to check if we're in the right directory
check_directory() {
    if [[ ! -f "$COMFYUI_DIR/main.py" ]]; then
        print_error "main.py not found. Please run this script from the ComfyUI directory."
        exit 1
    fi
}

# Function to check conda environment
check_conda_env() {
    if [[ -n "$CONDA_DEFAULT_ENV" && "$CONDA_DEFAULT_ENV" != "comfyui" ]]; then
        print_warning "Current conda environment: $CONDA_DEFAULT_ENV"
        print_warning "This script should be run in the 'comfyui' conda environment"
        print_warning "Please run: conda activate comfyui"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to check Python script dependencies
check_python_scripts() {
    local required_scripts=(
        "video_stylize_server_manager/video_pipeline_config.py"
        "video_stylize_server_manager/launch_video_pipeline_services.py"
        "video_stylize_server_manager/check_video_pipeline_status.py"
        "video_stylize_server_manager/stop_video_pipeline_services.py"
    )

    for script in "${required_scripts[@]}"; do
        if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
            print_error "Required script not found: $script"
            exit 1
        fi
    done
}

# Function to start video pipeline services
start_pipeline() {
    print_header "Starting Video Pipeline Services..."
    print_info "This will start 9 ComfyUI services:"
    print_info "  - Video Segmentation: port 8261, GPU 4"  
    print_info "  - Style Transfer: ports 8281-8288, GPUs 0,1,2,3,4,5,6,7"
    echo
    
    cd "$COMFYUI_DIR"
    python script_examples/video_stylize_server_manager/launch_video_pipeline_services.py
    
    if [[ $? -eq 0 ]]; then
        print_success "Video Pipeline services started successfully"
        print_info "Waiting for services to initialize..."
        sleep 10
        print_info "Checking service status..."
        python script_examples/video_stylize_server_manager/check_video_pipeline_status.py --brief
    else
        print_error "Failed to start Video Pipeline services"
        exit 1
    fi
}

# Function to stop video pipeline services
stop_pipeline() {
    print_header "Stopping Video Pipeline Services..."
    cd "$COMFYUI_DIR"
    python script_examples/video_stylize_server_manager/stop_video_pipeline_services.py
    
    if [[ $? -eq 0 ]]; then
        print_success "Video Pipeline services stopped successfully"
    else
        print_error "Failed to stop some Video Pipeline services"
        print_warning "You may need to use --force flag or manually kill processes"
        exit 1
    fi
}

# Function to check status
check_status() {
    print_header "Checking Video Pipeline Services Status..."
    cd "$COMFYUI_DIR"
    python script_examples/video_stylize_server_manager/check_video_pipeline_status.py
}

# Function to watch status continuously
watch_status() {
    print_header "Monitoring Video Pipeline Services (Press Ctrl+C to stop)..."
    cd "$COMFYUI_DIR"
    python script_examples/video_stylize_server_manager/check_video_pipeline_status.py --watch 10
}

# Function to monitor server utilization
monitor_utilization() {
    print_header "Starting Video Pipeline Server Utilization Monitor..."
    print_info "This will monitor real-time utilization of all 8 style transfer servers"
    print_info "Recommended: Run this while video_style_transfer_pipeline.py is executing"
    echo
    cd "$COMFYUI_DIR"
    python script_examples/video_stylize_server_manager/monitor_server_utilization.py
}

# Function to restart services
restart_pipeline() {
    print_header "Restarting Video Pipeline Services..."
    
    # Stop services first
    cd "$COMFYUI_DIR"
    python script_examples/video_stylize_server_manager/stop_video_pipeline_services.py
    
    if [[ $? -eq 0 ]]; then
        print_info "Services stopped, waiting 5 seconds before restart..."
        sleep 5
        
        # Start services
        python script_examples/video_stylize_server_manager/launch_video_pipeline_services.py
        
        if [[ $? -eq 0 ]]; then
            print_success "Video Pipeline services restarted successfully"
            sleep 10
            python script_examples/video_stylize_server_manager/check_video_pipeline_status.py --brief
        else
            print_error "Failed to restart Video Pipeline services"
            exit 1
        fi
    else
        print_error "Failed to stop services for restart"
        exit 1
    fi
}

# Function to force stop all services
force_stop() {
    print_header "Force stopping all Video Pipeline Services..."
    print_warning "This will forcefully kill all processes on pipeline ports"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$COMFYUI_DIR"
        python script_examples/video_stylize_server_manager/stop_video_pipeline_services.py --force
        print_success "Force stop completed"
    else
        print_info "Force stop cancelled"
    fi
}

# Function to show detailed usage
show_usage() {
    echo "Usage: $0 [start|stop|status|restart|watch|monitor|force-stop]"
    echo ""
    echo "Commands:"
    echo "  start      - Launch all video pipeline services"
    echo "  stop       - Stop all video pipeline services"
    echo "  status     - Check detailed status of all services"
    echo "  restart    - Stop and start all services"
    echo "  watch      - Continuously monitor service status"
    echo "  monitor    - Monitor real-time server utilization"
    echo "  force-stop - Force kill all pipeline processes"
    echo ""
    echo "Video Pipeline Service Configuration:"
    echo "  📹 Video Segmentation Service:"
    echo "     Port: 8261, GPU: 4"
    echo ""
    echo "  🎨 Style Transfer Services:"
    echo "     Port: 8281, GPU: 0  |  Port: 8282, GPU: 1"
    echo "     Port: 8283, GPU: 2  |  Port: 8284, GPU: 3"
    echo "     Port: 8285, GPU: 5  |  Port: 8286, GPU: 6"
    echo "     Port: 8287, GPU: 7  |  Port: 8288, GPU: 4"
    echo ""
    echo "Total: 9 services across 8 different GPUs"
    echo ""
    echo "Examples:"
    echo "  $0 start           # Start all services"
    echo "  $0 status          # Check status"
    echo "  $0 watch           # Monitor continuously"
    echo "  $0 monitor         # Monitor server utilization"
    echo "  $0 restart         # Restart all services"
}

# Function to show service configuration
show_config() {
    print_header "Video Pipeline Service Configuration"
    cd "$COMFYUI_DIR"
    python script_examples/video_stylize_server_manager/video_pipeline_config.py
}

# Main script logic
main() {
    # Print header
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║              Video Style Transfer Pipeline Manager           ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
    
    # Check if we're in the right directory
    check_directory
    
    # Check for required Python scripts
    check_python_scripts
    
    # Check conda environment
    check_conda_env
    
    # Parse command line arguments
    case "${1:-}" in
        start)
            start_pipeline
            ;;
        stop)
            stop_pipeline
            ;;
        status)
            check_status
            ;;
        watch)
            watch_status
            ;;
        monitor)
            monitor_utilization
            ;;
        restart)
            restart_pipeline
            ;;
        force-stop)
            force_stop
            ;;
        config)
            show_config
            ;;
        "")
            print_error "No command specified"
            show_usage
            exit 1
            ;;
        -h|--help|help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Set up signal handlers for cleanup
trap 'echo -e "\n${YELLOW}[INFO]${NC} Interrupted by user"; exit 130' INT
trap 'echo -e "\n${YELLOW}[INFO]${NC} Terminated"; exit 143' TERM

# Run main function with all arguments
main "$@"
#!/bin/bash
#
# ComfyUI Multi-Instance Manager Shell Script
#
# This is a convenience wrapper for the Python scripts.
# Usage:
#     ./script_examples/comfyui_manager.sh [start|stop|status|restart]
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Function to check if we're in the right directory
check_directory() {
    if [[ ! -f "$COMFYUI_DIR/main.py" ]]; then
        print_error "main.py not found. Please run this script from the ComfyUI directory."
        exit 1
    fi
}

# Function to check conda environment
check_conda_env() {
    if [[ "$CONDA_DEFAULT_ENV" != "comfyui" ]]; then
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

# Function to start ComfyUI instances
start_comfyui() {
    print_info "Starting ComfyUI instances..."
    cd "$COMFYUI_DIR"
    python script_examples/image_server_manager/launch_multiple_comfyui.py
    if [[ $? -eq 0 ]]; then
        print_success "ComfyUI instances started successfully"
        print_info "Waiting 30 seconds for startup..."
        sleep 30
        print_info "Checking status..."
        python script_examples/image_server_manager/check_comfyui_status.py
    else
        print_error "Failed to start ComfyUI instances"
        exit 1
    fi
}

# Function to stop ComfyUI instances
stop_comfyui() {
    print_info "Stopping ComfyUI instances..."
    cd "$COMFYUI_DIR"
    python script_examples/image_server_manager/stop_all_comfyui.py
    if [[ $? -eq 0 ]]; then
        print_success "ComfyUI instances stopped successfully"
    else
        print_error "Failed to stop some ComfyUI instances"
        exit 1
    fi
}

# Function to check status
check_status() {
    print_info "Checking ComfyUI instance status..."
    cd "$COMFYUI_DIR"
    python script_examples/image_server_manager/check_comfyui_status.py
}

# Function to restart ComfyUI instances
restart_comfyui() {
    print_info "Restarting ComfyUI instances..."
    stop_comfyui
    sleep 5
    start_comfyui
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [start|stop|status|restart]"
    echo ""
    echo "Commands:"
    echo "  start   - Launch all ComfyUI instances"
    echo "  stop    - Stop all ComfyUI instances"
    echo "  status  - Check status of all instances"
    echo "  restart - Stop and start all instances"
    echo ""
    echo "GPU allocation:"
    echo "  GPU 2: ports 8241, 8250, 8251"
    echo "  GPU 3: port 8266"
    echo "  GPU 4: port 8261"
    echo "  GPU 5: ports 8221, 8211"
    echo "  GPU 6: ports 8202, 8212"
    echo "  GPU 7: ports 8201, 8203, 8204, 8231"
}

# Main script logic
main() {
    # Check if we're in the right directory
    check_directory
    
    # Check conda environment
    check_conda_env
    
    # Parse command line arguments
    case "${1:-}" in
        start)
            start_comfyui
            ;;
        stop)
            stop_comfyui
            ;;
        status)
            check_status
            ;;
        restart)
            restart_comfyui
            ;;
        "")
            print_error "No command specified"
            show_usage
            exit 1
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"

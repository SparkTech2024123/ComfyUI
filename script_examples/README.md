# ComfyUI Multi-Instance Management Scripts

This directory contains Python scripts to manage multiple ComfyUI instances across different GPUs and ports.

## Scripts Overview

### 1. `launch_multiple_comfyui.py`
Launches multiple ComfyUI instances as background processes with proper GPU allocation.

### 2. `stop_all_comfyui.py`
Stops all running ComfyUI instances by finding and terminating processes on the configured ports.

### 3. `check_comfyui_status.py`
Checks the status of all ComfyUI instances, showing which are running and responding to HTTP requests.

## GPU and Port Allocation

The scripts manage the following ComfyUI instances:

| GPU | Ports |
|-----|-------|
| GPU 2 | 8241, 8250, 8251 |
| GPU 3 | 8266 |
| GPU 4 | 8261 |
| GPU 5 | 8221, 8211 |
| GPU 6 | 8202, 8212 |
| GPU 7 | 8201, 8203, 8204, 8231 |

**Total: 13 ComfyUI instances across 6 GPUs**

## Prerequisites

1. **Conda Environment**: Make sure you're in the `comfyui` conda environment:
   ```bash
   conda activate comfyui
   ```

2. **Working Directory**: Run all scripts from the ComfyUI root directory:
   ```bash
   cd /data/projs/WebServer/distributed-server-node/submodules/VisualForge/ComfyUI
   ```

3. **Dependencies**: Ensure the following are installed:
   - `requests` (for HTTP status checking)
   - `lsof` (for port checking)
   - `nvidia-smi` (for GPU checking)

## Usage

### Launch All Instances
```bash
python script_examples/launch_multiple_comfyui.py
```

This will:
- Check conda environment and GPU availability
- Launch all 13 ComfyUI instances with proper GPU allocation
- Add delays between launches to avoid resource conflicts
- Log all activities to `comfyui_launcher.log`

### Check Status
```bash
python script_examples/check_comfyui_status.py
```

This will:
- Check which processes are running on each port
- Test HTTP connectivity to each instance
- Display a comprehensive status report grouped by GPU

### Stop All Instances
```bash
python script_examples/stop_all_comfyui.py
```

This will:
- Find all ComfyUI processes by port and process name
- Gracefully terminate processes (SIGTERM first, then SIGKILL if needed)
- Log all activities to `comfyui_stopper.log`

## Example Workflow

```bash
# 1. Activate the conda environment
conda activate comfyui

# 2. Navigate to ComfyUI directory
cd /data/projs/WebServer/distributed-server-node/submodules/VisualForge/ComfyUI

# 3. Launch all instances
python script_examples/launch_multiple_comfyui.py

# 4. Check status (wait a minute for startup)
sleep 60
python script_examples/check_comfyui_status.py

# 5. When done, stop all instances
python script_examples/stop_all_comfyui.py
```

## Log Files

- `comfyui_launcher.log` - Launch activities and errors
- `comfyui_stopper.log` - Stop activities and errors
- Status checker outputs to console only

## Troubleshooting

### Common Issues

1. **Port already in use**: The launcher will skip instances if ports are already occupied
2. **GPU not available**: The launcher will skip instances if GPUs are not accessible
3. **Permission errors**: Make sure you have permission to kill processes
4. **Conda environment**: Ensure you're in the `comfyui` environment

### Manual Process Management

If the scripts fail, you can manually check and kill processes:

```bash
# Check what's running on specific ports
lsof -i :8250

# Kill specific process
kill -TERM <PID>

# Force kill if needed
kill -KILL <PID>

# Find all ComfyUI processes
pgrep -f "python.*main.py"
```

### GPU Monitoring

Check GPU usage:
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

## Customization

To modify the GPU/port allocation, edit the `COMFYUI_INSTANCES` list in `launch_multiple_comfyui.py`:

```python
COMFYUI_INSTANCES = [
    (gpu_id, port),
    # Add or modify entries as needed
]
```

Remember to update the same list in the other scripts for consistency.

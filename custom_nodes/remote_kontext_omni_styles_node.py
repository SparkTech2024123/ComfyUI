"""
Remote Kontext Omni Styles Node for ComfyUI

Uses remote ComfyUI service for Kontext Omni Style Transfer processing, avoiding local model loading
Supports load balancing, shared memory transfer and complete style transfer parameter control
"""

import torch
import numpy as np
import time
import json
import uuid
import urllib.request
import urllib.parse
import urllib.error
import websocket
from multiprocessing import shared_memory
from PIL import Image
import io
import cv2
import os
import sys
import tempfile
import random

class RemoteKontextOmniStylesNode:
    """
    Remote Kontext Omni Styles Node
    
    Uses remote ComfyUI servers for style transfer processing, supporting:
    - Automatic load balancing
    - Shared memory high-speed transfer
    - Complete style transfer parameter control
    - API key authentication
    - Loads workflow from JSON file (more stable)
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "style_description": ("STRING", {
                    "default": "Origami",
                    "tooltip": "Style description for the transformation"
                }),
                "server_addresses": ("STRING", {
                    "default": "127.0.0.1:8261,127.0.0.1:8262,127.0.0.1:8263,127.0.0.1:8264,127.0.0.1:8265",
                    "tooltip": "ComfyUI server address list, separated by commas, supports load balancing"
                }),
                "use_shared_memory_output": ("BOOLEAN", {"default": True, "tooltip": "Whether to use shared memory output (40-50x performance boost)"}),
            },
            "optional": {
                "comfy_api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API key for ComfyUI authentication"
                }),
                "seed": ("INT", {
                    "default": 81018347348975, "min": -1, "max": 2**63-1,
                    "tooltip": "Random seed, -1 for automatic"
                }),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 100,
                    "tooltip": "Sampling steps"
                }),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "CFG scale"
                }),
                "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "ddim", "uni_pc", "uni_pc_bh2"], {
                    "default": "euler",
                    "tooltip": "Sampler name"
                }),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"], {
                    "default": "simple",
                    "tooltip": "Scheduler type"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise strength"
                }),
                "guidance": ("FLOAT", {
                    "default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Guidance strength"
                }),
                "max_shift": ("FLOAT", {
                    "default": 1.15, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Max shift for ModelSamplingFlux"
                }),
                "base_shift": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Base shift for ModelSamplingFlux"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("styled_image",)
    FUNCTION = "apply_style_transfer"
    CATEGORY = "api/image/style"

    def apply_style_transfer(self, image, style_description, server_addresses, use_shared_memory_output=True,
                            comfy_api_key="", seed=81018347348975, steps=20, cfg=1.0, 
                            sampler_name="euler", scheduler="simple", denoise=1.0, guidance=2.5,
                            max_shift=1.15, base_shift=0.5):
        """
        Execute Kontext Omni Style Transfer
        """
        start_time = time.time()
        
        # Parse server address list
        server_list = [addr.strip() for addr in server_addresses.split(',') if addr.strip()]
        if not server_list:
            raise ValueError("At least one server address must be provided")
        
        print(f"Kontext Omni Style Transfer - Style: {style_description}")
        print(f"Server addresses: {server_list}")
        print(f"Style parameters: steps={steps}, cfg={cfg}, guidance={guidance}, denoise={denoise}")
        
        # Convert image format (ComfyUI tensor -> numpy RGB)
        if len(image.shape) == 4:
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        
        print(f"Input image shape: {image_np.shape}")
        
        # Handle random seed - ensure not passing None value
        if seed == -1:
            seed_to_use = random.randint(0, 2**31-1)  # Generate random seed instead of None
        else:
            seed_to_use = seed
        
        # Create independent client instance
        client_id = str(uuid.uuid4())
        
        try:
            # Select best server
            selected_server = self._select_best_server(server_list)
            if not selected_server:
                raise RuntimeError("No available ComfyUI servers found")
            
            # Execute style transfer processing
            result_np = self._process_with_shared_memory(
                image_np, selected_server, client_id,
                style_description=style_description,
                comfy_api_key=comfy_api_key if comfy_api_key else None,
                seed=seed_to_use, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler, denoise=denoise,
                guidance=guidance, max_shift=max_shift, base_shift=base_shift,
                use_shared_memory_output=use_shared_memory_output
            )
            
            # Convert back to ComfyUI tensor format
            # Input is RGBA (h, w, 4), need to convert to RGB for ComfyUI
            if result_np.shape[2] == 4:
                # Remove alpha channel for ComfyUI compatibility
                result_np = result_np[:, :, :3]
            
            result_tensor = torch.from_numpy(result_np.astype(np.float32) / 255.0).unsqueeze(0)
            
            total_time = time.time() - start_time
            print(f"✓ Kontext Omni Style Transfer completed in {total_time:.3f}s")
            print(f"  Output shape: {result_tensor.shape}")
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"✗ Kontext Omni Style Transfer failed: {e}")
            raise

    def _select_best_server(self, server_list):
        """Select the best ComfyUI server"""
        print("=== Checking ComfyUI servers status ===")
        available_servers = []
        
        for server in server_list:
            status = self._check_server_status(server)
            if status['available']:
                available_servers.append(status)
                print(f"✓ {server}: load={status['total_load']}, vram_free={status['vram_free']/(1024**3):.1f}GB")
            else:
                print(f"✗ {server}: {status.get('error', 'unavailable')}")
        
        if not available_servers:
            print("No available ComfyUI servers found!")
            return None
        
        # Select server with lowest load, if load is same then select lowest VRAM usage
        best_server = min(available_servers, key=lambda x: (x['total_load'], x['vram_usage_ratio']))
        
        print(f"Selected server: {best_server['server_address']} (load={best_server['total_load']})")
        print("=" * 50)
        
        return best_server['server_address']

    def _check_server_status(self, server_address):
        """Check ComfyUI server status"""
        try:
            # Check queue status
            queue_url = f"http://{server_address}/queue"
            queue_req = urllib.request.Request(queue_url)
            queue_req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(queue_req, timeout=3) as response:
                queue_data = json.loads(response.read())
            
            # Check system status
            stats_url = f"http://{server_address}/system_stats"
            stats_req = urllib.request.Request(stats_url)
            
            with urllib.request.urlopen(stats_req, timeout=3) as response:
                system_data = json.loads(response.read())
            
            # Calculate server load
            queue_running = len(queue_data.get('queue_running', []))
            queue_pending = len(queue_data.get('queue_pending', []))
            total_load = queue_running + queue_pending
            
            # Get VRAM usage
            vram_free = 0
            vram_total = 0
            if 'devices' in system_data and len(system_data['devices']) > 0:
                device = system_data['devices'][0]
                vram_free = device.get('vram_free', 0)
                vram_total = device.get('vram_total', 1)
            
            vram_usage_ratio = 1 - (vram_free / vram_total) if vram_total > 0 else 1
            
            return {
                'server_address': server_address,
                'queue_running': queue_running,
                'queue_pending': queue_pending,
                'total_load': total_load,
                'vram_free': vram_free,
                'vram_total': vram_total,
                'vram_usage_ratio': vram_usage_ratio,
                'available': True
            }
            
        except Exception as e:
            return {
                'server_address': server_address,
                'available': False,
                'error': str(e)
            }

    def _process_with_shared_memory(self, image_array, server_address, client_id,
                                   use_shared_memory_output=True, **kwargs):
        """Process image using shared memory"""
        shm_obj = None
        
        try:
            # 1. Create shared memory
            shm_name, shape, dtype, shm_obj = self._numpy_to_shared_memory(image_array)
            shm_data = (shm_name, shape, dtype)
            
            # 2. Load and modify workflow from file
            workflow = self._load_and_modify_workflow(shm_data, **kwargs)
            
            # 3. Execute workflow
            if use_shared_memory_output:
                result_array = self._execute_workflow_shared_output(workflow, server_address, client_id, **kwargs)
            else:
                result_array = self._execute_workflow_websocket_output(workflow, server_address, client_id, **kwargs)
            
            return result_array
            
        finally:
            # Clean up shared memory
            if shm_obj:
                try:
                    shm_obj.close()
                    shm_obj.unlink()
                except:
                    pass

    def _numpy_to_shared_memory(self, image_array):
        """Store numpy array to shared memory"""
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError(f"Expected image array shape (h, w, 3), got {image_array.shape}")
        
        # Generate shared memory name
        shm_name = f"comfyui_img_{uuid.uuid4().hex[:8]}"
        
        # Ensure array is contiguous
        if not image_array.flags['C_CONTIGUOUS']:
            image_array = np.ascontiguousarray(image_array)
        
        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, size=image_array.nbytes, name=shm_name)
        
        # Copy data to shared memory
        shm_array = np.ndarray(image_array.shape, dtype=image_array.dtype, buffer=shm.buf)
        shm_array[:] = image_array[:]
        
        return shm_name, list(image_array.shape), str(image_array.dtype), shm

    def _load_and_modify_workflow(self, shm_data, **kwargs):
        """Load workflow from file and modify parameters"""
        try:
            # Get workflow file path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            workflow_path = os.path.join(current_dir, "..", "user", "default", "workflows", "A-kontext-omni-styles-api.json")
            
            # If relative path doesn't exist, try absolute path
            if not os.path.exists(workflow_path):
                comfyui_root = os.path.dirname(current_dir)
                workflow_path = os.path.join(comfyui_root, "user", "default", "workflows", "A-kontext-omni-styles-api.json")
            
            # Check if file exists again
            if not os.path.exists(workflow_path):
                raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
            
            # Load original workflow
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            
            print(f"✓ Loaded workflow from: {workflow_path}")
            
            # Modify workflow
            modified_workflow = self._modify_workflow_parameters(workflow, shm_data, **kwargs)
            
            return modified_workflow
            
        except Exception as e:
            print(f"Failed to load workflow from file: {e}")
            raise

    def _modify_workflow_parameters(self, workflow, shm_data, **kwargs):
        """Modify workflow parameters"""
        # Create workflow copy
        modified_workflow = workflow.copy()
        
        # 1. Replace LoadImage node with LoadImageSharedMemory node (node "25")
        if "25" in modified_workflow:
            print(f"Replacing LoadImage node with LoadImageSharedMemory node")
            modified_workflow["25"] = {
                "inputs": {
                    "shm_name": shm_data[0],
                    "shape": json.dumps(shm_data[1]),
                    "dtype": shm_data[2],
                    "convert_bgr_to_rgb": False  # PIL input is already RGB format, no conversion needed
                },
                "class_type": "LoadImageSharedMemory",
                "_meta": {"title": "Load Image (Shared Memory)"}
            }
            print(f"Updated to LoadImageSharedMemory node with shm_name: {shm_data[0]}")
        
        # 2. Update style description (node "26")
        if "26" in modified_workflow:
            style_description = kwargs.get("style_description", "Origami")
            modified_workflow["26"]["inputs"]["text"] = style_description
            print(f"Updated style description: {style_description}")
        
        # 3. Update KSampler parameters (node "20")
        if "20" in modified_workflow:
            if "seed" in kwargs:
                modified_workflow["20"]["inputs"]["seed"] = kwargs["seed"]
                print(f"Updated seed: {kwargs['seed']}")
            if "steps" in kwargs:
                modified_workflow["20"]["inputs"]["steps"] = kwargs["steps"]
                print(f"Updated steps: {kwargs['steps']}")
            if "cfg" in kwargs:
                modified_workflow["20"]["inputs"]["cfg"] = kwargs["cfg"]
                print(f"Updated cfg: {kwargs['cfg']}")
            if "sampler_name" in kwargs:
                modified_workflow["20"]["inputs"]["sampler_name"] = kwargs["sampler_name"]
                print(f"Updated sampler_name: {kwargs['sampler_name']}")
            if "scheduler" in kwargs:
                modified_workflow["20"]["inputs"]["scheduler"] = kwargs["scheduler"]
                print(f"Updated scheduler: {kwargs['scheduler']}")
            if "denoise" in kwargs:
                modified_workflow["20"]["inputs"]["denoise"] = kwargs["denoise"]
                print(f"Updated denoise: {kwargs['denoise']}")
        
        # 4. Update FluxGuidance parameters (node "16")
        if "16" in modified_workflow and "guidance" in kwargs:
            modified_workflow["16"]["inputs"]["guidance"] = kwargs["guidance"]
            print(f"Updated guidance: {kwargs['guidance']}")
        
        # 5. Update ModelSamplingFlux parameters (node "24")
        if "24" in modified_workflow:
            if "max_shift" in kwargs:
                modified_workflow["24"]["inputs"]["max_shift"] = kwargs["max_shift"]
                print(f"Updated max_shift: {kwargs['max_shift']}")
            if "base_shift" in kwargs:
                modified_workflow["24"]["inputs"]["base_shift"] = kwargs["base_shift"]
                print(f"Updated base_shift: {kwargs['base_shift']}")
        
        return modified_workflow

    def _execute_workflow_shared_output(self, workflow, server_address, client_id, **kwargs):
        """Execute workflow and use shared memory output"""
        # Add shared memory output node - reference remove_bg_api_node.py approach
        output_shm_name = f"kontext_styles_{uuid.uuid4().hex[:16]}"  # Use longer random name to avoid conflicts
        workflow["save_image_shared_memory_node"] = {
            "inputs": {
                "images": ["11", 0],  # VAEDecode output
                "shm_name": output_shm_name,
                "output_format": "RGB", 
                "convert_rgb_to_bgr": False
            },
            "class_type": "SaveImageSharedMemory",
            "_meta": {"title": "Save Image (Shared Memory)"}
        }
        
        print(f"✓ Configured SaveImageSharedMemory with name: {output_shm_name}")
        
        # Execute workflow and get metadata
        images_metadata = self._execute_workflow_and_get_images(workflow, server_address, client_id, **kwargs)
        
        print(f"DEBUG: All collected metadata: {images_metadata}")
        
        # Get image information from metadata
        if 'save_image_shared_memory_node' not in images_metadata:
            raise RuntimeError("No shared memory output received from workflow")
        
        # Get shared memory information (extract from UI output)
        ui_data = images_metadata.get('save_image_shared_memory_node', {})
        print(f"DEBUG: UI data for save_image_shared_memory_node: {ui_data}")
        
        if not ui_data:
            print(f"✗ No UI metadata received from SaveImageSharedMemory node")
            print("This usually means the SaveImageSharedMemory node failed to execute properly")
            # Fall back to WebSocket approach directly, no more guessing
            print("Falling back to WebSocket output method...")
            return self._execute_workflow_websocket_output(workflow, server_address, client_id, **kwargs)
        
        # Parse SaveImageSharedMemory returned complete metadata - reference remove_bg_api_node.py implementation
        shared_memory_info = None
        
        # First try to get from direct shared_memory_info field (remove_bg_api_node.py approach)
        if isinstance(ui_data, dict) and 'shared_memory_info' in ui_data:
            shared_memory_info_list = ui_data['shared_memory_info']
            if isinstance(shared_memory_info_list, list) and len(shared_memory_info_list) > 0:
                shared_memory_info = shared_memory_info_list[0]  # Take first result
                print(f"✓ Found shared_memory_info directly (remove_bg_api_node.py style):")
                print(f"  - shm_name: {shared_memory_info.get('shm_name', 'N/A')}")
                print(f"  - shape: {shared_memory_info.get('shape', 'N/A')}")
                print(f"  - dtype: {shared_memory_info.get('dtype', 'N/A')}")
                print(f"  - format: {shared_memory_info.get('format', 'N/A')}")
                print(f"  - size: {shared_memory_info.get('size_mb', 'N/A')} MB")
        
        # If direct approach fails, try getting from UI field (original approach)
        if not shared_memory_info and isinstance(ui_data, dict) and 'ui' in ui_data:
            ui_inner = ui_data['ui']
            if isinstance(ui_inner, dict) and 'shared_memory_info' in ui_inner:
                shared_memory_info_list = ui_inner['shared_memory_info']
                if isinstance(shared_memory_info_list, list) and len(shared_memory_info_list) > 0:
                    shared_memory_info = shared_memory_info_list[0]
                    print(f"✓ Found shared_memory_info in UI field (original style):")
                    print(f"  - shm_name: {shared_memory_info.get('shm_name', 'N/A')}")
                    print(f"  - shape: {shared_memory_info.get('shape', 'N/A')}")
                    print(f"  - dtype: {shared_memory_info.get('dtype', 'N/A')}")
        
        if not shared_memory_info:
            print(f"Warning: No shared_memory_info found in any expected location")
            print(f"UI data keys: {list(ui_data.keys()) if isinstance(ui_data, dict) else 'not a dict'}")
            print(f"UI data content: {ui_data}")
        
        # Validate required parameters
        required_fields = ['shm_name', 'shape', 'dtype']
        if not shared_memory_info or not isinstance(shared_memory_info, dict):
            print(f"✗ shared_memory_info is not a valid dict: {shared_memory_info}")
            print("Falling back to WebSocket output method...")
            return self._execute_workflow_websocket_output(workflow, server_address, client_id, **kwargs)
        
        missing_fields = [field for field in required_fields if field not in shared_memory_info]
        if missing_fields:
            print(f"✗ Missing required fields in shared_memory_info: {missing_fields}")
            print(f"Available fields: {list(shared_memory_info.keys())}")
            print("Falling back to WebSocket output method...")
            return self._execute_workflow_websocket_output(workflow, server_address, client_id, **kwargs)
        
        # Read result from shared memory - use complete parameters provided by SaveImageSharedMemory
        try:
            # Extract all parameters returned by SaveImageSharedMemory
            actual_shm_name = shared_memory_info['shm_name']
            image_shape = shared_memory_info['shape']  # [height, width, channels]
            image_dtype = shared_memory_info['dtype']  # Get directly from node, not using default value
            output_format = shared_memory_info.get('format', 'RGB')
            expected_size = shared_memory_info.get('size_bytes', 0)
            
            print(f"✓ Reading image from shared memory using SaveImageSharedMemory parameters:")
            print(f"  - shm_name: {actual_shm_name}")
            print(f"  - shape: {image_shape} (H×W×C)")
            print(f"  - dtype: {image_dtype}")
            print(f"  - format: {output_format}")
            print(f"  - expected_size: {expected_size} bytes")
            
            # Validate image shape
            if not isinstance(image_shape, list) or len(image_shape) != 3:
                raise ValueError(f"Invalid image shape: {image_shape}, expected [height, width, channels]")
            
            height, width, channels = image_shape
            if channels not in [1, 3, 4]:
                raise ValueError(f"Unsupported number of channels: {channels}")
            
            # Connect to shared memory
            result_shm = shared_memory.SharedMemory(name=actual_shm_name)
            
            # Validate shared memory size
            actual_size = result_shm.size
            expected_size_calc = height * width * channels * np.dtype(image_dtype).itemsize
            
            if actual_size != expected_size_calc:
                print(f"Warning: Shared memory size mismatch!")
                print(f"  - Actual size: {actual_size} bytes")
                print(f"  - Expected size: {expected_size_calc} bytes")
                print(f"  - Reported size: {expected_size} bytes")
            
            # Use precise parameters provided by SaveImageSharedMemory to rebuild image
            numpy_dtype = np.dtype(image_dtype)
            image_array = np.ndarray(image_shape, dtype=numpy_dtype, buffer=result_shm.buf)
            
            # Copy data (avoid data loss after shared memory is released)
            result_copy = image_array.copy()
            
            # Clean up shared memory
            result_shm.close()
            result_shm.unlink()
            
            print(f"✓ Successfully loaded image with exact shape: {result_copy.shape}")
            print(f"✓ Image format: {output_format}, dtype: {result_copy.dtype}")
            
            # Add alpha channel if RGB (to match original kontext_omni_styles_api.py behavior)
            if len(result_copy.shape) == 3 and result_copy.shape[2] == 3:
                alpha_channel = np.ones((result_copy.shape[0], result_copy.shape[1], 1), dtype=result_copy.dtype) * 255
                result_copy = np.concatenate([result_copy, alpha_channel], axis=2)
                print(f"✓ Added alpha channel: {result_copy.shape} (RGB→RGBA)")
            
            return result_copy
            
        except Exception as e:
            print(f"✗ Failed to read from shared memory using SaveImageSharedMemory parameters: {e}")
            print(f"  - shm_name: {shared_memory_info.get('shm_name', 'N/A')}")
            print(f"  - shape: {shared_memory_info.get('shape', 'N/A')}")
            print(f"  - dtype: {shared_memory_info.get('dtype', 'N/A')}")
            
            # Try to clean up possible existing shared memory
            if shared_memory_info and 'shm_name' in shared_memory_info:
                try:
                    cleanup_shm = shared_memory.SharedMemory(name=shared_memory_info['shm_name'])
                    cleanup_shm.close()
                    cleanup_shm.unlink()
                    print(f"✓ Cleaned up orphaned shared memory: {shared_memory_info['shm_name']}")
                except:
                    pass  # Ignore cleanup failures
            
            print("Falling back to WebSocket output method...")
            return self._execute_workflow_websocket_output(workflow, server_address, client_id, **kwargs)

    def _execute_workflow_websocket_output(self, workflow, server_address, client_id, **kwargs):
        """Execute workflow and use WebSocket output"""
        # Add WebSocket output node
        workflow["save_image_websocket_node"] = {
            "inputs": {"images": ["11", 0]},  # VAEDecode output
            "class_type": "SaveImageWebsocket",
            "_meta": {"title": "Save Image (WebSocket)"}
        }
        
        # Execute workflow and get images
        images = self._execute_workflow_and_get_images(workflow, server_address, client_id, **kwargs)
        
        if 'save_image_websocket_node' in images:
            # Get image data from WebSocket output
            output_image_data = images['save_image_websocket_node'][0]
            
            # Convert to numpy array
            pil_image = Image.open(io.BytesIO(output_image_data))
            
            # Ensure RGB format
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            numpy_array = np.array(pil_image)
            
            # Add alpha channel to match shared memory behavior
            alpha_channel = np.ones((numpy_array.shape[0], numpy_array.shape[1], 1), dtype=numpy_array.dtype) * 255
            numpy_array = np.concatenate([numpy_array, alpha_channel], axis=2)
            
            return numpy_array
        else:
            raise RuntimeError("No output images received from workflow")

    def _execute_workflow_and_get_images(self, workflow, server_address, client_id, **kwargs):
        """Execute workflow and get output (including shared memory metadata)"""
        try:
            # Submit workflow
            comfy_api_key = kwargs.get('comfy_api_key')
            p = {"prompt": workflow, "client_id": client_id}
            
            # If API Key is provided, add to extra_data
            if comfy_api_key:
                p["extra_data"] = {
                    "api_key_comfy_org": comfy_api_key
                }
            
            data = json.dumps(p).encode('utf-8')
            req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
            req.add_header('Content-Type', 'application/json')
            
            response = urllib.request.urlopen(req)
            result = json.loads(response.read())
            prompt_id = result['prompt_id']
            
            # Connect WebSocket to get results
            ws = websocket.WebSocket()
            ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
            
            output_images = {}
            current_node = ""
            execution_error = None
            
            print(f"Waiting for execution of prompt {prompt_id}...")
            
            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['prompt_id'] == prompt_id:
                            if data['node'] is None:
                                print("✓ Execution completed")
                                break
                            else:
                                current_node = data['node']
                                print(f"  Executing node: {current_node}")
                    
                    elif message['type'] == 'execution_error':
                        error_data = message['data']
                        if error_data.get('prompt_id') == prompt_id:
                            execution_error = error_data
                            print(f"✗ Execution error: {error_data}")
                            break
                    
                    elif message['type'] == 'execution_interrupted':
                        print("✗ Execution interrupted")
                        break
                    
                    elif message['type'] == 'executed':
                        # Get node execution result (including shared memory info)
                        data = message['data']
                        if data['prompt_id'] == prompt_id and 'output' in data:
                            node_id = data['node']
                            node_output = data['output']
                            
                            # Save node output
                            if node_id not in output_images:
                                output_images[node_id] = {}
                            
                            # Check shared memory info - reference remove_bg_api_node.py implementation
                            if 'shared_memory_info' in node_output:
                                output_images[node_id]['shared_memory_info'] = node_output['shared_memory_info']
                                print(f"✓ Received shared memory info from node {node_id}: {node_output['shared_memory_info']}")
                            
                            # Also save UI output (backup)
                            if 'ui' in node_output:
                                if 'ui' not in output_images[node_id]:
                                    output_images[node_id]['ui'] = {}
                                output_images[node_id]['ui'].update(node_output['ui'])
                                print(f"✓ Saved UI output from node {node_id}: {node_output['ui']}")
                
                else:
                    # Binary data (WebSocket output images)
                    if current_node:
                        if current_node not in output_images:
                            output_images[current_node] = []
                        output_images[current_node].append(out[8:])  # Skip first 8 bytes header
                        print(f"✓ Saved binary data from {current_node}, size: {len(out)} bytes")
            
            ws.close()
            
            if execution_error:
                raise RuntimeError(f"Workflow execution failed: {execution_error}")
            
            print(f"Total outputs collected: {list(output_images.keys())}")
            return output_images
            
        except Exception as e:
            raise RuntimeError(f"Failed to execute workflow and get images: {e}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "RemoteKontextOmniStylesNode": RemoteKontextOmniStylesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteKontextOmniStylesNode": "Remote Kontext Omni Styles",
}
"""
Remote Photo Super Resolution Enhancement Node for ComfyUI

使用远程 ComfyUI 服务进行照片超分辨率增强处理，避免本地加载模型
支持负载均衡、共享内存传输和完整的增强参数控制
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

class RemotePhotoSREnhanceNode:
    """
    远程照片超分辨率增强节点
    
    使用远程 ComfyUI 服务器进行照片增强，支持：
    - 自动负载均衡
    - 共享内存高速传输
    - 完整的增强参数控制
    - 可选的上采样功能
    - 从工作流文件读取配置（更稳定）
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "enable_upscale": ("BOOLEAN", {"default": True, "tooltip": "是否启用上采样处理"}),
                "server_addresses": ("STRING", {
                    "default": "127.0.0.1:8211,127.0.0.1:8212,127.0.0.1:8213,127.0.0.1:8214,127.0.0.1:8215,127.0.0.1:8216,127.0.0.1:8217,127.0.0.1:8218,127.0.0.1:8219,127.0.0.1:8220",
                    "tooltip": "ComfyUI服务器地址列表，用逗号分隔，支持负载均衡"
                }),
                "use_shared_memory_output": ("BOOLEAN", {"default": True, "tooltip": "是否使用共享内存输出（40-50x 性能提升）"}),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "high quality, detailed, photograph , hd, 8k , 4k , sharp, highly detailed",
                    "tooltip": "增强提示词"
                }),
                "upscale_model": ("STRING", {
                    "default": "2xNomosUni_span_multijpg_ldl.safetensors",
                    "tooltip": "上采样模型名称"
                }),
                "split_denoise": ("FLOAT", {
                    "default": 0.5000000000000001, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "分割去噪强度 (SplitSigmasDenoise)"
                }),
                "scheduler_denoise": ("FLOAT", {
                    "default": 0.30000000000000004, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "调度器去噪强度 (BasicScheduler)"
                }),
                "guidance": ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "指导强度"
                }),
                "steps": ("INT", {
                    "default": 15, "min": 1, "max": 100,
                    "tooltip": "采样步数"
                }),
                "scheduler": (["kl_optimal", "normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"], {
                    "default": "kl_optimal",
                    "tooltip": "调度器类型"
                }),
                "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "ddim", "uni_pc", "uni_pc_bh2"], {
                    "default": "euler",
                    "tooltip": "采样器名称"
                }),
                "vae_device": ("STRING", {
                    "default": "cuda:1",
                    "tooltip": "VAE 运行设备"
                }),
                "random_seed": ("INT", {
                    "default": 710874769520552, "min": -1, "max": 2**63-1,
                    "tooltip": "随机种子，-1为自动"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "enhance_photo"
    CATEGORY = "api/image/enhance"

    def enhance_photo(self, image, enable_upscale, server_addresses, use_shared_memory_output=True, 
                     prompt="high quality, detailed, photograph , hd, 8k , 4k , sharp, highly detailed",
                     upscale_model="2xNomosUni_span_multijpg_ldl.safetensors",
                     split_denoise=0.5000000000000001, scheduler_denoise=0.30000000000000004, guidance=3.5, steps=15, 
                     scheduler="kl_optimal", sampler_name="euler",
                     vae_device="cuda:1", random_seed=710874769520552):
        """
        执行照片超分辨率增强
        """
        start_time = time.time()
        
        # 解析服务器地址列表
        server_list = [addr.strip() for addr in server_addresses.split(',') if addr.strip()]
        if not server_list:
            raise ValueError("至少需要提供一个服务器地址")
        
        print(f"Photo SR Enhancement - Enable upscale: {enable_upscale}")
        print(f"Server addresses: {server_list}")
        print(f"Enhancement parameters: split_denoise={split_denoise}, scheduler_denoise={scheduler_denoise}, guidance={guidance}, steps={steps}")
        
        # 转换图像格式 (ComfyUI tensor -> numpy RGB)
        if len(image.shape) == 4:
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        
        print(f"Input image shape: {image_np.shape}")
        
        # 处理随机种子 - 确保不传递None值
        if random_seed == -1:
            seed_to_use = random.randint(0, 2**31-1)  # 生成随机种子而不是None
        else:
            seed_to_use = random_seed
        
        # 创建独立的客户端实例
        client_id = str(uuid.uuid4())
        
        try:
            # 选择最佳服务器
            selected_server = self._select_best_server(server_list)
            if not selected_server:
                raise RuntimeError("没有可用的 ComfyUI 服务器")
            
            # 执行增强处理
            result_np = self._process_with_shared_memory(
                image_np, enable_upscale, selected_server, client_id,
                prompt=prompt, upscale_model=upscale_model,
                split_denoise=split_denoise, scheduler_denoise=scheduler_denoise, 
                guidance=guidance, steps=steps,
                scheduler=scheduler, sampler_name=sampler_name,
                vae_device=vae_device, random_seed=seed_to_use,
                use_shared_memory_output=use_shared_memory_output
            )
            
            # 转换回 ComfyUI tensor 格式
            result_tensor = torch.from_numpy(result_np.astype(np.float32) / 255.0).unsqueeze(0)
            
            total_time = time.time() - start_time
            print(f"✓ Photo SR Enhancement completed in {total_time:.3f}s")
            print(f"  Output shape: {result_tensor.shape}")
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"✗ Photo SR Enhancement failed: {e}")
            raise

    def _select_best_server(self, server_list):
        """选择最佳的ComfyUI服务器"""
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
        
        # 选择负载最低的服务器，如果负载相同则选择VRAM使用率最低的
        best_server = min(available_servers, key=lambda x: (x['total_load'], x['vram_usage_ratio']))
        
        print(f"Selected server: {best_server['server_address']} (load={best_server['total_load']})")
        print("=" * 50)
        
        return best_server['server_address']

    def _check_server_status(self, server_address):
        """检查ComfyUI服务器状态"""
        try:
            # 检查队列状态
            queue_url = f"http://{server_address}/queue"
            queue_req = urllib.request.Request(queue_url)
            queue_req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(queue_req, timeout=3) as response:
                queue_data = json.loads(response.read())
            
            # 检查系统状态
            stats_url = f"http://{server_address}/system_stats"
            stats_req = urllib.request.Request(stats_url)
            
            with urllib.request.urlopen(stats_req, timeout=3) as response:
                system_data = json.loads(response.read())
            
            # 计算服务器负载
            queue_running = len(queue_data.get('queue_running', []))
            queue_pending = len(queue_data.get('queue_pending', []))
            total_load = queue_running + queue_pending
            
            # 获取VRAM使用情况
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

    def _process_with_shared_memory(self, image_array, enable_upscale, server_address, client_id,
                                   use_shared_memory_output=True, **kwargs):
        """使用共享内存处理图像"""
        shm_obj = None
        
        try:
            # 1. 创建共享内存
            shm_name, shape, dtype, shm_obj = self._numpy_to_shared_memory(image_array)
            shm_data = (shm_name, shape, dtype)
            
            # 2. 从文件加载并修改工作流
            workflow = self._load_and_modify_workflow(shm_data, enable_upscale, **kwargs)
            
            # 3. 执行工作流
            if use_shared_memory_output:
                result_array = self._execute_workflow_shared_output(workflow, server_address, client_id)
            else:
                result_array = self._execute_workflow_websocket_output(workflow, server_address, client_id)
            
            return result_array
            
        finally:
            # 清理共享内存
            if shm_obj:
                try:
                    shm_obj.close()
                    shm_obj.unlink()
                except:
                    pass

    def _numpy_to_shared_memory(self, image_array):
        """将numpy数组存储到共享内存"""
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError(f"Expected image array shape (h, w, 3), got {image_array.shape}")
        
        # 生成共享内存名称
        shm_name = f"comfyui_img_{uuid.uuid4().hex[:8]}"
        
        # 确保数组是连续的
        if not image_array.flags['C_CONTIGUOUS']:
            image_array = np.ascontiguousarray(image_array)
        
        # 创建共享内存
        shm = shared_memory.SharedMemory(create=True, size=image_array.nbytes, name=shm_name)
        
        # 将数据复制到共享内存
        shm_array = np.ndarray(image_array.shape, dtype=image_array.dtype, buffer=shm.buf)
        shm_array[:] = image_array[:]
        
        return shm_name, list(image_array.shape), str(image_array.dtype), shm

    def _load_and_modify_workflow(self, shm_data, enable_upscale, **kwargs):
        """从文件加载工作流并修改参数"""
        try:
            # 获取工作流文件路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            workflow_path = os.path.join(current_dir, "..", "user", "default", "workflows", "A-photo-sr-enhance-api.json")
            
            # 如果相对路径不存在，尝试绝对路径
            if not os.path.exists(workflow_path):
                comfyui_root = os.path.dirname(current_dir)
                workflow_path = os.path.join(comfyui_root, "user", "default", "workflows", "A-photo-sr-enhance-api.json")
            
            # 再次检查文件是否存在
            if not os.path.exists(workflow_path):
                raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
            
            # 加载原始工作流
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            
            print(f"✓ Loaded workflow from: {workflow_path}")
            
            # 修改工作流
            modified_workflow = self._modify_workflow_parameters(workflow, shm_data, enable_upscale, **kwargs)
            
            return modified_workflow
            
        except Exception as e:
            print(f"Failed to load workflow from file: {e}")
            raise

    def _modify_workflow_parameters(self, workflow, shm_data, enable_upscale, **kwargs):
        """修改工作流参数"""
        # 创建工作流副本
        modified_workflow = workflow.copy()
        
        # 1. 替换LoadImage节点为LoadImageSharedMemory节点（节点"1"）
        if "1" in modified_workflow:
            print(f"Replacing LoadImage node with LoadImageSharedMemory node")
            modified_workflow["1"] = {
                "inputs": {
                    "shm_name": shm_data[0],
                    "shape": json.dumps(shm_data[1]),
                    "dtype": shm_data[2],
                    "convert_bgr_to_rgb": False  # PIL输入已经是RGB格式，无需转换
                },
                "class_type": "LoadImageSharedMemory",
                "_meta": {"title": "Load Image (Shared Memory)"}
            }
            print(f"Updated to LoadImageSharedMemory node with shm_name: {shm_data[0]}")
        
        # 2. 修改布尔值节点以控制是否启用上采样（节点"102"）
        if "102" in modified_workflow:
            modified_workflow["102"]["inputs"]["value"] = enable_upscale
            print(f"Updated upscale boolean value: {enable_upscale}")
        
        # 3. 修改提示词（节点"87"）
        if "87" in modified_workflow:
            modified_workflow["87"]["inputs"]["text"] = kwargs.get("prompt", "high quality, detailed, photograph , hd, 8k , 4k , sharp, highly detailed")
            print(f"Updated prompt: {kwargs.get('prompt', 'default')}")
        
        # 4. 更新上采样模型（节点"98"）
        if "98" in modified_workflow:
            modified_workflow["98"]["inputs"]["model_name"] = kwargs.get("upscale_model", "2xNomosUni_span_multijpg_ldl.safetensors")
            print(f"Updated upscale model: {kwargs.get('upscale_model', 'default')}")
        
        # 5. 更新分割去噪强度（节点"58" SplitSigmasDenoise）
        if "58" in modified_workflow:
            modified_workflow["58"]["inputs"]["denoise"] = kwargs.get("split_denoise", 0.5)
            print(f"Updated split denoise strength: {kwargs.get('split_denoise', 0.5)}")
        
        # 5.5. 更新调度器去噪强度（节点"59" BasicScheduler）
        if "59" in modified_workflow and "inputs" in modified_workflow["59"]:
            modified_workflow["59"]["inputs"]["denoise"] = kwargs.get("scheduler_denoise", 0.3)
            print(f"Updated scheduler denoise strength: {kwargs.get('scheduler_denoise', 0.3)}")
        
        # 6. 更新指导强度（节点"55"）
        if "55" in modified_workflow:
            modified_workflow["55"]["inputs"]["guidance"] = kwargs.get("guidance", 3.5)
            print(f"Updated guidance: {kwargs.get('guidance', 3.5)}")
        
        # 7. 更新调度器参数（节点"59"）
        if "59" in modified_workflow:
            modified_workflow["59"]["inputs"]["scheduler"] = kwargs.get("scheduler", "kl_optimal")
            modified_workflow["59"]["inputs"]["steps"] = kwargs.get("steps", 15)
            print(f"Updated scheduler: {kwargs.get('scheduler', 'kl_optimal')}, steps: {kwargs.get('steps', 15)}")
        
        # 8. 更新采样器（节点"56"）
        if "56" in modified_workflow:
            modified_workflow["56"]["inputs"]["sampler_name"] = kwargs.get("sampler_name", "euler")
            print(f"Updated sampler: {kwargs.get('sampler_name', 'euler')}")
        
        # 9. 更新VAE设备配置（节点"103"）
        if "103" in modified_workflow:
            modified_workflow["103"]["inputs"]["device"] = kwargs.get("vae_device", "cuda:1")
            print(f"Updated VAE device: {kwargs.get('vae_device', 'cuda:1')}")
        
        # 10. 更新随机种子（节点"63"） - 确保不传递None值
        if "63" in modified_workflow:
            # random_seed 已经在 enhance_photo 中处理为有效整数
            modified_workflow["63"]["inputs"]["noise_seed"] = kwargs.get("random_seed", 710874769520552)
            print(f"Updated random seed: {kwargs.get('random_seed', 710874769520552)}")
        
        return modified_workflow

    def _execute_workflow_shared_output(self, workflow, server_address, client_id):
        """执行工作流并使用共享内存输出"""
        # 添加共享内存输出节点 - 参考 remove_bg_api_node.py 的方式
        output_shm_name = f"photo_sr_{uuid.uuid4().hex[:16]}"  # 使用更长的随机名称避免冲突
        workflow["save_image_shared_memory_node"] = {
            "inputs": {
                "images": ["42", 0],
                "shm_name": output_shm_name,
                "output_format": "RGB", 
                "convert_rgb_to_bgr": False
            },
            "class_type": "SaveImageSharedMemory",
            "_meta": {"title": "Save Image (Shared Memory)"}
        }
        
        print(f"✓ Configured SaveImageSharedMemory with name: {output_shm_name}")
        
        # 执行工作流并获取元数据
        images_metadata = self._execute_workflow_and_get_images(workflow, server_address, client_id)
        
        print(f"DEBUG: All collected metadata: {images_metadata}")
        
        # 从元数据获取图像信息
        if 'save_image_shared_memory_node' not in images_metadata:
            raise RuntimeError("No shared memory output received from workflow")
        
        # 获取共享内存信息（从UI输出中提取）
        ui_data = images_metadata.get('save_image_shared_memory_node', {})
        print(f"DEBUG: UI data for save_image_shared_memory_node: {ui_data}")
        
        if not ui_data:
            print(f"✗ No UI metadata received from SaveImageSharedMemory node")
            print("This usually means the SaveImageSharedMemory node failed to execute properly")
            # 直接回退到WebSocket方案，不再尝试猜测
            print("Falling back to WebSocket output method...")
            return self._execute_workflow_websocket_output(workflow, server_address, client_id)
        
        # 解析 SaveImageSharedMemory 返回的完整元数据 - 参考 remove_bg_api_node.py 的实现
        shared_memory_info = None
        
        # 首先尝试从直接的 shared_memory_info 字段获取（remove_bg_api_node.py 的方式）
        if isinstance(ui_data, dict) and 'shared_memory_info' in ui_data:
            shared_memory_info_list = ui_data['shared_memory_info']
            if isinstance(shared_memory_info_list, list) and len(shared_memory_info_list) > 0:
                shared_memory_info = shared_memory_info_list[0]  # 取第一个结果
                print(f"✓ Found shared_memory_info directly (remove_bg_api_node.py style):")
                print(f"  - shm_name: {shared_memory_info.get('shm_name', 'N/A')}")
                print(f"  - shape: {shared_memory_info.get('shape', 'N/A')}")
                print(f"  - dtype: {shared_memory_info.get('dtype', 'N/A')}")
                print(f"  - format: {shared_memory_info.get('format', 'N/A')}")
                print(f"  - size: {shared_memory_info.get('size_mb', 'N/A')} MB")
        
        # 如果直接方式失败，尝试从UI字段获取（原始方式）
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
        
        # 验证必需的参数
        required_fields = ['shm_name', 'shape', 'dtype']
        if not shared_memory_info or not isinstance(shared_memory_info, dict):
            print(f"✗ shared_memory_info is not a valid dict: {shared_memory_info}")
            print("Falling back to WebSocket output method...")
            return self._execute_workflow_websocket_output(workflow, server_address, client_id)
        
        missing_fields = [field for field in required_fields if field not in shared_memory_info]
        if missing_fields:
            print(f"✗ Missing required fields in shared_memory_info: {missing_fields}")
            print(f"Available fields: {list(shared_memory_info.keys())}")
            print("Falling back to WebSocket output method...")
            return self._execute_workflow_websocket_output(workflow, server_address, client_id)
        
        # 从共享内存读取结果 - 使用 SaveImageSharedMemory 提供的完整参数
        try:
            # 提取 SaveImageSharedMemory 返回的所有参数
            actual_shm_name = shared_memory_info['shm_name']
            image_shape = shared_memory_info['shape']  # [height, width, channels]
            image_dtype = shared_memory_info['dtype']  # 从节点直接获取，不使用默认值
            output_format = shared_memory_info.get('format', 'RGB')
            expected_size = shared_memory_info.get('size_bytes', 0)
            
            print(f"✓ Reading image from shared memory using SaveImageSharedMemory parameters:")
            print(f"  - shm_name: {actual_shm_name}")
            print(f"  - shape: {image_shape} (H×W×C)")
            print(f"  - dtype: {image_dtype}")
            print(f"  - format: {output_format}")
            print(f"  - expected_size: {expected_size} bytes")
            
            # 验证图像形状
            if not isinstance(image_shape, list) or len(image_shape) != 3:
                raise ValueError(f"Invalid image shape: {image_shape}, expected [height, width, channels]")
            
            height, width, channels = image_shape
            if channels not in [1, 3, 4]:
                raise ValueError(f"Unsupported number of channels: {channels}")
            
            # 连接到共享内存
            result_shm = shared_memory.SharedMemory(name=actual_shm_name)
            
            # 验证共享内存大小
            actual_size = result_shm.size
            expected_size_calc = height * width * channels * np.dtype(image_dtype).itemsize
            
            if actual_size != expected_size_calc:
                print(f"Warning: Shared memory size mismatch!")
                print(f"  - Actual size: {actual_size} bytes")
                print(f"  - Expected size: {expected_size_calc} bytes")
                print(f"  - Reported size: {expected_size} bytes")
            
            # 使用 SaveImageSharedMemory 提供的精确参数重建图像
            numpy_dtype = np.dtype(image_dtype)
            image_array = np.ndarray(image_shape, dtype=numpy_dtype, buffer=result_shm.buf)
            
            # 复制数据（避免共享内存被释放后数据丢失）
            result_copy = image_array.copy()
            
            # 清理共享内存
            result_shm.close()
            result_shm.unlink()
            
            print(f"✓ Successfully loaded image with exact shape: {result_copy.shape}")
            print(f"✓ Image format: {output_format}, dtype: {result_copy.dtype}")
            
            return result_copy
            
        except Exception as e:
            print(f"✗ Failed to read from shared memory using SaveImageSharedMemory parameters: {e}")
            print(f"  - shm_name: {shared_memory_info.get('shm_name', 'N/A')}")
            print(f"  - shape: {shared_memory_info.get('shape', 'N/A')}")
            print(f"  - dtype: {shared_memory_info.get('dtype', 'N/A')}")
            
            # 尝试清理可能存在的共享内存
            if shared_memory_info and 'shm_name' in shared_memory_info:
                try:
                    cleanup_shm = shared_memory.SharedMemory(name=shared_memory_info['shm_name'])
                    cleanup_shm.close()
                    cleanup_shm.unlink()
                    print(f"✓ Cleaned up orphaned shared memory: {shared_memory_info['shm_name']}")
                except:
                    pass  # 忽略清理失败
            
            print("Falling back to WebSocket output method...")
            return self._execute_workflow_websocket_output(workflow, server_address, client_id)

    def _execute_workflow_websocket_output(self, workflow, server_address, client_id):
        """执行工作流并使用WebSocket输出"""
        # 添加WebSocket输出节点
        workflow["save_image_websocket_node"] = {
            "inputs": {"images": ["42", 0]},
            "class_type": "SaveImageWebsocket",
            "_meta": {"title": "Save Image (WebSocket)"}
        }
        
        # 执行工作流并获取图像
        images = self._execute_workflow_and_get_images(workflow, server_address, client_id)
        
        if 'save_image_websocket_node' in images:
            # 从WebSocket输出获取图像数据
            output_image_data = images['save_image_websocket_node'][0]
            
            # 转换为numpy数组
            pil_image = Image.open(io.BytesIO(output_image_data))
            
            # 确保是RGB格式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            numpy_array = np.array(pil_image)
            return numpy_array
        else:
            raise RuntimeError("No output images received from workflow")

    def _execute_workflow(self, workflow, server_address, client_id):
        """执行工作流（仅执行，不获取图像）"""
        try:
            # 提交工作流
            p = {"prompt": workflow, "client_id": client_id}
            data = json.dumps(p).encode('utf-8')
            req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
            req.add_header('Content-Type', 'application/json')
            
            response = urllib.request.urlopen(req)
            result = json.loads(response.read())
            prompt_id = result['prompt_id']
            
            # 连接WebSocket监听执行状态
            ws = websocket.WebSocket()
            ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
            
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
                                print(f"  Executing node: {data['node']}")
                    
                    elif message['type'] == 'execution_error':
                        error_data = message['data']
                        if error_data.get('prompt_id') == prompt_id:
                            raise RuntimeError(f"Workflow execution failed: {error_data}")
                    
                    elif message['type'] == 'execution_interrupted':
                        raise RuntimeError("Execution interrupted")
            
            ws.close()
            
        except Exception as e:
            raise RuntimeError(f"Failed to execute workflow: {e}")

    def _execute_workflow_and_get_images(self, workflow, server_address, client_id):
        """执行工作流并获取输出（包括共享内存元数据）"""
        try:
            # 提交工作流
            p = {"prompt": workflow, "client_id": client_id}
            data = json.dumps(p).encode('utf-8')
            req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
            req.add_header('Content-Type', 'application/json')
            
            response = urllib.request.urlopen(req)
            result = json.loads(response.read())
            prompt_id = result['prompt_id']
            
            # 连接WebSocket获取结果
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
                        # 获取节点执行结果（包括共享内存信息）
                        data = message['data']
                        if data['prompt_id'] == prompt_id and 'output' in data:
                            node_id = data['node']
                            node_output = data['output']
                            
                            # 保存节点输出
                            if node_id not in output_images:
                                output_images[node_id] = {}
                            
                            # 检查共享内存信息 - 参考 remove_bg_api_node.py 的实现
                            if 'shared_memory_info' in node_output:
                                output_images[node_id]['shared_memory_info'] = node_output['shared_memory_info']
                                print(f"✓ Received shared memory info from node {node_id}: {node_output['shared_memory_info']}")
                            
                            # 同时保存UI输出（备用）
                            if 'ui' in node_output:
                                if 'ui' not in output_images[node_id]:
                                    output_images[node_id]['ui'] = {}
                                output_images[node_id]['ui'].update(node_output['ui'])
                                print(f"✓ Saved UI output from node {node_id}: {node_output['ui']}")
                
                else:
                    # 二进制数据（WebSocket输出的图像）
                    if current_node:
                        if current_node not in output_images:
                            output_images[current_node] = []
                        output_images[current_node].append(out[8:])  # 跳过前8字节头部
                        print(f"✓ Saved binary data from {current_node}, size: {len(out)} bytes")
            
            ws.close()
            
            if execution_error:
                raise RuntimeError(f"Workflow execution failed: {execution_error}")
            
            print(f"Total outputs collected: {list(output_images.keys())}")
            return output_images
            
        except Exception as e:
            raise RuntimeError(f"Failed to execute workflow and get images: {e}")

# 节点注册
NODE_CLASS_MAPPINGS = {
    "RemotePhotoSREnhanceNode": RemotePhotoSREnhanceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemotePhotoSREnhanceNode": "Remote Photo SR Enhancement",
} 
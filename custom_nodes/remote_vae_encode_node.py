"""
Remote VAE Encode Node for ComfyUI
包装 vae_encode_api.py 的功能，通过远程 ComfyUI 服务进行 VAE 编码处理
支持负载均衡、共享内存高速数据传输，避免重新加载VAE模型
"""

import websocket
import uuid
import json
import urllib.request
import urllib.parse
import urllib.error
import numpy as np
import torch
import time
from multiprocessing import shared_memory
from PIL import Image
import io
import cv2
import os
import sys

# 添加ComfyUI根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
comfyui_root = os.path.dirname(current_dir)
if comfyui_root not in sys.path:
    sys.path.insert(0, comfyui_root)

class RemoteVAEEncodeNode:
    """
    远程VAE编码节点
    通过调用远程ComfyUI服务进行VAE编码，避免重新加载VAE模型
    支持负载均衡和共享内存高速数据传输
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "server_address": ("STRING", {"default": "127.0.0.1:8221", "tooltip": "ComfyUI服务器地址 host:port"}),
                "vae_name": ("STRING", {"default": "ae.safetensors", "tooltip": "VAE模型名称"})
            },
            "optional": {
                "use_load_balancing": ("BOOLEAN", {"default": False, "tooltip": "是否启用负载均衡（检查多个服务器）"}),
                "additional_servers": ("STRING", {"default": "127.0.0.1:8222,127.0.0.1:8223", "tooltip": "其他服务器地址，用逗号分隔"}),
                "use_shared_memory_output": ("BOOLEAN", {"default": True, "tooltip": "是否使用共享内存输出（提升性能）"})
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "process"
    CATEGORY = "api/latent/remote"
    DESCRIPTION = "通过远程ComfyUI服务进行VAE编码，支持负载均衡和共享内存传输"

    def check_server_status(self, server_address):
        """
        检查ComfyUI服务器状态
        Args:
            server_address: 服务器地址 "host:port"
        Returns:
            dict: 服务器状态信息，包含队列信息和系统状态，失败返回None
        """
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
            print(f"Failed to check server {server_address}: {e}")
            return {
                'server_address': server_address,
                'available': False,
                'error': str(e)
            }

    def select_best_server(self, primary_server, additional_servers="", use_load_balancing=False):
        """
        选择最佳的ComfyUI服务器
        Args:
            primary_server: 主服务器地址
            additional_servers: 其他服务器地址，用逗号分隔
            use_load_balancing: 是否启用负载均衡
        Returns:
            str: 最佳服务器地址
        """
        if not use_load_balancing:
            return primary_server
        
        # 构建服务器列表
        servers = [primary_server]
        if additional_servers and additional_servers.strip():
            additional_list = [s.strip() for s in additional_servers.split(',') if s.strip()]
            servers.extend(additional_list)
        
        print("=== Checking ComfyUI servers status ===")
        available_servers = []
        
        for server in servers:
            status = self.check_server_status(server)
            if status['available']:
                available_servers.append(status)
                print(f"✓ {server}: load={status['total_load']}, vram_free={status['vram_free']/(1024**3):.1f}GB")
            else:
                print(f"✗ {server}: {status.get('error', 'unavailable')}")
        
        if not available_servers:
            print(f"No available servers found in load balancing, falling back to primary: {primary_server}")
            return primary_server
        
        # 选择负载最低的服务器，如果负载相同则选择VRAM使用率最低的
        best_server = min(available_servers, key=lambda x: (x['total_load'], x['vram_usage_ratio']))
        
        print(f"Selected server: {best_server['server_address']} (load={best_server['total_load']})")
        print("=" * 50)
        
        return best_server['server_address']

    def queue_prompt(self, workflow, server_address, client_id):
        """提交工作流到ComfyUI服务器队列"""
        p = {"prompt": workflow, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
        req.add_header('Content-Type', 'application/json')
        try:
            return json.loads(urllib.request.urlopen(req).read())
        except urllib.error.HTTPError as e:
            print("Server returned error:", e.read().decode())
            raise

    def get_latent_output_shared_memory(self, ws, workflow, server_address, client_id):
        """
        获取潜在表示输出（共享内存版本）
        """
        prompt_id = self.queue_prompt(workflow, server_address, client_id)['prompt_id']
        latent_shm_info = None
        current_node = ""
        execution_error = None
        
        print(f"Waiting for execution of prompt {prompt_id} on {server_address}...")
        
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                
                if message['type'] == 'executing':
                    data = message['data']
                    if data['prompt_id'] == prompt_id:
                        if data['node'] is None:
                            print("✓ Execution completed")
                            break  # Execution is done
                        else:
                            current_node = data['node']
                            print(f"  Executing node: {current_node}")
                            
                elif message['type'] == 'executed':
                    data = message['data']
                    if data['prompt_id'] == prompt_id:
                        node_id = data['node']
                        
                        # 检查是否有输出数据
                        if 'output' in data and data['output']:
                            node_output = data['output']
                            
                            # 检查SaveLatentSharedMemory节点的输出
                            if node_id == "save_latent_shared_memory_node":
                                if 'latent_shm_info' in node_output:
                                    latent_shm_info = node_output['latent_shm_info'][0]
                                    print(f"✓ Latent shared memory info received: {latent_shm_info}")
                                    
                elif message['type'] == 'execution_error':
                    execution_error = message['data']
                    print(f"✗ Execution error: {execution_error}")
                    break
                    
                elif message['type'] == 'execution_interrupted':
                    print("✗ Execution interrupted")
                    break

        if execution_error:
            raise RuntimeError(f"Workflow execution failed: {execution_error}")

        return latent_shm_info

    def get_latent_output_websocket(self, ws, workflow, server_address, client_id):
        """
        获取潜在表示输出（WebSocket版本）
        """
        prompt_id = self.queue_prompt(workflow, server_address, client_id)['prompt_id']
        output_latents = {}
        current_node = ""
        execution_error = None
        
        print(f"Waiting for execution of prompt {prompt_id} on {server_address}...")
        
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                
                if message['type'] == 'executing':
                    data = message['data']
                    if data['prompt_id'] == prompt_id:
                        if data['node'] is None:
                            print("✓ Execution completed")
                            break  # Execution is done
                        else:
                            current_node = data['node']
                            print(f"  Executing node: {current_node}")
                            
                elif message['type'] == 'execution_error':
                    execution_error = message['data']
                    print(f"✗ Execution error: {execution_error}")
                    break
                    
                elif message['type'] == 'execution_interrupted':
                    print("✗ Execution interrupted")
                    break
                    
            else:
                print(f"Received binary data from node: {current_node}, size: {len(out)} bytes")
                if current_node == 'save_latent_websocket_node':
                    latents_output = output_latents.get(current_node, [])
                    latents_output.append(out[8:])
                    output_latents[current_node] = latents_output
                    print(f"✓ Saved latent data from {current_node}")

        if execution_error:
            raise RuntimeError(f"Workflow execution failed: {execution_error}")
        
        return output_latents

    def numpy_array_to_shared_memory(self, image_array, shm_name=None):
        """
        将numpy数组存储到共享内存中
        Args:
            image_array: 输入图像的numpy数组 (h, w, 3) RGB格式
            shm_name: 共享内存名称，如果为None则自动生成
        Returns:
            tuple: (shm_name, shape, dtype, shared_memory_object)
        """
        # 验证输入格式
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError(f"Expected image array shape (h, w, 3), got {image_array.shape}")
        
        # 生成共享内存名称
        if shm_name is None:
            shm_name = f"comfyui_img_{uuid.uuid4().hex[:8]}"
        
        # 确保数组是连续的
        if not image_array.flags['C_CONTIGUOUS']:
            image_array = np.ascontiguousarray(image_array)
        
        # 创建共享内存
        shm = shared_memory.SharedMemory(create=True, size=image_array.nbytes, name=shm_name)
        
        # 将数据复制到共享内存
        shm_array = np.ndarray(image_array.shape, dtype=image_array.dtype, buffer=shm.buf)
        shm_array[:] = image_array[:]
        
        print(f"Image stored in shared memory: {shm_name}, size: {image_array.nbytes / 1024 / 1024:.2f} MB")
        
        return shm_name, list(image_array.shape), str(image_array.dtype), shm

    def cleanup_shared_memory(self, shm_name=None, shm_object=None):
        """
        清理共享内存
        Args:
            shm_name: 共享内存名称
            shm_object: 共享内存对象
        """
        try:
            if shm_object:
                shm_object.close()
                shm_object.unlink()
            elif shm_name:
                shm = shared_memory.SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
        except Exception as e:
            print(f"Warning: Failed to cleanup shared memory: {e}")

    def load_latent_from_shared_memory(self, shm_info):
        """
        从共享内存加载latent数据
        Args:
            shm_info: 共享内存信息字典，包含shm_name, shape, dtype等
        Returns:
            torch.Tensor: latent张量
        """
        try:
            # 获取共享内存信息
            shm_name = shm_info["shm_name"]
            shape = shm_info["shape"]
            dtype_str = shm_info["dtype"]
            
            # 将字符串转换为numpy dtype
            if hasattr(np, dtype_str):
                numpy_dtype = getattr(np, dtype_str)
            else:
                # 处理torch tensor dtype字符串（如torch.float32）
                if dtype_str.startswith('torch.'):
                    torch_dtype = getattr(torch, dtype_str.split('.')[1])
                    numpy_dtype = torch.zeros(1, dtype=torch_dtype).numpy().dtype
                else:
                    numpy_dtype = np.float32  # 默认类型
            
            # 连接到共享内存
            shm = shared_memory.SharedMemory(name=shm_name)
            
            # 从共享内存重建numpy数组
            latent_array = np.ndarray(shape, dtype=numpy_dtype, buffer=shm.buf)
            
            # 复制数据（避免共享内存被释放）
            latent_copy = latent_array.copy()
            
            # 关闭共享内存连接（不删除）
            shm.close()
            
            # 转换为torch tensor
            latent_tensor = torch.from_numpy(latent_copy).float()
            
            print(f"✓ Loaded latent from shared memory: {shm_name}")
            print(f"  - Shape: {latent_tensor.shape}")
            print(f"  - Dtype: {latent_tensor.dtype}")
            
            return latent_tensor
            
        except Exception as e:
            raise ValueError(f"Error loading latent from shared memory: {e}")

    def create_vae_encode_workflow(self, shm_data, vae_name, use_shared_memory_output=True):
        """
        创建VAE编码工作流
        """
        # 内嵌的工作流模板
        workflow_template = {
            "1": {
                "inputs": {
                    "shm_name": "",
                    "shape": "",
                    "dtype": "",
                    "convert_bgr_to_rgb": False
                },
                "class_type": "LoadImageSharedMemory",
                "_meta": {
                    "title": "Load Image (Shared Memory)"
                }
            },
            "2": {
                "inputs": {
                    "pixels": ["1", 0],
                    "vae": ["3", 0]
                },
                "class_type": "VAEEncode",
                "_meta": {
                    "title": "VAE Encode"
                }
            },
            "3": {
                "inputs": {
                    "vae_name": ""
                },
                "class_type": "VAELoader",
                "_meta": {
                    "title": "Load VAE"
                }
            }
        }
        
        # 设置LoadImageSharedMemory节点参数
        workflow_template["1"]["inputs"]["shm_name"] = shm_data[0]
        workflow_template["1"]["inputs"]["shape"] = json.dumps(shm_data[1])
        workflow_template["1"]["inputs"]["dtype"] = shm_data[2]
        
        # 设置VAE名称
        workflow_template["3"]["inputs"]["vae_name"] = vae_name
        
        # 添加保存节点
        if use_shared_memory_output:
            workflow_template["save_latent_shared_memory_node"] = {
                "inputs": {
                    "samples": ["2", 0]
                },
                "class_type": "SaveLatentSharedMemory",
                "_meta": {
                    "title": "Save Latent (Shared Memory)"
                }
            }
        else:
            workflow_template["save_latent_websocket_node"] = {
                "inputs": {
                    "samples": ["2", 0]
                },
                "class_type": "SaveLatentWebsocket",
                "_meta": {
                    "title": "Save Latent (WebSocket)"
                }
            }
        
        return workflow_template

    def process(self, images, server_address, vae_name, use_load_balancing=False, 
                additional_servers="", use_shared_memory_output=True):
        """
        处理图像进行VAE编码
        """
        start_time = time.time()
        
        # 提取第一张图像（批处理的第一个）
        if images.shape[0] > 1:
            print(f"Warning: Multiple images provided ({images.shape[0]}), only processing the first one")
        
        image_tensor = images[0]  # (H, W, C)
        
        # 转换为numpy数组 (0-255 uint8范围)
        image_numpy = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        print(f"Processing image with VAE encoding, shape: {image_numpy.shape}")
        print(f"Parameters: vae_name={vae_name}")
        
        # 选择最佳服务器
        selected_server = self.select_best_server(
            server_address, additional_servers, use_load_balancing
        )
        
        image_shm_obj = None
        latent_shm_name = None
        
        try:
            # 1. 将图像存储到共享内存
            print("=== Converting image to shared memory ===")
            shm_name, shape, dtype, image_shm_obj = self.numpy_array_to_shared_memory(image_numpy)
            shm_data = (shm_name, shape, dtype)
            
            # 2. 创建工作流
            workflow = self.create_vae_encode_workflow(shm_data, vae_name, use_shared_memory_output)
            
            # 3. 执行工作流
            client_id = str(uuid.uuid4())
            ws = websocket.WebSocket()
            ws.connect(f"ws://{selected_server}/ws?clientId={client_id}")
            
            print(f"Executing ComfyUI VAE encoding workflow on {selected_server}...")
            
            if use_shared_memory_output:
                latent_shm_info = self.get_latent_output_shared_memory(ws, workflow, selected_server, client_id)
                ws.close()
                
                if latent_shm_info:
                    # 从共享内存加载latent数据
                    latent_tensor = self.load_latent_from_shared_memory(latent_shm_info)
                    latent_shm_name = latent_shm_info["shm_name"]
                else:
                    raise RuntimeError("No latent shared memory info received from workflow")
            else:
                latent_outputs = self.get_latent_output_websocket(ws, workflow, selected_server, client_id)
                ws.close()
                
                if 'save_latent_websocket_node' in latent_outputs:
                    # 从WebSocket输出解析latent数据
                    latent_data = latent_outputs['save_latent_websocket_node'][0]
                    # 这里需要根据SaveLatentWebsocket的实际输出格式来解析
                    # 目前ComfyUI可能没有SaveLatentWebsocket节点，所以主要使用共享内存方式
                    raise NotImplementedError("WebSocket latent output parsing not implemented")
                else:
                    raise RuntimeError("No latent output received from workflow")
            
            total_time = time.time() - start_time
            
            print(f"\n=== VAE Encoding Processing Summary ===")
            print(f"  - Server: {selected_server}")
            print(f"  - Total processing time: {total_time:.4f}s")
            print(f"  - Latent shape: {latent_tensor.shape}")
            print(f"  - Using shared memory: {use_shared_memory_output}")
            print("=" * 42)
            
            # 返回ComfyUI latent格式
            return ({"samples": latent_tensor},)
            
        except Exception as e:
            print(f"Error processing VAE encoding: {e}")
            raise
        finally:
            # 清理共享内存
            if image_shm_obj:
                self.cleanup_shared_memory(shm_object=image_shm_obj)
            if latent_shm_name:
                self.cleanup_shared_memory(shm_name=latent_shm_name)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "RemoteVAEEncodeNode": RemoteVAEEncodeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteVAEEncodeNode": "Remote VAE Encode",
}
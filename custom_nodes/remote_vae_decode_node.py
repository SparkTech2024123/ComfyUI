"""
Remote VAE Decode Node for ComfyUI

This node wraps the functionality of vae_decode_api.py into a ComfyUI node,
providing VAE decoding with shared memory support and load balancing.

Features:
- Shared memory input/output for maximum performance
- Load balancing across multiple ComfyUI servers
- All original parameters preserved as node inputs
- No global variables - everything configurable through parameters
"""

import websocket
import uuid
import json
import urllib.request
import urllib.parse
import urllib.error
import numpy as np
import cv2
import os
import sys
from PIL import Image
import io
import torch
import time
from multiprocessing import shared_memory

class RemoteVAEDecodeNode:
    """
    Remote VAE Decode Node - processes latent arrays using remote ComfyUI services
    with shared memory support for maximum performance.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "server_address": ("STRING", {
                    "default": "127.0.0.1:8221", 
                    "tooltip": "ComfyUI server address (host:port)"
                }),
                "vae_name": ("STRING", {
                    "default": "ae.safetensors", 
                    "tooltip": "VAE model filename"
                }),
            },
            "optional": {
                "backup_servers": ("STRING", {
                    "default": "127.0.0.1:8222,127.0.0.1:8223,127.0.0.1:8224", 
                    "tooltip": "Backup server addresses, comma-separated"
                }),
                "use_shared_memory_output": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Use shared memory for output (recommended for performance)"
                }),
                "enable_load_balancing": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Enable automatic server selection based on load"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "api/remote"
    
    def __init__(self):
        self.client_id = str(uuid.uuid4())
    
    def check_server_status(self, server_address):
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
            print(f"Failed to check server {server_address}: {e}")
            return {
                'server_address': server_address,
                'available': False,
                'error': str(e)
            }
    
    def select_best_server(self, primary_server, backup_servers_str, enable_load_balancing):
        """选择最佳的ComfyUI服务器"""
        servers = [primary_server]
        if backup_servers_str.strip():
            backup_servers = [s.strip() for s in backup_servers_str.split(',') if s.strip()]
            servers.extend(backup_servers)
        
        if not enable_load_balancing:
            # 直接使用主服务器
            return primary_server
        
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
            print("No available ComfyUI servers found!")
            return None
        
        # 选择负载最低的服务器，如果负载相同则选择VRAM使用率最低的
        best_server = min(available_servers, key=lambda x: (x['total_load'], x['vram_usage_ratio']))
        
        print(f"Selected server: {best_server['server_address']} (load={best_server['total_load']})")
        print("=" * 50)
        
        return best_server['server_address']
    
    def queue_prompt(self, prompt, server_address):
        """提交工作流到ComfyUI服务器"""
        p = {"prompt": prompt, "client_id": self.client_id}
        
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
        req.add_header('Content-Type', 'application/json')
        try:
            return json.loads(urllib.request.urlopen(req).read())
        except urllib.error.HTTPError as e:
            print("Server returned error:", e.read().decode())
            raise
    
    def get_output_shared_memory(self, ws, prompt, server_address, use_shared_memory):
        """获取输出（共享内存或WebSocket版本）"""
        prompt_id = self.queue_prompt(prompt, server_address)['prompt_id']
        output_data = None
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
                            break
                        else:
                            current_node = data['node']
                            print(f"  Executing node: {current_node}")
                            
                elif message['type'] == 'executed':
                    data = message['data']
                    if data['prompt_id'] == prompt_id:
                        node_id = data['node']
                        
                        if 'output' in data and data['output']:
                            node_output = data['output']
                            
                            # 根据输出模式检查不同的节点
                            if use_shared_memory and node_id == "save_image_shared_memory_node":
                                if 'shared_memory_info' in node_output:
                                    output_data = ('shared_memory', node_output['shared_memory_info'][0])
                                    print(f"✓ Image shared memory info received")
                            elif not use_shared_memory and node_id == "save_image_websocket_node":
                                # WebSocket模式下不会在这里处理，而是在二进制数据中处理
                                pass
                            
                elif message['type'] == 'execution_error':
                    execution_error = message['data']
                    print(f"✗ Execution error: {execution_error}")
                    break
                    
                elif message['type'] == 'execution_interrupted':
                    print("✗ Execution interrupted")
                    break
                    
            else:
                # 二进制数据（WebSocket模式）
                if not use_shared_memory and current_node == 'save_image_websocket_node':
                    print(f"Received binary data from node: {current_node}, size: {len(out)} bytes")
                    output_data = ('websocket', out[8:])  # 跳过前8字节
                    print(f"✓ Saved image data from {current_node}")

        if execution_error:
            raise RuntimeError(f"Workflow execution failed: {execution_error}")

        return output_data
    
    def numpy_array_to_shared_memory(self, latent_array, shm_name=None):
        """将numpy数组存储到共享内存中"""
        start_time = time.time()
        
        # 验证输入格式
        if len(latent_array.shape) != 4:
            raise ValueError(f"Expected latent array shape (batch, channels, height, width), got {latent_array.shape}")
        
        # 生成共享内存名称
        if shm_name is None:
            shm_name = f"comfyui_latent_{uuid.uuid4().hex[:8]}"
        
        # 确保数组是连续的
        if not latent_array.flags['C_CONTIGUOUS']:
            latent_array = np.ascontiguousarray(latent_array)
        
        # 创建共享内存
        shm = shared_memory.SharedMemory(create=True, size=latent_array.nbytes, name=shm_name)
        
        # 将数据复制到共享内存
        shm_array = np.ndarray(latent_array.shape, dtype=latent_array.dtype, buffer=shm.buf)
        shm_array[:] = latent_array[:]
        
        total_time = time.time() - start_time
        
        print(f"Shared memory transfer timing for {latent_array.shape}:")
        print(f"  - Memory allocation and copy: {total_time:.4f}s")
        print(f"  - Shared memory name: {shm_name}")
        print(f"  - Data size: {latent_array.nbytes / 1024 / 1024:.2f} MB")
        
        return shm_name, list(latent_array.shape), str(latent_array.dtype), shm
    
    def cleanup_shared_memory(self, shm_name=None, shm_object=None):
        """清理共享内存"""
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
    
    def load_image_from_shared_memory(self, shm_info):
        """从共享内存加载image数据"""
        try:
            # 获取共享内存信息
            shm_name = shm_info["shm_name"]
            shape = shm_info["shape"]
            dtype_str = shm_info["dtype"]
            
            # 将字符串转换为numpy dtype
            if hasattr(np, dtype_str):
                numpy_dtype = getattr(np, dtype_str)
            else:
                # 处理torch tensor dtype字符串
                if dtype_str.startswith('torch.'):
                    torch_dtype = getattr(torch, dtype_str.split('.')[1])
                    numpy_dtype = torch.zeros(1, dtype=torch_dtype).numpy().dtype
                else:
                    numpy_dtype = np.float32
            
            # 连接到共享内存
            shm = shared_memory.SharedMemory(name=shm_name)
            
            # 从共享内存重建numpy数组
            image_array = np.ndarray(shape, dtype=numpy_dtype, buffer=shm.buf)
            
            # 复制数据（避免共享内存被释放）
            image_copy = image_array.copy()
            
            # 关闭共享内存连接（不删除）
            shm.close()
            
            print(f"✓ Loaded image from shared memory: {shm_name}")
            print(f"  - Shape: {image_copy.shape}")
            print(f"  - Dtype: {image_copy.dtype}")
            
            return image_copy
            
        except Exception as e:
            raise ValueError(f"Error loading image from shared memory: {e}")
    
    def create_workflow(self, shm_data, vae_name, use_shared_memory):
        """创建VAE解码工作流"""
        # 基础工作流模板（基于 A-vae-decode-api.json）
        workflow = {
            "1": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["2", 0]
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAE Decode"
                }
            },
            "2": {
                "inputs": {
                    "vae_name": vae_name
                },
                "class_type": "VAELoader",
                "_meta": {
                    "title": "Load VAE"
                }
            },
            "3": {
                "inputs": {
                    "shm_name": shm_data[0],
                    "shape": json.dumps(shm_data[1]),
                    "dtype": shm_data[2]
                },
                "class_type": "LoadLatentSharedMemory",
                "_meta": {
                    "title": "Load Latent (Shared Memory)"
                }
            }
        }
        
        # 根据输出模式添加保存节点
        if use_shared_memory:
            workflow["save_image_shared_memory_node"] = {
                "inputs": {
                    "images": ["1", 0]
                },
                "class_type": "SaveImageSharedMemory",
                "_meta": {
                    "title": "Save Image (Shared Memory)"
                }
            }
        else:
            workflow["save_image_websocket_node"] = {
                "inputs": {
                    "images": ["1", 0]
                },
                "class_type": "SaveImageWebsocket",
                "_meta": {
                    "title": "Save Image (WebSocket)"
                }
            }
        
        return workflow
    
    def process(self, samples, server_address, vae_name, backup_servers="", 
                use_shared_memory_output=True, enable_load_balancing=True):
        """处理VAE解码"""
        process_start_time = time.time()
        
        # 提取latent张量并转换为numpy数组
        latent_tensor = samples["samples"]
        latent_numpy = latent_tensor.cpu().numpy()
        
        print(f"Processing latent array with shape: {latent_numpy.shape}")
        print(f"Parameters: vae_name={vae_name}, server={server_address}")
        
        # 选择最佳服务器
        selected_server = self.select_best_server(server_address, backup_servers, enable_load_balancing)
        if selected_server is None:
            raise RuntimeError("No available ComfyUI servers found")
        
        latent_shm_obj = None
        output_shm_name = None
        
        try:
            # 1. 创建共享内存存储latent数据
            workflow_start_time = time.time()
            print("Storing latent in shared memory...")
            shm_name, shape, dtype, latent_shm_obj = self.numpy_array_to_shared_memory(latent_numpy)
            shm_data = (shm_name, shape, dtype)
            
            # 创建工作流
            workflow = self.create_workflow(shm_data, vae_name, use_shared_memory_output)
            workflow_time = time.time() - workflow_start_time
            
            # 2. 执行工作流
            execution_start_time = time.time()
            ws = websocket.WebSocket()
            ws.connect("ws://{}/ws?clientId={}".format(selected_server, self.client_id))
            
            print(f"Executing ComfyUI VAE decode workflow on {selected_server}...")
            output_data = self.get_output_shared_memory(ws, workflow, selected_server, use_shared_memory_output)
            ws.close()
            execution_time = time.time() - execution_start_time
            
            # 3. 处理输出结果
            output_start_time = time.time()
            if output_data:
                output_type, data = output_data
                
                if output_type == 'shared_memory':
                    # 从共享内存加载图像数据
                    image_numpy = self.load_image_from_shared_memory(data)
                    output_shm_name = data["shm_name"]
                elif output_type == 'websocket':
                    # 从WebSocket数据解码图像
                    pil_image = Image.open(io.BytesIO(data))
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    image_numpy = np.array(pil_image).astype(np.float32) / 255.0
                else:
                    raise RuntimeError(f"Unknown output type: {output_type}")
                
                # 转换为ComfyUI张量格式
                if len(image_numpy.shape) == 4:
                    # 已经有batch维度
                    image_tensor = torch.from_numpy(image_numpy.astype(np.float32))
                else:
                    # 添加batch维度并确保是float32类型
                    if image_numpy.dtype == np.uint8:
                        image_numpy = image_numpy.astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_numpy.astype(np.float32)).unsqueeze(0)
                
                output_time = time.time() - output_start_time
                total_process_time = time.time() - process_start_time
                
                print(f"\n=== VAE Decode Processing Time Summary ===")
                print(f"  - Server selected: {selected_server}")
                print(f"  - Output mode: {'Shared Memory' if use_shared_memory_output else 'WebSocket'}")
                print(f"  - Workflow creation + latent shared memory setup: {workflow_time:.4f}s")
                print(f"  - ComfyUI execution: {execution_time:.4f}s")
                print(f"  - Output processing: {output_time:.4f}s")
                print(f"  - Total processing time: {total_process_time:.4f}s")
                print(f"  - Output image tensor shape: {image_tensor.shape}")
                print("=" * 50)
                
                return (image_tensor,)
            else:
                raise RuntimeError("No output data received from workflow")
        
        except Exception as e:
            print(f"Error processing VAE decode on {selected_server}: {e}")
            raise
        finally:
            # 清理共享内存
            if latent_shm_obj:
                self.cleanup_shared_memory(shm_object=latent_shm_obj)
            if output_shm_name:
                self.cleanup_shared_memory(shm_name=output_shm_name)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "RemoteVAEDecodeNode": RemoteVAEDecodeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteVAEDecodeNode": "Remote VAE Decode",
} 
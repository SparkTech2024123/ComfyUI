"""
Remote Captioner Node for ComfyUI
包装 captioner_api.py 功能为 ComfyUI 节点，支持远程图像描述生成

主要功能:
- 使用远程 ComfyUI 服务进行图像描述生成
- 支持共享内存高性能数据传输
- 支持负载均衡和服务器选择
- 避免全局变量依赖，每次调用生成独立的 client_id
- 支持所有 JoyCaption 参数配置
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
import copy
import os
import sys

# 添加 ComfyUI 根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
comfyui_root = os.path.dirname(current_dir)
if comfyui_root not in sys.path:
    sys.path.insert(0, comfyui_root)

class RemoteCaptionerNode:
    """
    使用远程 ComfyUI 服务进行图像描述生成的节点
    支持所有原始 captioner_api.py 的功能和参数
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "user_prompt": ("STRING", {
                    "default": "Describe the image in detail.", 
                    "multiline": True,
                    "tooltip": "用户输入的提示文本"
                }),
                "server_address": ("STRING", {
                    "default": "127.0.0.1:8231", 
                    "tooltip": "远程 ComfyUI 服务器地址 (格式: host:port，留空则自动选择最佳服务器)"
                }),
            },
            "optional": {
                "model": ("STRING", {
                    "default": "fancyfeast/llama-joycaption-beta-one-hf-llava",
                    "tooltip": "JoyCaption 模型名称"
                }),
                "quantization_mode": (["nf4", "int8", "fp16", "fp32"], {
                    "default": "nf4",
                    "tooltip": "量化模式"
                }),
                "device": ("STRING", {
                    "default": "cuda:0",
                    "tooltip": "设备 (cuda:0, cuda:1, cpu 等)"
                }),
                "caption_type": (["Descriptive", "Training Prompt", "MidJourney", "Booru tag list", "Booru-like tag list", "Art Critic", "Product Listing", "Social Media Post"], {
                    "default": "Descriptive",
                    "tooltip": "描述类型"
                }),
                "caption_length": (["any", "very short", "short", "medium-length", "long", "very long"], {
                    "default": "any",
                    "tooltip": "描述长度"
                }),
                "max_new_tokens": ("INT", {
                    "default": 512, 
                    "min": 50, 
                    "max": 2048, 
                    "step": 1,
                    "tooltip": "最大新 token 数"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Top-p 采样参数"
                }),
                "top_k": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 100, 
                    "step": 1,
                    "tooltip": "Top-k 采样参数 (0 表示禁用)"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.6, 
                    "min": 0.1, 
                    "max": 2.0, 
                    "step": 0.1,
                    "tooltip": "温度参数，控制生成的随机性"
                }),
                "enable_load_balancing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用负载均衡自动选择最佳服务器"
                }),
                "fallback_servers": ("STRING", {
                    "default": "127.0.0.1:8232,127.0.0.1:8233,127.0.0.1:8234",
                    "tooltip": "备用服务器列表，用逗号分隔"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption_text",)
    FUNCTION = "process"
    CATEGORY = "api/text/remote"
    
    def __init__(self):
        # 内嵌工作流模板，避免外部文件依赖
        self.workflow_template = {
            "10": {
                "inputs": {
                    "model": "fancyfeast/llama-joycaption-beta-one-hf-llava",
                    "quantization_mode": "nf4",
                    "device": "cuda:0"
                },
                "class_type": "LayerUtility: LoadJoyCaptionBeta1Model",
                "_meta": {
                    "title": "LayerUtility: Load JoyCaption Beta One Model (Advance)"
                }
            },
            "11": {
                "inputs": {
                    "caption_type": "Descriptive",
                    "caption_length": "any",
                    "max_new_tokens": 512,
                    "top_p": 0.9,
                    "top_k": 0,
                    "temperature": 0.6,
                    "user_prompt": ["15", 0],
                    "image": ["12", 0],
                    "joycaption_beta1_model": ["10", 0]
                },
                "class_type": "LayerUtility: JoyCaptionBeta1",
                "_meta": {
                    "title": "LayerUtility: JoyCaption Beta One (Advance)"
                }
            },
            "12": {
                "inputs": {
                    "shm_name": "",
                    "shape": "",
                    "dtype": "uint8",
                    "convert_bgr_to_rgb": False
                },
                "class_type": "LoadImageSharedMemory",
                "_meta": {
                    "title": "Load Image (Shared Memory)"
                }
            },
            "13": {
                "inputs": {
                    "text_undefined": "",
                    "text": ["11", 0]
                },
                "class_type": "ShowText|pysssss",
                "_meta": {
                    "title": "Show Text 🐍"
                }
            },
            "15": {
                "inputs": {
                    "text": "Describe the image in detail."
                },
                "class_type": "TextInput_",
                "_meta": {
                    "title": "Text Input ♾️Mixlab"
                }
            }
        }
    
    def process(self, image, user_prompt, server_address="127.0.0.1:8231",
                model="fancyfeast/llama-joycaption-beta-one-hf-llava", 
                quantization_mode="nf4", device="cuda:0",
                caption_type="Descriptive", caption_length="any",
                max_new_tokens=512, top_p=0.9, top_k=0, temperature=0.6,
                enable_load_balancing=True, fallback_servers=""):
        
        start_time = time.time()
        
        # 输入验证
        if image.dim() != 4 or image.shape[0] != 1:
            raise ValueError("Expected image tensor with shape (1, H, W, C)")
        
        # 移除batch维度并转换为numpy
        image_np = image[0].cpu().numpy()
        image_uint8 = (image_np * 255).astype(np.uint8)
        
        # 生成唯一的客户端ID，避免全局变量
        client_id = str(uuid.uuid4())
        
        # 选择服务器
        if enable_load_balancing:
            # 构建服务器列表
            servers = []
            if server_address.strip():
                servers.append(server_address.strip())
            if fallback_servers.strip():
                servers.extend([s.strip() for s in fallback_servers.split(",") if s.strip()])
            
            if not servers:
                servers = ["127.0.0.1:8231"]  # 默认服务器
            
            selected_server = self.select_best_server(servers)
            if selected_server is None:
                raise RuntimeError("No available ComfyUI servers found")
        else:
            selected_server = server_address if server_address.strip() else "127.0.0.1:8231"
        
        shm_obj = None
        
        try:
            # 创建共享内存用于输入
            shm_name, shape, dtype, shm_obj = self.numpy_array_to_shared_memory(image_uint8)
            
            # 创建工作流
            workflow = self.create_workflow(
                shm_name, shape, dtype, user_prompt, model, quantization_mode,
                device, caption_type, caption_length, max_new_tokens, 
                top_p, top_k, temperature
            )
            
            # 执行工作流
            caption_result = self.execute_workflow(workflow, selected_server, client_id)
            
            if not caption_result:
                raise RuntimeError("No caption result received from workflow")
            
            total_time = time.time() - start_time
            print(f"✓ Remote image captioning completed in {total_time:.3f}s on {selected_server}")
            print(f"✓ Generated caption ({len(caption_result)} chars): {caption_result[:100]}...")
            
            return (caption_result,)
            
        except Exception as e:
            print(f"Error in remote image captioning: {e}")
            raise
        finally:
            # 清理共享内存
            if shm_obj:
                self.cleanup_shared_memory(shm_object=shm_obj)
    
    def create_workflow(self, shm_name, shape, dtype, user_prompt, model, quantization_mode,
                       device, caption_type, caption_length, max_new_tokens, 
                       top_p, top_k, temperature):
        """创建工作流配置"""
        workflow = copy.deepcopy(self.workflow_template)
        
        # 配置输入节点 (LoadImageSharedMemory)
        workflow["12"]["inputs"].update({
            "shm_name": shm_name,
            "shape": json.dumps(shape),
            "dtype": dtype,
            "convert_bgr_to_rgb": False
        })
        
        # 配置文本输入节点
        workflow["15"]["inputs"]["text"] = user_prompt
        
        # 配置模型加载节点
        workflow["10"]["inputs"].update({
            "model": model,
            "quantization_mode": quantization_mode,
            "device": device
        })
        
        # 配置 JoyCaption 节点
        workflow["11"]["inputs"].update({
            "caption_type": caption_type,
            "caption_length": caption_length,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature
        })
        
        return workflow
    
    def execute_workflow(self, workflow, server_address, client_id):
        """执行工作流并获取描述结果"""
        
        ws = websocket.WebSocket()
        ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
        
        try:
            # 执行工作流
            caption_result = self.get_caption_result(ws, workflow, server_address, client_id)
            return caption_result
            
        finally:
            ws.close()
    
    def get_caption_result(self, ws, workflow, server_address, client_id):
        """获取图像描述结果（文本输出）"""
        prompt_id = self.queue_prompt(workflow, server_address, client_id)['prompt_id']
        output_text = None
        current_node = ""
        
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
                        
                        # 检查是否有输出数据
                        if 'output' in data and data['output']:
                            node_output = data['output']
                            
                            # 检查 ShowText 节点(节点"13")或 JoyCaptionBeta1 节点(节点"11")的输出
                            if node_id in ["11", "13"]:
                                # 尝试不同的输出格式
                                if 'text' in node_output:
                                    if isinstance(node_output['text'], list) and len(node_output['text']) > 0:
                                        output_text = node_output['text'][0]
                                        print(f"Caption result received from node {node_id}: {output_text[:100]}...")
                                    else:
                                        output_text = str(node_output['text'])
                                        print(f"Caption result (string) from node {node_id}: {output_text[:100]}...")
                                elif 'result' in node_output:
                                    output_text = str(node_output['result'])
                                    print(f"Caption result (result field) from node {node_id}: {output_text[:100]}...")
                                else:
                                    # 尝试获取第一个可用的输出
                                    for key, value in node_output.items():
                                        if isinstance(value, (str, list)):
                                            output_text = str(value[0] if isinstance(value, list) else value)
                                            print(f"Caption result (from {key}) from node {node_id}: {output_text[:100]}...")
                                            break
                                            
                elif message['type'] == 'execution_error':
                    data = message['data']
                    if data.get('prompt_id') == prompt_id:
                        print(f"✗ Execution error: {data}")
                        raise RuntimeError(f"Workflow execution failed: {data}")
                        
                elif message['type'] == 'execution_interrupted':
                    print("✗ Execution interrupted")
                    raise RuntimeError("Workflow execution was interrupted")

        return output_text
    
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
    
    def select_best_server(self, servers):
        """选择最佳的ComfyUI服务器"""
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
    
    def queue_prompt(self, prompt, server_address, client_id):
        """提交工作流到远程服务器"""
        p = {"prompt": prompt, "client_id": client_id}
        
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
        req.add_header('Content-Type', 'application/json')
        try:
            response = urllib.request.urlopen(req)
            result = json.loads(response.read())
            print("got prompt")
            return result
        except urllib.error.HTTPError as e:
            print("Server returned error:", e.read().decode())
            raise

    def numpy_array_to_shared_memory(self, image_array, shm_name=None):
        """将 numpy 数组存储到共享内存中"""
        if shm_name is None:
            shm_name = f"comfyui_img_{uuid.uuid4().hex[:8]}"
        
        if not image_array.flags['C_CONTIGUOUS']:
            image_array = np.ascontiguousarray(image_array)
        
        shm = shared_memory.SharedMemory(create=True, size=image_array.nbytes, name=shm_name)
        shm_array = np.ndarray(image_array.shape, dtype=image_array.dtype, buffer=shm.buf)
        shm_array[:] = image_array[:]
        
        return shm_name, list(image_array.shape), str(image_array.dtype), shm

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

# 节点注册
NODE_CLASS_MAPPINGS = {
    "RemoteCaptionerNode": RemoteCaptionerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteCaptionerNode": "Remote Image Captioner",
} 
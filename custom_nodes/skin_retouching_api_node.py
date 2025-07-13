"""
Remote Skin Retouching Node for ComfyUI
包装 skin_retouching_api.py 功能的 ComfyUI 节点，用于调用远程 ComfyUI 服务进行皮肤美化
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
import tempfile
from PIL import Image
import io
import torch
import time
from multiprocessing import shared_memory
import copy

# 添加 ComfyUI 根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
comfyui_root = os.path.dirname(current_dir)
if comfyui_root not in sys.path:
    sys.path.insert(0, comfyui_root)

class RemoteSkinRetouchingNode:
    """
    使用远程 ComfyUI 服务进行皮肤美化的节点
    支持所有原始 skin_retouching_api.py 的功能和参数
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "server_address": ("STRING", {"default": "127.0.0.1:8204"}),
                "model_path": ("STRING", {"default": "./models/huggingface/cv_unet_skin_retouching_torch"}),
            },
            "optional": {
                "use_shared_memory_output": ("BOOLEAN", {"default": True, "tooltip": "使用共享内存输出(更快)而不是WebSocket"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result_image",) 
    FUNCTION = "process"
    CATEGORY = "api/image/remote"
    
    def __init__(self):
        self.workflow_template = {
            "1": {
                "inputs": {
                    "model_path": "./models/huggingface/cv_unet_skin_retouching_torch"
                },
                "class_type": "SkinRetouchingModelLoader",
                "_meta": {"title": "SkinRetouchingModelLoader"}
            },
            "2": {
                "inputs": {
                    "image": ["3", 0],
                    "skin_retouching_model": ["1", 0]
                },
                "class_type": "SkinRetouchingProcessor",
                "_meta": {"title": "SkinRetouchingProcessor"}
            },
            "3": {
                "inputs": {
                    "shm_name": "",
                    "shape": "",
                    "dtype": "uint8",
                    "convert_bgr_to_rgb": False
                },
                "class_type": "LoadImageSharedMemory",
                "_meta": {"title": "Load Image (Shared Memory)"}
            }
        }
    
    def process(self, image, server_address, model_path="./models/huggingface/cv_unet_skin_retouching_torch",
                use_shared_memory_output=True):
        
        start_time = time.time()
        
        # 输入验证
        if image.dim() != 4 or image.shape[0] != 1:
            raise ValueError("Expected image tensor with shape (1, H, W, C)")
        
        # 移除batch维度并转换为numpy
        image_np = image[0].cpu().numpy()
        image_uint8 = (image_np * 255).astype(np.uint8)
        
        shm_obj = None
        output_shm_obj = None
        
        try:
            # 创建共享内存用于输入
            shm_name, shape, dtype, shm_obj = self.numpy_array_to_shared_memory(image_uint8)
            
            # 创建工作流
            workflow = self.create_workflow(
                shm_name, shape, dtype, model_path, use_shared_memory_output
            )
            
            # 生成唯一的客户端ID
            client_id = str(uuid.uuid4())
            
            # 执行工作流
            if use_shared_memory_output:
                result_image = self.execute_workflow_shared_memory(workflow, server_address, client_id)
            else:
                result_image = self.execute_workflow_websocket(workflow, server_address, client_id)
            
            # 转换回ComfyUI格式
            if result_image.dtype != np.float32:
                result_image = result_image.astype(np.float32) / 255.0
            
            result_tensor = torch.from_numpy(result_image).unsqueeze(0)
            
            total_time = time.time() - start_time
            output_mode = "Shared Memory" if use_shared_memory_output else "WebSocket"
            print(f"✓ Remote skin retouching completed in {total_time:.3f}s using {output_mode}")
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"Error in remote skin retouching: {e}")
            raise
        finally:
            # 清理共享内存
            if shm_obj:
                self.cleanup_shared_memory(shm_object=shm_obj)
            if output_shm_obj:
                self.cleanup_shared_memory(shm_object=output_shm_obj)
    
    def create_workflow(self, shm_name, shape, dtype, model_path, use_shared_memory_output):
        """创建工作流配置"""
        workflow = copy.deepcopy(self.workflow_template)
        
        # 配置输入节点
        workflow["3"]["inputs"].update({
            "shm_name": shm_name,
            "shape": json.dumps(shape),
            "dtype": dtype,
            "convert_bgr_to_rgb": False
        })
        
        # 配置SkinRetouchingModelLoader节点
        workflow["1"]["inputs"]["model_path"] = model_path
        
        # 配置输出节点
        if use_shared_memory_output:
            workflow["save_node"] = {
                "inputs": {
                    "images": ["2", 0],
                    "shm_name": f"output_{uuid.uuid4().hex[:16]}",
                    "convert_rgb_to_bgr": False
                },
                "class_type": "SaveImageSharedMemory",
                "_meta": {"title": "Save Image (Shared Memory)"}
            }
        else:
            workflow["save_node"] = {
                "inputs": {
                    "images": ["2", 0]
                },
                "class_type": "SaveImageWebsocket",
                "_meta": {"title": "Save Image (WebSocket)"}
            }
        
        return workflow
    
    def execute_workflow_shared_memory(self, workflow, server_address, client_id):
        """使用共享内存执行工作流并获取结果"""
        
        # 获取输出共享内存名称
        output_shm_name = workflow["save_node"]["inputs"]["shm_name"]
        
        # 提交工作流
        ws = websocket.WebSocket()
        ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
        
        try:
            # 执行工作流
            shared_memory_info = self.get_shared_memory_result(ws, workflow, client_id, server_address)
            
            if not shared_memory_info:
                raise RuntimeError("No shared memory info received from workflow")
            
            # 从共享内存读取结果
            result_info = shared_memory_info[0]
            shm_name = result_info["shm_name"]
            shape = result_info["shape"]
            dtype = result_info["dtype"]
            
            print(f"✓ Reading result from shared memory: {shm_name}")
            print(f"  - Shape: {shape}")
            print(f"  - Size: {result_info.get('size_mb', 'Unknown')} MB")
            
            # 连接到输出共享内存
            output_shm = shared_memory.SharedMemory(name=shm_name)
            
            # 重建numpy数组
            numpy_dtype = getattr(np, dtype)
            result_array = np.ndarray(shape, dtype=numpy_dtype, buffer=output_shm.buf)
            
            # 复制数据
            result_copy = result_array.copy()
            
            # 清理输出共享内存
            output_shm.close()
            output_shm.unlink()
            
            return result_copy
            
        finally:
            ws.close()
    
    def execute_workflow_websocket(self, workflow, server_address, client_id):
        """使用WebSocket执行工作流并获取结果"""
        
        ws = websocket.WebSocket()
        ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
        
        try:
            # 执行工作流
            images = self.get_images(ws, workflow, client_id, server_address)
            
            if 'save_node' in images:
                output_image_data = images['save_node'][0]
                pil_image = Image.open(io.BytesIO(output_image_data))
                
                # 确保是RGB格式
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                result_array = np.array(pil_image)
                return result_array
            else:
                raise RuntimeError("No output images received from workflow")
                
        finally:
            ws.close()
    
    def get_shared_memory_result(self, ws, workflow, client_id, server_address):
        """执行工作流并获取共享内存信息"""
        prompt_id = self.queue_prompt(workflow, server_address, client_id)['prompt_id']
        shared_memory_info = None
        current_node = ""
        
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
                
                elif message['type'] == 'executed':
                    data = message['data']
                    if data['prompt_id'] == prompt_id:
                        node_id = data['node']
                        if 'shared_memory_info' in data['output']:
                            shared_memory_info = data['output']['shared_memory_info']
                            print(f"✓ Received shared memory info from node {node_id}")
                
                elif message['type'] == 'execution_error':
                    execution_error = message['data']
                    print(f"✗ Execution error: {execution_error}")
                    raise RuntimeError(f"Workflow execution failed: {execution_error}")
                    
                elif message['type'] == 'execution_interrupted':
                    print("✗ Execution interrupted")
                    raise RuntimeError("Workflow execution was interrupted")
        
        return shared_memory_info

    def queue_prompt(self, prompt, server_address, client_id):
        """提交工作流到远程服务器"""
        p = {"prompt": prompt, "client_id": client_id}
        
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
        req.add_header('Content-Type', 'application/json')
        try:
            response = urllib.request.urlopen(req)
            result = json.loads(response.read())
            print("✓ Prompt queued successfully")
            return result
        except urllib.error.HTTPError as e:
            print("Server returned error:", e.read().decode())
            raise

    def get_images(self, ws, workflow, client_id, server_address):
        """获取工作流执行结果"""
        prompt_id = self.queue_prompt(workflow, server_address, client_id)['prompt_id']
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
                    execution_error = message['data']
                    print(f"✗ Execution error: {execution_error}")
                    break
                    
                elif message['type'] == 'execution_interrupted':
                    print("✗ Execution interrupted")
                    break
                    
            else:
                print(f"Received binary data from node: {current_node}, size: {len(out)} bytes")
                if current_node == 'save_node':
                    images_output = output_images.get(current_node, [])
                    images_output.append(out[8:])
                    output_images[current_node] = images_output
                    print(f"✓ Saved image data from {current_node}")

        if execution_error:
            raise RuntimeError(f"Workflow execution failed: {execution_error}")
        
        print(f"Total output images collected: {len(output_images)}")
        return output_images

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
    "RemoteSkinRetouchingNode": RemoteSkinRetouchingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteSkinRetouchingNode": "Remote Skin Retouching",
} 
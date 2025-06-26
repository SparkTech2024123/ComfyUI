"""
ComfyUI Remote Face Swap Node

包装faceswap_api.py功能为ComfyUI节点，支持：
- 双图像输入（源图像和目标图像）
- 远程ComfyUI服务调用
- 共享内存高性能数据传输
- 无全局变量，完全参数化配置
"""

import torch
import numpy as np
import comfy.utils
import time
import json
import uuid
import websocket
import urllib.request
import urllib.parse
import urllib.error
import io
from PIL import Image
from multiprocessing import shared_memory

class RemoteFaceSwapNode:
    """
    远程人脸交换节点
    通过调用远程ComfyUI服务执行人脸交换，避免在本地重复加载模型
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE", {"tooltip": "源图像（提供人脸）"}),
                "target_image": ("IMAGE", {"tooltip": "目标图像（被替换人脸）"}),
                "server_address": ("STRING", {"default": "127.0.0.1:8203", "tooltip": "远程ComfyUI服务器地址:端口"}),
                "swap_own_model": ("STRING", {"default": "faceswap/swapper_own.pth", "tooltip": "人脸交换模型路径"}),
                "arcface_model": ("STRING", {"default": "faceswap/arcface_checkpoint.tar", "tooltip": "ArcFace模型路径"}),
                "detect_model": ("STRING", {"default": "facedetect/scrfd_10g_bnkps_shape640x640.onnx", "tooltip": "人脸检测模型路径"}),
                "device": (["cuda:0", "cuda:1", "cpu"], {"default": "cuda:0", "tooltip": "计算设备"}),
            },
            "optional": {
                "use_shared_memory_output": ("BOOLEAN", {"default": True, "tooltip": "是否使用共享内存输出（本地高性能模式）"}),
            }
        }

    CATEGORY = "api/face"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("swapped_image",)
    FUNCTION = "process_faceswap"
    
    # 内嵌的工作流模板，基于A-faceswap-api.json
    WORKFLOW_TEMPLATE = {
        "2": {
            "inputs": {
                "swap_own_model": "faceswap/swapper_own.pth",
                "arcface_model": "faceswap/arcface_checkpoint.tar",
                "facealign_config_dir": "face_align",
                "phase1_model": "facealign/p1.pt",
                "phase2_model": "facealign/p2.pt",
                "device": "cuda:0"
            },
            "class_type": "FaceSwapPipeBuilder",
            "_meta": {"title": "FaceSwapPipeBuilder"}
        },
        "8": {
            "inputs": {
                "detect_model_path": "facedetect/scrfd_10g_bnkps_shape640x640.onnx",
                "deca_dir": "deca",
                "gpu_choose": "cuda:0"
            },
            "class_type": "FaceWarpPipeBuilder",
            "_meta": {"title": "FaceWarpPipeBuilder"}
        },
        "12": {
            "inputs": {
                "image": "source_image.jpg"
            },
            "class_type": "LoadImage",
            "_meta": {"title": "Load Source Image"}
        },
        "13": {
            "inputs": {
                "image": "target_image.jpg"
            },
            "class_type": "LoadImage", 
            "_meta": {"title": "Load Target Image"}
        },
        "6": {
            "inputs": {
                "image": ["12", 0]
            },
            "class_type": "ConvertTensorToNumpy",
            "_meta": {"title": "Convert Source to Numpy"}
        },
        "10": {
            "inputs": {
                "image": ["13", 0]
            },
            "class_type": "ConvertTensorToNumpy",
            "_meta": {"title": "Convert Target to Numpy"}
        },
        "5": {
            "inputs": {
                "model": ["8", 0],
                "image": ["6", 0]
            },
            "class_type": "FaceWarpDetectFacesMethod",
            "_meta": {"title": "Detect Faces in Source"}
        },
        "9": {
            "inputs": {
                "model": ["8", 0],
                "image": ["10", 0]
            },
            "class_type": "FaceWarpDetectFacesMethod",
            "_meta": {"title": "Detect Faces in Target"}
        },
        "1": {
            "inputs": {
                "model": ["2", 0],
                "src_image": ["6", 0],
                "src_faces": ["5", 0],
                "ptstype": "256"
            },
            "class_type": "FaceSwapDetectPts",
            "_meta": {"title": "Detect Source Points 256"}
        },
        "3": {
            "inputs": {
                "model": ["2", 0],
                "src_image": ["6", 0],
                "src_faces": ["5", 0],
                "ptstype": "5"
            },
            "class_type": "FaceSwapDetectPts",
            "_meta": {"title": "Detect Source Points 5"}
        },
        "4": {
            "inputs": {
                "model": ["2", 0],
                "src_image": ["10", 0],
                "src_faces": ["9", 0],
                "ptstype": "5"
            },
            "class_type": "FaceSwapDetectPts",
            "_meta": {"title": "Detect Target Points 5"}
        },
        "7": {
            "inputs": {
                "model": ["2", 0],
                "src_image": ["10", 0],
                "two_stage_image": ["6", 0],
                "source_5pts": ["4", 0],
                "target_5pts": ["3", 0],
                "target_256pts": ["1", 0]
            },
            "class_type": "FaceSwapMethod",
            "_meta": {"title": "Face Swap Method"}
        },
        "11": {
            "inputs": {
                "image": ["7", 0]
            },
            "class_type": "ConvertNumpyToTensor",
            "_meta": {"title": "Convert Result to Tensor"}
        }
    }
    
    def numpy_array_to_shared_memory(self, image_array, shm_name=None):
        """将numpy数组存储到共享内存中"""
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError(f"Expected image array shape (h, w, 3), got {image_array.shape}")
        
        if shm_name is None:
            shm_name = f"faceswap_img_{uuid.uuid4().hex[:8]}"
        
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
    
    def queue_prompt(self, prompt, server_address, client_id):
        """提交工作流到远程服务器"""
        p = {"prompt": prompt, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
        req.add_header('Content-Type', 'application/json')
        try:
            return json.loads(urllib.request.urlopen(req).read())
        except urllib.error.HTTPError as e:
            print("Server returned error:", e.read().decode())
            raise
    
    def get_images(self, ws, prompt, server_address, client_id, use_shared_memory_output=True):
        """执行工作流并获取结果图像"""
        prompt_id = self.queue_prompt(prompt, server_address, client_id)['prompt_id']
        output_images = {}
        current_node = ""
        execution_error = None
        
        print(f"Waiting for face swap execution of prompt {prompt_id}...")
        
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                
                if message['type'] == 'executing':
                    data = message['data']
                    if data['prompt_id'] == prompt_id:
                        if data['node'] is None:
                            print("✓ Face swap execution completed")
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
                
                # 处理SaveImageSharedMemory的UI输出
                elif message['type'] == 'executed' and use_shared_memory_output:
                    data = message['data']
                    if data['node'] == 'save_image_shared_memory_node' and 'output' in data:
                        ui_output = data['output']
                        if 'shared_memory_info' in ui_output:
                            print(f"Received shared memory info from {data['node']}")
                            images_output = output_images.get('save_image_shared_memory_node', [])
                            # 将共享内存信息存储为JSON字符串
                            shm_info_json = json.dumps(ui_output['shared_memory_info'][0])
                            images_output.append(shm_info_json.encode('utf-8'))
                            output_images['save_image_shared_memory_node'] = images_output
            else:
                if use_shared_memory_output and current_node == 'save_image_shared_memory_node':
                    print(f"Received binary data from {current_node}, size: {len(out)} bytes")
                    images_output = output_images.get(current_node, [])
                    images_output.append(out[8:])
                    output_images[current_node] = images_output
                elif not use_shared_memory_output and current_node == 'save_image_websocket_node':
                    print(f"Received binary data from {current_node}, size: {len(out)} bytes")
                    images_output = output_images.get(current_node, [])
                    images_output.append(out[8:])
                    output_images[current_node] = images_output
        
        if execution_error:
            raise RuntimeError(f"Face swap workflow execution failed: {execution_error}")
        
        return output_images
    
    def modify_workflow_for_shared_memory_input(self, workflow, source_shm_data, target_shm_data,
                                               swap_own_model, arcface_model, detect_model, device,
                                               use_shared_memory_output=True):
        """修改工作流以使用共享内存输入和输出"""
        modified_workflow = workflow.copy()
        
        # 1. 替换源图像LoadImage节点为LoadImageSharedMemory节点
        modified_workflow["12"] = {
            "inputs": {
                "shm_name": source_shm_data[0],
                "shape": json.dumps(source_shm_data[1]),
                "dtype": source_shm_data[2],
                "convert_bgr_to_rgb": False
            },
            "class_type": "LoadImageSharedMemory",
            "_meta": {"title": "Load Source Image (Shared Memory)"}
        }
        
        # 2. 替换目标图像LoadImage节点为LoadImageSharedMemory节点
        modified_workflow["13"] = {
            "inputs": {
                "shm_name": target_shm_data[0],
                "shape": json.dumps(target_shm_data[1]),
                "dtype": target_shm_data[2],
                "convert_bgr_to_rgb": False
            },
            "class_type": "LoadImageSharedMemory",
            "_meta": {"title": "Load Target Image (Shared Memory)"}
        }
        
        # 3. 更新模型参数
        modified_workflow["2"]["inputs"]["swap_own_model"] = swap_own_model
        modified_workflow["2"]["inputs"]["arcface_model"] = arcface_model
        modified_workflow["2"]["inputs"]["device"] = device
        
        modified_workflow["8"]["inputs"]["detect_model_path"] = detect_model
        modified_workflow["8"]["inputs"]["gpu_choose"] = device
        
        # 4. 添加输出节点
        if use_shared_memory_output:
            modified_workflow["save_image_shared_memory_node"] = {
                "inputs": {
                    "images": ["11", 0],
                    "output_format": "RGB",
                    "convert_rgb_to_bgr": False
                },
                "class_type": "SaveImageSharedMemory",
                "_meta": {"title": "Save Image (Shared Memory)"}
            }
        else:
            modified_workflow["save_image_websocket_node"] = {
                "inputs": {
                    "images": ["11", 0]
                },
                "class_type": "SaveImageWebsocket",
                "_meta": {"title": "Save Image (WebSocket)"}
            }
        
        return modified_workflow
    
    def create_faceswap_workflow_with_shared_memory(self, source_image_array, target_image_array,
                                                   swap_own_model, arcface_model, detect_model, device,
                                                   use_shared_memory_output=True):
        """创建带共享内存输入的人脸交换工作流"""
        print("=== Converting images to shared memory ===")
        
        # 存储图像到共享内存
        source_shm_name, source_shape, source_dtype, source_shm_obj = self.numpy_array_to_shared_memory(source_image_array)
        target_shm_name, target_shape, target_dtype, target_shm_obj = self.numpy_array_to_shared_memory(target_image_array)
        
        source_shm_data = (source_shm_name, source_shape, source_dtype)
        target_shm_data = (target_shm_name, target_shape, target_dtype)
        shm_objects = [source_shm_obj, target_shm_obj]
        
        # 修改工作流
        modified_workflow = self.modify_workflow_for_shared_memory_input(
            self.WORKFLOW_TEMPLATE, source_shm_data, target_shm_data,
            swap_own_model, arcface_model, detect_model, device, use_shared_memory_output
        )
        
        return modified_workflow, shm_objects
    
    def process_faceswap(self, source_image, target_image, server_address,
                        swap_own_model, arcface_model, detect_model, device,
                        use_shared_memory_output=True):
        """执行人脸交换处理"""
        start_time = time.time()
        
        try:
            # 转换输入图像为numpy数组
            source_numpy = (source_image[0].cpu().numpy() * 255).astype(np.uint8)
            target_numpy = (target_image[0].cpu().numpy() * 255).astype(np.uint8)
            
            print(f"Face swap processing - Source: {source_numpy.shape}, Target: {target_numpy.shape}")
            print(f"Server: {server_address}, Model: {swap_own_model}, Device: {device}")
            
            # 生成唯一的客户端ID
            client_id = str(uuid.uuid4())
            
            # 创建工作流
            workflow, shm_objects = self.create_faceswap_workflow_with_shared_memory(
                source_numpy, target_numpy, swap_own_model, arcface_model, 
                detect_model, device, use_shared_memory_output
            )
            
            # 连接WebSocket并执行
            ws = websocket.WebSocket()
            ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
            
            images = self.get_images(ws, workflow, server_address, client_id, use_shared_memory_output)
            ws.close()
            
            # 处理输出
            output_node = 'save_image_shared_memory_node' if use_shared_memory_output else 'save_image_websocket_node'
            
            if output_node in images:
                if use_shared_memory_output:
                    # 解析共享内存信息
                    import json
                    shm_info = json.loads(images[output_node][0].decode('utf-8'))
                    
                    # 从共享内存读取结果
                    try:
                        shm = shared_memory.SharedMemory(name=shm_info['shm_name'])
                        result_array = np.ndarray(shm_info['shape'], dtype=np.dtype(shm_info['dtype']), buffer=shm.buf)
                        result_copy = result_array.copy()
                        shm.close()
                        
                        # 清理远程共享内存
                        try:
                            temp_shm = shared_memory.SharedMemory(name=shm_info['shm_name'])
                            temp_shm.close()
                            temp_shm.unlink()
                        except:
                            pass
                            
                        print(f"✓ Retrieved result from shared memory: {shm_info['shm_name']}")
                    except Exception as e:
                        print(f"Error reading shared memory: {e}")
                        raise
                else:
                    # WebSocket输出
                    pil_image = Image.open(io.BytesIO(images[output_node][0]))
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    result_copy = np.array(pil_image)
                
                # 转换为torch tensor
                result_normalized = result_copy.astype(np.float32) / 255.0
                result_tensor = torch.from_numpy(result_normalized).unsqueeze(0)
                
                total_time = time.time() - start_time
                transfer_method = "Shared Memory" if use_shared_memory_output else "WebSocket"
                print(f"✓ Face swap completed in {total_time:.3f}s using {transfer_method}")
                
                return (result_tensor,)
            else:
                raise RuntimeError("No output images received from face swap workflow")
                
        except Exception as e:
            print(f"Error in face swap processing: {e}")
            raise
        finally:
            # 清理本地共享内存
            if 'shm_objects' in locals():
                for shm_obj in shm_objects:
                    self.cleanup_shared_memory(shm_object=shm_obj)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "RemoteFaceSwapNode": RemoteFaceSwapNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteFaceSwapNode": "Remote Face Swap",
}
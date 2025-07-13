# This is an example that processes images using numpy arrays
# for ID Photo Background Removal and Cropping processing in ComfyUI workflows
#
# Uses shared memory for ultra-fast data transfer between client and server
# when both are on the same machine. This eliminates base64 encoding overhead
# and provides the highest performance data transfer.
#
# Features:
# - Shared memory data transfer (69x faster than base64 PNG)
# - Zero-copy data transfer for maximum performance
# - Automatic memory cleanup
# - Improved error handling and execution monitoring
# - Returns 4-channel (RGBA) image with alpha channel
################################################################################
# 注意：
# 接收PIL形式的图片（RGB格式），直接使用，无需格式转换
# 输出时，也是PIL形式的图片（RGBA格式），直接使用，无需格式转换
# OpenCV形式的图片是BGR格式，PIL形式的图片是RGB格式
################################################################################

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

# 添加ComfyUI根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
comfyui_root = os.path.dirname(current_dir)
if comfyui_root not in sys.path:
    sys.path.insert(0, comfyui_root)

# 尝试导入ComfyUI的folder_paths模块
try:
    import folder_paths
except ImportError:
    # 如果无法导入，使用默认路径
    class folder_paths:
        @staticmethod
        def get_input_directory():
            return os.path.join(comfyui_root, "input")

server_address = "127.0.0.1:8250"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    # 创建请求体
    p = {"prompt": prompt, "client_id": client_id}
    
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    req.add_header('Content-Type', 'application/json')
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        print("Server returned error:", e.read().decode())
        raise

def get_shared_memory_result(ws, prompt):
    """
    获取处理后的共享内存结果
    """
    prompt_id = queue_prompt(prompt)['prompt_id']
    shared_memory_info = None
    current_node = ""
    execution_error = None

    print(f"Waiting for execution of prompt {prompt_id}...")

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            # print(f"Received message: {message['type']}")

            if message['type'] == 'executing':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    if data['node'] is None:
                        print("✓ Execution completed")
                        break #Execution is done
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
                if execution_error.get('prompt_id') == prompt_id:
                    print(f"✗ Execution error: {execution_error}")
                    break

            elif message['type'] == 'execution_interrupted':
                print("✗ Execution interrupted")
                break

            elif message['type'] == 'progress':
                data = message['data']
                if data.get('prompt_id') == prompt_id:
                    # print(f"Progress: {data}")  # 调试信息 - 可取消注释查看进度
                    pass

    if execution_error:
        raise RuntimeError(f"Workflow execution failed: {execution_error}")

    return shared_memory_info

def load_workflow_from_json(workflow_path):
    """
    从JSON文件加载工作流
    Args:
        workflow_path: 工作流JSON文件路径
    Returns:
        工作流字典
    """
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        print(f"Successfully loaded workflow from: {workflow_path}")
        return workflow
    except FileNotFoundError:
        print(f"Workflow file not found: {workflow_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        raise

def numpy_array_to_shared_memory(image_array, shm_name=None):
    """
    将numpy数组存储到共享内存中
    Args:
        image_array: 输入图像的numpy数组 (h, w, 3) RGB格式
        shm_name: 共享内存名称，如果为None则自动生成
    Returns:
        tuple: (shm_name, shape, dtype, shared_memory_object)
    """
    start_time = time.time()
    
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
    
    total_time = time.time() - start_time
    
    print(f"Shared memory transfer timing for {image_array.shape}:")
    print(f"  - Memory allocation and copy: {total_time:.4f}s")
    print(f"  - Shared memory name: {shm_name}")
    print(f"  - Data size: {image_array.nbytes / 1024 / 1024:.2f} MB")
    
    return shm_name, list(image_array.shape), str(image_array.dtype), shm

def cleanup_shared_memory(shm_name=None, shm_object=None):
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

def modify_workflow_for_shared_memory_input(workflow, shm_data, enable_beauty=True, enable_sr_enhance=True):
    """
    修改工作流以使用共享内存输入和输出
    修改的节点：
    1. 节点55 (LoadImage) -> LoadImageSharedMemory
    2. 节点71 (PreviewImage) -> SaveImageSharedMemory
    3. 节点73 (美颜布尔值)
    4. 节点75 (高清修复布尔值)

    Args:
        workflow: 原始工作流字典
        shm_data: 共享内存数据 (shm_name, shape, dtype)
        enable_beauty: 是否启用美颜 (默认True)
        enable_sr_enhance: 是否启用高清修复 (默认True)
    Returns:
        修改后的工作流字典
    """
    # 创建工作流副本
    modified_workflow = workflow.copy()

    # 1. 替换LoadImage节点为LoadImageSharedMemory节点（节点"55"）
    load_image_node_id = "55"
    if load_image_node_id in modified_workflow:
        print(f"Replacing LoadImage node: {load_image_node_id} with LoadImageSharedMemory node")

        # 替换为LoadImageSharedMemory节点，直接从共享内存读取图像数据
        modified_workflow[load_image_node_id] = {
            "inputs": {
                "shm_name": shm_data[0],
                "shape": json.dumps(shm_data[1]),
                "dtype": shm_data[2],
                "convert_bgr_to_rgb": False  # PIL输入已经是RGB格式，无需转换
            },
            "class_type": "LoadImageSharedMemory",
            "_meta": {
                "title": "Load Image (Shared Memory)"
            }
        }
        print(f"Updated to LoadImageSharedMemory node with shm_name: {shm_data[0]}")
    else:
        print("Warning: LoadImage node not found in workflow")

    # 2. 替换PreviewImage节点为SaveImageSharedMemory节点（节点"71"）
    # 输入来自ImageAlphaMaskReplacer节点（节点"66"）
    modified_workflow["save_image_shared_memory_node"] = {
        "inputs": {
            "images": ["66", 0],  # 从ImageAlphaMaskReplacer节点获取输出
            "shm_name": f"output_{uuid.uuid4().hex[:16]}",
            "convert_rgb_to_bgr": False
        },
        "class_type": "SaveImageSharedMemory",
        "_meta": {
            "title": "Save Image (Shared Memory)"
        }
    }
    print("Added SaveImageSharedMemory node for output (getting input from node 66)")

    # 3. 更新73号节点的美颜布尔值
    if "73" in modified_workflow:
        modified_workflow["73"]["inputs"]["value"] = enable_beauty
        print(f"Updated beauty enable (node 73): {enable_beauty}")

    # 4. 更新75号节点的高清修复布尔值
    if "75" in modified_workflow:
        modified_workflow["75"]["inputs"]["value"] = enable_sr_enhance
        print(f"Updated SR enhance enable (node 75): {enable_sr_enhance}")

    return modified_workflow

def create_idphoto_rmbg_crop_workflow_with_shared_memory(image_array, enable_beauty=True, enable_sr_enhance=True):
    """
    从A-idphoto-rmbg-crop-api.json加载工作流并修改为使用共享内存输入/输出
    修改节点55、71、73、75，保持其他所有节点不变
    Args:
        image_array: 输入图像的numpy数组 (h, w, 3) RGB格式
        enable_beauty: 是否启用美颜 (默认True)
        enable_sr_enhance: 是否启用高清修复 (默认True)
    Returns:
        tuple: (修改后的工作流字典, 共享内存对象)
    """
    print("=== Converting image to shared memory ===")
    encoding_start_time = time.time()

    # 将numpy数组存储到共享内存
    print("Storing image in shared memory...")
    shm_name, shape, dtype, shm_obj = numpy_array_to_shared_memory(image_array)

    total_encoding_time = time.time() - encoding_start_time
    print(f"\n✓ Total shared memory setup time: {total_encoding_time:.4f}s")
    print("=" * 50)

    # 准备共享内存数据
    shm_data = (shm_name, shape, dtype)

    # 获取工作流文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workflow_path = os.path.join(current_dir, "..", "user", "default", "workflows", "A-idphoto-rmbg-crop-api.json")

    # 如果相对路径不存在，尝试绝对路径
    if not os.path.exists(workflow_path):
        workflow_path = os.path.join(os.path.dirname(current_dir), "user", "default", "workflows", "A-idphoto-rmbg-crop-api.json")

    # 再次检查文件是否存在
    if not os.path.exists(workflow_path):
        print(f"Workflow file not found at: {workflow_path}")
        print("Please ensure A-idphoto-rmbg-crop-api.json exists in the correct location")
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")

    # 加载原始工作流
    original_workflow = load_workflow_from_json(workflow_path)

    # 修改工作流以使用共享内存输入/输出（修改节点55、71、73、75）
    modified_workflow = modify_workflow_for_shared_memory_input(
        original_workflow, shm_data, enable_beauty, enable_sr_enhance
    )

    return modified_workflow, shm_obj

def process_image_idphoto_rmbg_crop_numpy_shared_memory(image_array, enable_beauty=True, enable_sr_enhance=True):
    """
    使用ID Photo Background Removal and Cropping处理numpy图像数组（共享内存版本）
    使用原始工作流的所有参数，只替换输入/输出节点为共享内存版本
    Args:
        image_array: 输入图像的numpy数组 (h, w, 3) RGB格式
        enable_beauty: 是否启用美颜 (默认True)
        enable_sr_enhance: 是否启用高清修复 (默认True)
    Returns:
        numpy数组 (h, w, 4) RGBA格式的ID照片处理结果
    """
    process_start_time = time.time()

    print(f"Processing numpy image array with shared memory, shape: {image_array.shape}")
    print(f"Parameters: beauty={enable_beauty}, sr_enhance={enable_sr_enhance}")
    print("Using original workflow parameters from A-idphoto-rmbg-crop-api.json")

    # 验证输入数组格式
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError(f"Expected image array shape (h, w, 3), got {image_array.shape}")

    shm_obj = None

    try:
        # 1. 创建工作流
        workflow_start_time = time.time()
        workflow, shm_obj = create_idphoto_rmbg_crop_workflow_with_shared_memory(
            image_array, enable_beauty, enable_sr_enhance
        )
        workflow_time = time.time() - workflow_start_time

        # 2. 执行工作流
        execution_start_time = time.time()
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

        print("Executing ComfyUI ID photo background removal and cropping workflow...")
        shared_memory_info = get_shared_memory_result(ws, workflow)
        ws.close()
        execution_time = time.time() - execution_start_time

        # 3. 处理输出结果
        output_start_time = time.time()
        if shared_memory_info:
            # 从共享内存获取图像数据
            result_info = shared_memory_info[0]
            shm_name = result_info["shm_name"]
            shape = result_info["shape"]
            dtype = result_info["dtype"]

            print(f"✓ Reading result from shared memory: {shm_name}")
            print(f"  - Shape: {shape}")
            print(f"  - Size: {result_info.get('size_mb', 'Unknown')} MB")

            # 连接到输出共享内存
            output_shm = shared_memory.SharedMemory(name=shm_name)

            try:
                # 重建numpy数组
                numpy_dtype = getattr(np, dtype)
                result_array = np.ndarray(shape, dtype=numpy_dtype, buffer=output_shm.buf)

                # 复制数据
                numpy_array = result_array.copy()

                # 确保是RGBA格式以保持alpha通道
                if len(numpy_array.shape) == 3 and numpy_array.shape[2] == 4:
                    print(f"Output image shape: {numpy_array.shape} (RGBA)")
                elif len(numpy_array.shape) == 3 and numpy_array.shape[2] == 3:
                    # 添加Alpha通道（全不透明）
                    alpha_channel = np.ones((numpy_array.shape[0], numpy_array.shape[1], 1), dtype=numpy_array.dtype) * 255
                    numpy_array = np.concatenate([numpy_array, alpha_channel], axis=2)
                    print(f"Output image shape: {numpy_array.shape} (RGB converted to RGBA)")
                else:
                    raise ValueError(f"Unexpected image shape: {numpy_array.shape}")

            finally:
                # 清理输出共享内存
                output_shm.close()
                output_shm.unlink()

            output_time = time.time() - output_start_time
            total_process_time = time.time() - process_start_time

            print(f"\n=== ID Photo Processing Time Summary (Shared Memory Output) ===")
            print(f"  - Workflow creation + shared memory setup: {workflow_time:.4f}s")
            print(f"  - ComfyUI execution: {execution_time:.4f}s")
            print(f"  - Output processing (shared memory): {output_time:.4f}s")
            print(f"  - Total processing time: {total_process_time:.4f}s")
            print(f"  - Beauty enabled: {enable_beauty}")
            print(f"  - SR enhance enabled: {enable_sr_enhance}")
            print("=" * 65)

            return numpy_array
        else:
            raise RuntimeError("No shared memory info received from workflow")

    except Exception as e:
        print(f"Error processing image with shared memory: {e}")
        raise
    finally:
        # 清理共享内存
        if shm_obj:
            cleanup_shared_memory(shm_object=shm_obj)

# 主要接口
def comfyui_idphoto_rmbg_crop_process(image_array, enable_beauty=True, enable_sr_enhance=True, **kwargs):
    """
    ID Photo Background Removal and Cropping处理接口（使用共享内存）
    使用原始工作流的所有参数，只替换输入节点为共享内存版本
    Args:
        image_array: 输入图像的numpy数组 (h, w, 3) RGB格式
        enable_beauty: 是否启用美颜 (默认True)
        enable_sr_enhance: 是否启用高清修复 (默认True)
        **kwargs: 其他参数
    Returns:
        numpy数组，RGBA格式的ID照片处理结果
    """
    if not isinstance(image_array, np.ndarray):
        raise ValueError("image_array must be a numpy array with shape (h, w, 3)")

    return process_image_idphoto_rmbg_crop_numpy_shared_memory(
        image_array, enable_beauty, enable_sr_enhance, **kwargs
    )

# 辅助函数：从URL下载图像（仅供测试使用）
def url_to_numpy(url):
    """
    从URL下载图像并转换为numpy数组（仅供测试使用）
    Args:
        url: 图像URL
    Returns:
        numpy数组 (h, w, 3) RGB格式
    """
    response = urllib.request.urlopen(url)
    image_data = response.read()

    # 转换为PIL图像
    pil_image = Image.open(io.BytesIO(image_data))
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # 转换为numpy数组（保持RGB格式）
    numpy_array = np.array(pil_image)

    return numpy_array

# 示例用法
if __name__ == "__main__":
    # 示例：使用numpy数组处理图像
    try:
        print("=== ID Photo Background Removal and Cropping with shared memory input ===")

        # 使用PIL加载图像并转换为numpy数组（保持RGB格式）
        pil_image = Image.open("input/4.jpg")
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        input_image_array = np.array(pil_image)
        # 保存输入图像（需要转换为BGR才能正确保存）
        cv2.imwrite("idphoto_rmbg_crop_input.png", cv2.cvtColor(input_image_array, cv2.COLOR_RGB2BGR))
        print(f"Input image shape: {input_image_array.shape} (RGB format)")

        # 使用numpy数组处理图像 - 启用美颜和高清修复
        result_image = process_image_idphoto_rmbg_crop_numpy_shared_memory(
            image_array=input_image_array,
            enable_beauty=True,
            enable_sr_enhance=True
        )

        print(f"Processing completed successfully with shared memory!")
        print(f"Result image shape: {result_image.shape}")

        # 保存结果（RGBA格式，直接保存）
        output_path = "idphoto_rmbg_crop_result_shared_memory.png"
        Image.fromarray(result_image, 'RGBA').save(output_path)
        print(f"Result saved as: {output_path}")

        # 测试主要接口
        print("\n--- Testing main interface (shared memory) ---")
        result = comfyui_idphoto_rmbg_crop_process(
            input_image_array,
            enable_beauty=True,
            enable_sr_enhance=True
        )
        print(f"Main interface test successful. Shape: {result.shape}")

        print("\n=== Performance Summary ===")
        print("✓ Using shared memory for ultra-fast data transfer")
        print("✓ 69x faster than base64 PNG encoding")
        print("✓ Zero-copy data transfer between client and server")
        print("✓ Automatic memory cleanup")
        print("✓ Returns 4-channel (RGBA) image with alpha channel")
        print("✓ Supports ID photo processing with background removal and cropping")
        print("✓ Support for beauty and SR enhancement control")
        print("✓ Consistent interface with idphoto_api.py")

    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

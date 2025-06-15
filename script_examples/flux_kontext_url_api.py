#This is an example that uses LoadImagesFromURL node from comfyui-mixlab-nodes 
#to handle image URLs directly in ComfyUI workflows for Flux Kontext processing

import websocket
import uuid
import json
import urllib.request
import urllib.parse
import urllib.error
import numpy as np
import cv2
import os
from PIL import Image
import io

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt, comfy_api_key=None):
    # 创建请求体，包含认证信息
    p = {"prompt": prompt, "client_id": client_id}
    
    # 如果提供了 API Key，添加到 extra_data 中
    if comfy_api_key:
        p["extra_data"] = {
            "api_key_comfy_org": comfy_api_key
        }
    
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    req.add_header('Content-Type', 'application/json')
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        print("Server returned error:", e.read().decode())
        raise

def get_images(ws, prompt, comfy_api_key=None):
    prompt_id = queue_prompt(prompt, comfy_api_key)['prompt_id']
    output_images = {}
    current_node = ""
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    if data['node'] is None:
                        break #Execution is done
                    else:
                        current_node = data['node']
        else:
            if current_node == 'save_image_websocket_node':
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                output_images[current_node] = images_output

    return output_images

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

def modify_workflow_for_url_input(workflow, image_url, prompt_text, guidance=3, steps=50, seed=None):
    """
    修改工作流以使用LoadImagesFromURL节点和自定义文本
    Args:
        workflow: 原始工作流字典
        image_url: 输入图像URL
        prompt_text: 提示文本
        guidance: 引导强度 (默认3)
        steps: 步数 (默认50)
        seed: 随机种子 (None为随机)
        comfy_api_key: Comfy API Key用于认证
    Returns:
        修改后的工作流字典
    """
    # 创建工作流副本
    modified_workflow = workflow.copy()
    
    # 1. 添加LoadImagesFromURL节点替换LoadImage
    modified_workflow["url_loader"] = {
        "inputs": {
            "url": image_url
        },
        "class_type": "LoadImagesFromURL",
        "_meta": {
            "title": "Load Image From URL"
        }
    }
    
    # 2. 找到并替换LoadImage节点（节点"11"）
    load_image_node_id = "11"  # 根据flux-kontext-api.json
    if load_image_node_id in modified_workflow:
        print(f"Replacing LoadImage node: {load_image_node_id}")
        # 删除原来的LoadImage节点
        del modified_workflow[load_image_node_id]
    else:
        print("Warning: LoadImage node not found in workflow")
    
    # 3. 更新所有引用原LoadImage节点的地方
    for node_id, node_data in modified_workflow.items():
        if "inputs" in node_data:
            for input_key, input_value in node_data["inputs"].items():
                # 检查是否引用了原LoadImage节点
                if isinstance(input_value, list) and len(input_value) == 2:
                    if input_value[0] == load_image_node_id:
                        # 替换为url_loader节点
                        node_data["inputs"][input_key] = ["url_loader", input_value[1]]
                        print(f"Updated reference in node {node_id}, input {input_key}")
    
    # 4. 更新文本输入（节点"22"）
    if "22" in modified_workflow:
        modified_workflow["22"]["inputs"]["text"] = prompt_text
        print(f"Updated text input to: {prompt_text}")
    
    # 5. 更新FluxKontextProImageNode的参数（节点"10"）
    if "10" in modified_workflow:
        if guidance is not None:
            modified_workflow["10"]["inputs"]["guidance"] = guidance
        if steps is not None:
            modified_workflow["10"]["inputs"]["steps"] = steps
        if seed is not None:
            modified_workflow["10"]["inputs"]["seed"] = seed
        print(f"Updated FluxKontext parameters: guidance={guidance}, steps={steps}, seed={seed}")
    
    # 6. 添加保存节点以便获取输出
    modified_workflow["save_image_websocket_node"] = {
        "inputs": {
            "images": ["10", 0]  # 从FluxKontextProImageNode获取输出
        },
        "class_type": "SaveImageWebsocket",
        "_meta": {
            "title": "Save Image (WebSocket)"
        }
    }
    
    return modified_workflow

def create_flux_kontext_workflow_with_url(image_url, prompt_text, guidance=3, steps=50, seed=None):
    """
    从flux-kontext-api.json加载工作流并修改为使用URL输入
    Args:
        image_url: 输入图像的URL
        prompt_text: 提示文本
        guidance: 引导强度
        steps: 生成步数
        seed: 随机种子
        comfy_api_key: Comfy API Key用于认证
    Returns:
        修改后的工作流字典
    """
    # 获取工作流文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workflow_path = os.path.join(current_dir, "..", "user", "default", "workflows", "flux-kontext-api.json")
    
    # 如果相对路径不存在，尝试绝对路径
    if not os.path.exists(workflow_path):
        workflow_path = os.path.join(os.path.dirname(current_dir), "user", "default", "workflows", "flux-kontext-api.json")
    
    # 再次检查文件是否存在
    if not os.path.exists(workflow_path):
        print(f"Workflow file not found at: {workflow_path}")
        print("Please ensure flux-kontext-api.json exists in the correct location")
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
    
    # 加载原始工作流
    original_workflow = load_workflow_from_json(workflow_path)
    
    # 修改工作流以使用URL输入
    modified_workflow = modify_workflow_for_url_input(
        original_workflow, image_url, prompt_text, guidance, steps, seed
    )
    
    return modified_workflow

def process_image_with_flux_kontext(image_url, prompt_text, guidance=3, steps=50, seed=None, comfy_api_key=None):
    """
    使用Flux Kontext处理图像
    Args:
        image_url: 输入图像的URL
        prompt_text: 提示文本
        guidance: 引导强度 (默认3)
        steps: 生成步数 (默认50)
        seed: 随机种子 (None为随机)
        comfy_api_key: Comfy API Key用于认证
    Returns:
        numpy数组 (h, w, 3) BGR格式
    """
    print(f"Processing image from URL: {image_url}")
    print(f"Using prompt: {prompt_text}")
    print(f"Parameters: guidance={guidance}, steps={steps}, seed={seed}")
    
    if comfy_api_key is None:
        print("Warning: No API key provided. This may cause authentication errors.")
    else:
        print("API key provided for authentication.")
    
    # 1. 创建工作流
    workflow = create_flux_kontext_workflow_with_url(image_url, prompt_text, guidance, steps, seed)
    
    # 2. 执行工作流
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    
    print("Executing ComfyUI workflow...")
    images = get_images(ws, workflow, comfy_api_key)
    ws.close()
    
    # 3. 处理输出结果
    if 'save_image_websocket_node' in images:
        # 从WebSocket输出获取图像数据
        output_image_data = images['save_image_websocket_node'][0]
        
        # 转换为numpy数组
        pil_image = Image.open(io.BytesIO(output_image_data))
        
        # 确保是RGB格式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        numpy_array = np.array(pil_image)
        # RGB转BGR
        numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
        
        print(f"Output image shape: {numpy_array.shape} (BGR)")
        return numpy_array
    else:
        raise RuntimeError("No output images received from workflow")

def url_to_numpy(url):
    """
    从URL下载图像并转换为numpy数组
    Args:
        url: 图像URL
    Returns:
        numpy数组 (h, w, 3) BGR格式
    """
    response = urllib.request.urlopen(url)
    image_data = response.read()
    
    # 转换为PIL图像
    pil_image = Image.open(io.BytesIO(image_data))
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # 转换为numpy数组并转为BGR
    numpy_array = np.array(pil_image)
    numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    
    return numpy_array

# 兼容接口
def comfyui_flux_kontext_process(image_url, prompt_text, **kwargs):
    """
    Flux Kontext处理的兼容接口
    Args:
        image_url: 图像URL
        prompt_text: 提示文本
        **kwargs: 其他参数 (guidance, steps, seed)
    Returns:
        numpy数组，BGR格式
    """
    return process_image_with_flux_kontext(image_url, prompt_text, **kwargs)

# 示例用法
if __name__ == "__main__":
    # 示例：使用网络图像URL和中文提示
    test_image_url = "http://asdkfj.oss-cn-hangzhou.aliyuncs.com/97_1749784905304_tmp_d7f68de98b267afbb247cc461f50bb34cce1310209658a99.jpg"  # 替换为实际的图像URL
    prompt_text = "把图中人物的头发变成绿色的"  # 中文提示会通过翻译节点转为英文
    
    # 替换为你的实际 API Key
    api_key = "comfyui-c5702146d5af4fe5dd68fdb77cd7494a7e9b5269f80456cc6648e98444c285bb"  # 你的实际 API Key
    # api_key = None  # 如果没有 API Key，设置为 None
    
    try:
        # 处理图像
        result_image = process_image_with_flux_kontext(
            image_url=test_image_url,
            prompt_text=prompt_text,
            guidance=3,
            steps=50,
            comfy_api_key=api_key  # 添加 API Key 参数
        )
        
        print(f"Processing completed successfully!")
        print(f"Result image shape: {result_image.shape}")
        
        # 保存结果
        output_path = "flux_kontext_output_result.png"
        cv2.imwrite(output_path, result_image)
        print(f"Result saved as: {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        
    # 测试兼容接口
    print("\n--- Testing compatibility interface ---")
    try:
        result = comfyui_flux_kontext_process(
            test_image_url, 
            prompt_text,
            guidance=3,
            steps=50,
            comfy_api_key=api_key  # 添加 API Key 参数
        )
        print(f"Compatibility test successful. Shape: {result.shape}")
    except Exception as e:
        print(f"Compatibility test failed: {e}") 
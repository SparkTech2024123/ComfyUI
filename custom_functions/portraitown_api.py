import os
import sys
import json
import time
import socket
import uuid
import argparse
import subprocess
import threading
import numpy as np
import websocket
from PIL import Image
import io
import cv2
import urllib.request
import urllib.parse
import requests

# 寻找ComfyUI目录的函数
def find_path(name: str, path: str = None) -> str:
    """
    递归查找父目录以定位指定的目录或文件
    如果找到返回路径，否则返回None
    """
    # 如果没有提供路径，使用当前工作目录
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))

    # 检查当前目录是否包含目标名称
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} 已找到: {path_name}")
        return path_name

    # 获取父目录
    parent_directory = os.path.dirname(path)

    # 如果父目录与当前目录相同，说明已经到达根目录，停止搜索
    if parent_directory == path:
        return None

    # 递归在父目录中继续搜索
    return find_path(name, parent_directory)

# 添加ComfyUI目录到系统路径
def add_comfyui_directory_to_sys_path() -> None:
    """
    将ComfyUI目录添加到sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        # 确保ComfyUI路径在sys.path的最前面
        if comfyui_path in sys.path:
            sys.path.remove(comfyui_path)
        sys.path.insert(0, comfyui_path)
        print(f"'{comfyui_path}' 已添加到sys.path")

# 检查端口是否被占用
def is_port_in_use(port: int) -> bool:
    """
    检查指定端口是否已被占用
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# 等待服务器启动
def wait_for_server(port: int, timeout: int = 60) -> bool:
    """
    等待ComfyUI服务器启动，超时返回False
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/system_stats")
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False

# 启动ComfyUI服务器
def start_comfyui_server(port: int = 8190, gpu_devices: str = "0,1") -> subprocess.Popen:
    """
    启动ComfyUI服务器实例
    """
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_devices
    
    # 查找ComfyUI根目录
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is None:
        raise FileNotFoundError("无法找到ComfyUI目录")
    
    # 构建启动命令
    main_py = os.path.join(comfyui_path, "main.py")
    command = [
        sys.executable, 
        main_py, 
        "--port", str(port),
        "--listen", "0.0.0.0"  # 监听所有网络接口
    ]
    
    # 启动进程
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 启动线程来处理输出
    def log_output(stream, log_file):
        with open(log_file, 'w') as f:
            for line in stream:
                f.write(line)
                f.flush()
                print(line, end='')
    
    stdout_log = os.path.join(comfyui_path, f"comfyui_{port}.log")
    stderr_log = os.path.join(comfyui_path, f"comfyui_api_{port}.log")
    
    threading.Thread(target=log_output, args=(process.stdout, stdout_log), daemon=True).start()
    threading.Thread(target=log_output, args=(process.stderr, stderr_log), daemon=True).start()
    
    # 等待服务器启动
    if not wait_for_server(port):
        process.terminate()
        raise TimeoutError(f"等待ComfyUI服务器启动超时（端口：{port}）")
    
    print(f"ComfyUI服务器已在端口 {port} 启动")
    return process

class PortraitownAPI:
    def __init__(self, server_address="localhost:8190"):
        """
        初始化API客户端
        """
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.workflow_file = None
        self.workflow = None
        
    def load_workflow(self, workflow_path):
        """
        加载工作流JSON文件
        """
        self.workflow_file = workflow_path
        with open(workflow_path, 'r') as f:
            self.workflow = json.load(f)
        return self.workflow
    
    def queue_prompt(self, prompt):
        """
        将prompt提交到ComfyUI队列
        """
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())
    
    def get_image(self, filename, subfolder, folder_type):
        """
        从ComfyUI服务器获取图像
        """
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()
    
    def get_history(self, prompt_id):
        """
        获取指定prompt_id的历史记录
        """
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())
    
    def get_images_from_execution(self, prompt):
        """
        执行工作流并获取图像结果
        """
        # 创建WebSocket连接
        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
        
        # 提交工作流
        prompt_id = self.queue_prompt(prompt)['prompt_id']
        output_images = {}
        
        try:
            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            break  # 执行完成
                else:
                    # 二进制数据（预览）
                    continue
            
            # 获取执行历史记录
            history = self.get_history(prompt_id)[prompt_id]
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                if 'images' in node_output:
                    images_output = []
                    for image in node_output['images']:
                        image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                        images_output.append(image_data)
                    output_images[node_id] = images_output
        finally:
            ws.close()
        
        return output_images, history
    
    def modify_workflow_for_portraitown(self, image_path, template_id, abs_path=True, is_ndarray=False):
        """
        修改工作流参数，适配肖像处理需求
        
        Args:
            image_path: 输入图像的路径
            template_id: 要使用的模板ID
            abs_path: 图像路径是否为绝对路径
        
        Returns:
            修改后的工作流
        """
        if self.workflow is None:
            raise ValueError("请先加载工作流")
        
        # 深拷贝工作流避免修改原始数据
        workflow = json.loads(json.dumps(self.workflow))
        
        # 修改24号节点LoadImage的参数
        if "24" in workflow:
            workflow["24"]["inputs"]["image"] = image_path
            # 如果前端不支持这些参数，则在提交时会被忽略，但不会报错
            # workflow["24"]["inputs"]["abs_path"] = abs_path
            # workflow["24"]["inputs"]["is_ndarray"] = is_ndarray
        
        # 修改68号节点PreprocNewGetConds的template_id参数
        if "68" in workflow:
            workflow["68"]["inputs"]["template_id"] = template_id
        
        return workflow
    
    def process_result_image(self, image_data):
        """
        处理结果图像数据
        
        Args:
            image_data: 二进制图像数据
            
        Returns:
            处理后的numpy数组图像
        """
        # 转换为PIL图像
        image = Image.open(io.BytesIO(image_data))
        
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 如果图像有透明通道（RGBA），保留它
        if img_array.shape[2] == 4:
            return img_array
        else:
            # 确保是BGR格式（OpenCV默认格式）
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

def init(port=8190, gpu_devices="0,1"):
    """
    初始化函数：检查ComfyUI实例，必要时启动新实例
    
    Args:
        port: ComfyUI服务器端口
        gpu_devices: 使用的GPU设备
        
    Returns:
        PortraitownAPI对象和服务器进程（如果启动了新实例）
    """
    print(f"初始化PortraitownAPI，端口: {port}，GPU设备: {gpu_devices}")
    
    # 检查端口是否已经被占用
    server_process = None
    if is_port_in_use(port):
        print(f"端口 {port} 已经被使用，连接到现有的ComfyUI实例")
    else:
        print(f"端口 {port} 未被使用，启动新的ComfyUI实例")
        server_process = start_comfyui_server(port=port, gpu_devices=gpu_devices)
    
    # 创建API客户端
    api = PortraitownAPI(server_address=f"localhost:{port}")
    
    # 查找并加载工作流
    comfyui_path = find_path("ComfyUI")
    default_workflow_path = os.path.join(comfyui_path, "user", "default", "workflows", "flux-portraitown-api.json")
    
    # 也尝试当前目录和上层目录
    alternate_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "flux-portraitown-api.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "user", "default", "workflows", "flux-portraitown-api.json")
    ]
    
    workflow_path = None
    if os.path.exists(default_workflow_path):
        workflow_path = default_workflow_path
    else:
        for path in alternate_paths:
            if os.path.exists(path):
                workflow_path = path
                break
    
    if workflow_path is None:
        raise FileNotFoundError("无法找到'flux-portraitown-api.json'工作流文件")
    
    print(f"加载工作流: {workflow_path}")
    api.load_workflow(workflow_path)
    
    return api, server_process

def process(api, image_path, template_id="formal/female-formal-2", abs_path=True, is_ndarray=False, output_path=None):
    """
    处理函数：提交修改后的工作流到ComfyUI，获取并处理结果
    
    Args:
        api: PortraitownAPI对象
        image_path: 输入图像路径
        template_id: 模板ID
        abs_path: 图像路径是否为绝对路径
        output_path: 输出图像保存路径
        
    Returns:
        处理后的图像数组
    """
    print(f"处理图像: {image_path}")
    print(f"使用模板: {template_id}")
    
    # 修改工作流
    modified_workflow = api.modify_workflow_for_portraitown(
        image_path=image_path,
        template_id=template_id,
        abs_path=abs_path,
        is_ndarray=is_ndarray
    )
    
    # 执行工作流并获取结果
    start_time = time.time()
    output_images, history = api.get_images_from_execution(modified_workflow)
    processing_time = time.time() - start_time
    print(f"处理完成，耗时: {processing_time:.2f}秒")
    
    # 查找最终结果图像（从64节点的上一个节点获取）
    # 通常会是171节点或71节点（根据flux-portraitown-api.json中的连接判断）
    result_image = None
    
    # 根据工作流连接确定最终结果节点
    if "64" in modified_workflow:
        input_connection = modified_workflow["64"]["inputs"]["images"]
        source_node_id = str(input_connection[0])
        
        if source_node_id in output_images and output_images[source_node_id]:
            result_image = api.process_result_image(output_images[source_node_id][0])
        else:
            # 尝试查找其他可能的结果节点
            for node_id in ["171", "71", "167"]:
                if node_id in output_images and output_images[node_id]:
                    result_image = api.process_result_image(output_images[node_id][0])
                    break
    
    # 如果没有找到结果，尝试获取任何输出图像
    if result_image is None and output_images:
        for node_id in output_images:
            if output_images[node_id]:
                result_image = api.process_result_image(output_images[node_id][0])
                break
    
    # 保存结果（如果提供了输出路径）
    if result_image is not None and output_path is not None:
        if result_image.shape[2] == 4:  # 带Alpha通道
            cv2.imwrite(output_path, result_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:  # 无Alpha通道
            cv2.imwrite(output_path, result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"结果已保存到: {output_path}")
    
    return result_image

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Portraitown API处理')
    parser.add_argument('--image', type=str, default='/data/projs/WebServer/distributed-server-node/submodules/VisualForge/ComfyUI/input/example.jpg', help='输入图像路径')
    parser.add_argument('--output', type=str, default='./output/portraitown_output.png', help='输出图像路径')
    parser.add_argument('--template', type=str, default="formal/female-formal-2", help='模板ID')
    parser.add_argument('--port', type=int, default=8190, help='ComfyUI服务器端口')
    parser.add_argument('--gpu', type=str, default="0,1", help='使用的GPU设备')
    args = parser.parse_args()
    
    # 打印系统信息
    print("=== Portraitown API 测试 ===")
    print(f"Python版本: {sys.version}")
    if os.path.exists(args.image):
        print(f"输入图像存在: {args.image}")
    else:
        print(f"警告: 输入图像不存在: {args.image}")
    
    try:
        # 初始化
        api, server_process = init(port=args.port, gpu_devices=args.gpu)
        
        # 处理图像
        result = process(
            api=api,
            image_path=args.image,
            template_id=args.template,
            is_ndarray=True,
            output_path=args.output
        )
        
        if result is not None:
            print(f"处理成功，图像大小: {result.shape}")
        else:
            print("处理失败，未能获取结果图像")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 如果我们启动了新的ComfyUI实例，在这里可以选择是否终止它
        # 在实际应用中，你可能希望保持服务器运行
        # if server_process is not None:
        #     server_process.terminate()
        #     print("ComfyUI服务器已终止")
        pass
    
    print("处理完成") 
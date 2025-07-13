import torch
import numpy as np
import comfy.utils
import time
import json
import base64
import io
from PIL import Image, ImageOps, ImageSequence
import hashlib

class SaveLatentWebsocket:
    """
    自定义节点：通过WebSocket输出保存latent数据到内存
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ()
    FUNCTION = "save_latent"
    OUTPUT_NODE = True
    CATEGORY = "api/latent"

    def save_latent(self, samples):
        # 将latent tensor转换为可序列化的数据
        latent_tensor = samples["samples"]
        
        # 转换为numpy数组并序列化
        latent_numpy = latent_tensor.cpu().numpy()
        latent_bytes = latent_numpy.tobytes()
        
        # 使用base64编码以便JSON传输
        latent_b64 = base64.b64encode(latent_bytes).decode('utf-8')
        
        # 创建包含所有信息的结果
        result = {
            "tensor_b64": latent_b64,
            "shape": list(latent_numpy.shape),
            "dtype": str(latent_numpy.dtype)
        }
        
        # 将数据作为JSON字符串返回在ui中
        results = [{
            "latent_data": json.dumps(result)
        }]
        
        return {"ui": {"latent_data": results}}

    @classmethod
    def IS_CHANGED(s, samples):
        return time.time()

class LoadLatentWebsocket:
    """
    自定义节点：从base64编码的latent数据中加载，支持JSON格式和原始格式
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_data": ("STRING", {"default": "", "multiline": True, "tooltip": "Base64编码的latent数据。可以是完整JSON格式（包含tensor_b64、shape、dtype），或仅原始base64数据"}),
            },
            "optional": {
                "tensor_shape": ("STRING", {"default": "[1,4,64,64]", "tooltip": "当latent_data为原始base64时需要：Tensor形状，格式如[1,4,64,64]"}),
                "tensor_dtype": ("STRING", {"default": "float32", "tooltip": "当latent_data为原始base64时需要：数据类型，如float32"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load_latent"
    CATEGORY = "api/latent"

    def load_latent(self, latent_data, tensor_shape="[1,4,64,64]", tensor_dtype="float32"):
        if not latent_data or latent_data.strip() == "":
            raise ValueError("latent_data is required")
            
        latent_data = latent_data.strip()
        
        # 尝试解析为JSON格式
        try:
            data = json.loads(latent_data)
            if isinstance(data, dict) and "tensor_b64" in data:
                return self._load_from_json_data(data)
        except json.JSONDecodeError:
            pass
        
        # 如果不是JSON格式，当作原始base64数据处理
        return self._load_from_base64(latent_data, tensor_shape, tensor_dtype)

    def _load_from_json_data(self, data):
        """
        从JSON格式的latent数据中加载
        """
        try:
            tensor_b64 = data["tensor_b64"]
            shape = data["shape"]
            dtype = data["dtype"]
            
            # 解码base64数据
            latent_bytes = base64.b64decode(tensor_b64)
            
            # 重建numpy数组
            numpy_dtype = getattr(np, dtype.replace("torch.", ""))
            latent_numpy = np.frombuffer(latent_bytes, dtype=numpy_dtype).reshape(shape)
            
            # 转换为torch tensor
            latent_tensor = torch.from_numpy(latent_numpy.copy())
            
            # 注意：不应用缩放因子，保持与SaveLatentWebsocket的数据一致
            # SaveLatentWebsocket保存的是原始数据，所以这里也直接使用原始数据
            latent_tensor = latent_tensor.float()
            
            return ({"samples": latent_tensor},)
            
        except KeyError as e:
            raise ValueError("Missing required field in JSON: {}".format(str(e)))
        except Exception as e:
            raise ValueError("Error loading latent from JSON: {}".format(str(e)))

    def _load_from_base64(self, base64_data, tensor_shape, tensor_dtype):
        """
        从原始base64数据中加载
        """
        try:
            # 解析形状
            shape = json.loads(tensor_shape)
            if not isinstance(shape, list):
                raise ValueError("tensor_shape must be a list")
            
            # 解码base64数据
            latent_bytes = base64.b64decode(base64_data)
            
            # 重建numpy数组
            numpy_dtype = getattr(np, tensor_dtype.replace("torch.", ""))
            latent_numpy = np.frombuffer(latent_bytes, dtype=numpy_dtype).reshape(shape)
            
            # 转换为torch tensor
            latent_tensor = torch.from_numpy(latent_numpy.copy())
            
            # 注意：不应用缩放因子，保持与SaveLatentWebsocket的数据一致
            # 如果数据来源于SaveLatentWebsocket，应该保持原始值
            latent_tensor = latent_tensor.float()
            
            return ({"samples": latent_tensor},)
            
        except json.JSONDecodeError as e:
            raise ValueError("Invalid tensor_shape format: {}".format(str(e)))
        except Exception as e:
            raise ValueError("Error loading latent from base64: {}".format(str(e)))

    @classmethod
    def IS_CHANGED(s, latent_data, tensor_shape="[1,4,64,64]", tensor_dtype="float32"):
        # 基于输入数据生成hash，确保数据变化时重新处理
        if latent_data:
            m = hashlib.sha256()
            m.update(latent_data.encode('utf-8'))
            m.update(tensor_shape.encode('utf-8'))
            m.update(tensor_dtype.encode('utf-8'))
            return m.digest().hex()
        return time.time()

    @classmethod
    def VALIDATE_INPUTS(s, latent_data, tensor_shape="[1,4,64,64]", tensor_dtype="float32"):
        if not latent_data or latent_data.strip() == "":
            return "latent_data is required"
            
        latent_data = latent_data.strip()
        
        # 检查是否为JSON格式
        try:
            data = json.loads(latent_data)
            if isinstance(data, dict) and "tensor_b64" in data:
                required_fields = ["tensor_b64", "shape", "dtype"]
                for field in required_fields:
                    if field not in data:
                        return "Missing required field '{}' in JSON latent_data".format(field)
                return True
        except json.JSONDecodeError:
            pass
        
        # 如果不是JSON格式，验证base64和其他参数
        try:
            base64.b64decode(latent_data)
        except Exception:
            return "Invalid base64 data in latent_data"
            
        try:
            shape = json.loads(tensor_shape)
            if not isinstance(shape, list):
                return "tensor_shape must be a valid JSON list"
        except json.JSONDecodeError:
            return "Invalid tensor_shape format, must be valid JSON list"
        
        return True

class LoadImageWebsocket:
    """
    图像加载节点：只接受base64编码的图像数据
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_image": ("STRING", {"default": "", "multiline": True, "tooltip": "Base64编码的图像数据"}),
            }
        }

    CATEGORY = "api/image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    
    def load_image(self, base64_image):
        if not base64_image or base64_image.strip() == "":
            raise ValueError("base64_image is required")
        
        return self._process_base64_image(base64_image.strip())
    
    def _process_base64_image(self, base64_image):
        """
        处理base64编码的图像数据
        """
        try:
            # 解码base64数据
            image_bytes = base64.b64decode(base64_image)
            
            # 转换为PIL图像
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image = ImageOps.exif_transpose(pil_image)
            
            # 处理透明度
            has_alpha = False
            alpha_channel = None
            
            if pil_image.mode in ('RGBA', 'LA'):
                has_alpha = True
                if pil_image.mode == 'RGBA':
                    alpha_channel = pil_image.getchannel('A')
                else:  # LA
                    alpha_channel = pil_image.getchannel('A')
            
            # 转换为RGB
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 转换为numpy数组然后转为torch tensor
            image_array = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # 添加batch维度
            
            # 创建mask
            if has_alpha and alpha_channel is not None:
                mask_array = np.array(alpha_channel).astype(np.float32) / 255.0
                mask_tensor = 1.0 - torch.from_numpy(mask_array).unsqueeze(0)  # 反转并添加batch维度
            else:
                height, width = image_array.shape[:2]
                mask_tensor = torch.zeros((1, height, width), dtype=torch.float32)
            
            return (image_tensor, mask_tensor)
            
        except Exception as e:
            raise ValueError("Error processing base64 image data: {}".format(str(e)))

    @classmethod
    def IS_CHANGED(s, base64_image):
        if base64_image and base64_image.strip():
            # 为base64图像数据生成hash
            m = hashlib.sha256()
            m.update(base64_image.strip().encode('utf-8'))
            return m.digest().hex()
        return time.time()

    @classmethod
    def VALIDATE_INPUTS(s, base64_image):
        if not base64_image or base64_image.strip() == "":
            return "base64_image is required"
        
        try:
            # 验证base64格式
            base64.b64decode(base64_image.strip())
        except Exception as e:
            return "Invalid base64 image data: {}".format(str(e))
        
        return True

# 节点注册
NODE_CLASS_MAPPINGS = {
    "SaveLatentWebsocket": SaveLatentWebsocket,
    "LoadLatentWebsocket": LoadLatentWebsocket,
    "LoadImageWebsocket": LoadImageWebsocket,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveLatentWebsocket": "Save Latent (WebSocket)",
    "LoadLatentWebsocket": "Load Latent (WebSocket)",
    "LoadImageWebsocket": "Load Image (WebSocket)",
} 
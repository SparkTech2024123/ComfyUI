"""
高效数据传输节点集合
支持共享内存、pickle序列化、内存映射等方式进行本地高速数据传输
适用于客户端和服务器在同一台机器上的场景
"""

import torch
import numpy as np
import comfy.utils
import time
import json
import pickle
import tempfile
import os
import uuid
import hashlib
from multiprocessing import shared_memory
from PIL import Image, ImageOps
import io
import cv2

class LoadImageSharedMemory:
    """
    通过共享内存加载图像数据 - 最高性能方案
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shm_name": ("STRING", {"default": "", "tooltip": "共享内存块名称"}),
                "shape": ("STRING", {"default": "[1080,1920,3]", "tooltip": "图像形状 [height,width,channels]"}),
                "dtype": ("STRING", {"default": "uint8", "tooltip": "数据类型"}),
            },
            "optional": {
                "convert_bgr_to_rgb": ("BOOLEAN", {"default": True, "tooltip": "是否将BGR格式转换为RGB格式 (OpenCV默认使用BGR)"}),
            }
        }

    CATEGORY = "api/image/efficient"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    
    def load_image(self, shm_name, shape, dtype, convert_bgr_to_rgb=True):
        if not shm_name or shm_name.strip() == "":
            raise ValueError("shm_name is required")
        
        try:
            # 解析参数
            shape_list = json.loads(shape)
            if len(shape_list) != 3:
                raise ValueError("Shape must be [height, width, channels]")
            
            height, width, channels = shape_list
            numpy_dtype = getattr(np, dtype)
            
            # 连接到共享内存
            shm = shared_memory.SharedMemory(name=shm_name.strip())
            
            # 从共享内存重建numpy数组
            image_array = np.ndarray(shape_list, dtype=numpy_dtype, buffer=shm.buf)
            
            # 复制数据（避免共享内存被释放）
            image_copy = image_array.copy()
            
            # 关闭共享内存连接（不删除）
            shm.close()
            
            # BGR到RGB转换 (OpenCV默认使用BGR格式，ComfyUI期望RGB格式)
            if convert_bgr_to_rgb and channels == 3:
                image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
                print(f"✓ Converted BGR to RGB for ComfyUI processing")
            
            # 标准化到0-1范围
            if image_copy.dtype == np.uint8:
                image_normalized = image_copy.astype(np.float32) / 255.0
            else:
                image_normalized = image_copy.astype(np.float32)
            
            # 转换为torch tensor并添加batch维度
            image_tensor = torch.from_numpy(image_normalized).unsqueeze(0)
            
            # 创建空mask
            mask_tensor = torch.zeros((1, height, width), dtype=torch.float32)
            
            return (image_tensor, mask_tensor)
            
        except Exception as e:
            raise ValueError(f"Error loading image from shared memory: {e}")

    @classmethod
    def IS_CHANGED(s, shm_name, shape, dtype, convert_bgr_to_rgb=True):
        if shm_name and shm_name.strip():
            m = hashlib.sha256()
            m.update(shm_name.strip().encode('utf-8'))
            m.update(shape.encode('utf-8'))
            m.update(dtype.encode('utf-8'))
            m.update(str(convert_bgr_to_rgb).encode('utf-8'))
            return m.digest().hex()
        return time.time()

class SaveImageSharedMemory:
    """
    将图像数据保存到共享内存 - 高性能数据传输方案
    相比 SaveImageWebsocket 避免了 PNG 编码和网络传输开销
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "shm_name": ("STRING", {"default": "", "tooltip": "可选的共享内存块名称，留空则自动生成"}),
                "output_format": (["RGB", "RGBA", "BGR"], {"default": "RGB", "tooltip": "输出图像格式"}),
                "convert_rgb_to_bgr": ("BOOLEAN", {"default": False, "tooltip": "是否将RGB转换为BGR格式 (OpenCV兼容)"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "api/image/efficient"

    def save_images(self, images, shm_name="", output_format="RGB", convert_rgb_to_bgr=False):
        try:
            # 只处理第一张图像（批处理的第一个）
            if images.shape[0] > 1:
                print(f"Warning: Multiple images provided ({images.shape[0]}), only processing the first one")
            
            image_tensor = images[0]  # 移除batch维度: (H, W, C)
            
            # 转换为numpy数组 (0-255 uint8范围)
            image_numpy = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            # 格式转换
            if convert_rgb_to_bgr and image_numpy.shape[2] == 3:
                image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
                print(f"✓ Converted RGB to BGR for output")
            
            # 生成共享内存名称（如果未提供）
            if not shm_name or shm_name.strip() == "":
                shm_name = f"comfyui_output_{uuid.uuid4().hex[:16]}"
            else:
                shm_name = shm_name.strip()
            
            # 创建共享内存块
            shm_size = image_numpy.nbytes
            shm = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)
            
            # 将数据写入共享内存
            shm_array = np.ndarray(image_numpy.shape, dtype=image_numpy.dtype, buffer=shm.buf)
            shm_array[:] = image_numpy[:]
            
            # 关闭共享内存连接（保持数据存在）
            shm.close()
            
            # 准备返回信息
            result_info = {
                "shm_name": shm_name,
                "shape": list(image_numpy.shape),
                "dtype": str(image_numpy.dtype),
                "format": output_format,
                "size_bytes": shm_size,
                "size_mb": round(shm_size / 1024 / 1024, 2)
            }
            
            print(f"✓ Image saved to shared memory: {shm_name}")
            print(f"  - Shape: {image_numpy.shape}")
            print(f"  - Size: {result_info['size_mb']} MB")
            print(f"  - Format: {output_format}")
            
            # 返回共享内存信息
            results = [result_info]
            
            return {"ui": {"shared_memory_info": results}}
            
        except Exception as e:
            raise ValueError(f"Error saving image to shared memory: {e}")

    @classmethod
    def IS_CHANGED(s, images, shm_name="", output_format="RGB", convert_rgb_to_bgr=False):
        return time.time()

class SaveLatentSharedMemory:
    """
    将latent数据保存到共享内存 - 高性能数据传输方案
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
            },
            "optional": {
                "shm_name": ("STRING", {"default": "", "tooltip": "可选的共享内存块名称，留空则自动生成"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_latent"
    OUTPUT_NODE = True
    CATEGORY = "api/latent/efficient"

    def save_latent(self, samples, shm_name=""):
        try:
            # 获取latent tensor
            latent_tensor = samples["samples"]
            
            # 转换为numpy数组
            latent_numpy = latent_tensor.cpu().numpy()
            
            # 生成共享内存名称（如果未提供）
            if not shm_name or shm_name.strip() == "":
                shm_name = f"latent_{uuid.uuid4().hex[:16]}"
            else:
                shm_name = shm_name.strip()
            
            # 创建共享内存块
            shm_size = latent_numpy.nbytes
            shm = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)
            
            # 将数据写入共享内存
            shm_array = np.ndarray(latent_numpy.shape, dtype=latent_numpy.dtype, buffer=shm.buf)
            shm_array[:] = latent_numpy[:]
            
            # 关闭共享内存连接（保持数据存在）
            shm.close()
            
            # 准备返回信息
            result_info = {
                "shm_name": shm_name,
                "shape": list(latent_numpy.shape),
                "dtype": str(latent_numpy.dtype),
                "size_bytes": shm_size
            }
            
            # 返回共享内存信息
            results = [result_info]
            
            return {"ui": {"latent_shm_info": results}}
            
        except Exception as e:
            raise ValueError(f"Error saving latent to shared memory: {e}")

    @classmethod
    def IS_CHANGED(s, samples, shm_name=""):
        return time.time()

class LoadLatentSharedMemory:
    """
    从共享内存加载latent数据 - 高性能数据传输方案
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shm_name": ("STRING", {"default": "", "tooltip": "共享内存块名称"}),
                "shape": ("STRING", {"default": "[1,4,64,64]", "tooltip": "Latent形状 [batch,channels,height,width]"}),
                "dtype": ("STRING", {"default": "float32", "tooltip": "数据类型"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load_latent"
    CATEGORY = "api/latent/efficient"

    def load_latent(self, shm_name, shape, dtype):
        if not shm_name or shm_name.strip() == "":
            raise ValueError("shm_name is required")
        
        try:
            # 解析参数
            shape_list = json.loads(shape)
            if not isinstance(shape_list, list) or len(shape_list) != 4:
                raise ValueError("Shape must be [batch, channels, height, width]")
            
            numpy_dtype = getattr(np, dtype)
            
            # 连接到共享内存
            shm = shared_memory.SharedMemory(name=shm_name.strip())
            
            # 从共享内存重建numpy数组
            latent_array = np.ndarray(shape_list, dtype=numpy_dtype, buffer=shm.buf)
            
            # 复制数据（避免共享内存被释放）
            latent_copy = latent_array.copy()
            
            # 关闭共享内存连接（不删除）
            shm.close()
            
            # 转换为torch tensor
            latent_tensor = torch.from_numpy(latent_copy).float()
            
            return ({"samples": latent_tensor},)
            
        except Exception as e:
            raise ValueError(f"Error loading latent from shared memory: {e}")

    @classmethod
    def IS_CHANGED(s, shm_name, shape, dtype):
        if shm_name and shm_name.strip():
            m = hashlib.sha256()
            m.update(shm_name.strip().encode('utf-8'))
            m.update(shape.encode('utf-8'))
            m.update(dtype.encode('utf-8'))
            return m.digest().hex()
        return time.time()

    @classmethod
    def VALIDATE_INPUTS(s, shm_name, shape, dtype):
        if not shm_name or shm_name.strip() == "":
            return "shm_name is required"
            
        try:
            shape_list = json.loads(shape)
            if not isinstance(shape_list, list) or len(shape_list) != 4:
                return "Shape must be [batch, channels, height, width]"
        except json.JSONDecodeError:
            return "Invalid shape format, must be valid JSON list"
        
        try:
            getattr(np, dtype)
        except AttributeError:
            return f"Invalid dtype: {dtype}"
        
        return True

# 节点注册
NODE_CLASS_MAPPINGS = {
    "LoadImageSharedMemory": LoadImageSharedMemory,
    "SaveImageSharedMemory": SaveImageSharedMemory,
    "SaveLatentSharedMemory": SaveLatentSharedMemory,
    "LoadLatentSharedMemory": LoadLatentSharedMemory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageSharedMemory": "Load Image (Shared Memory)",
    "SaveImageSharedMemory": "Save Image (Shared Memory)",
    "SaveLatentSharedMemory": "Save Latent (Shared Memory)",
    "LoadLatentSharedMemory": "Load Latent (Shared Memory)",
} 
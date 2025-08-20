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
    支持单图和多图批量加载
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shm_name": ("STRING", {"default": "", "tooltip": "共享内存块名称，多个名称用逗号分隔"}),
                "shape": ("STRING", {"default": "[1080,1920,3]", "tooltip": "图像形状 [height,width,channels]"}),
                "dtype": ("STRING", {"default": "uint8", "tooltip": "数据类型"}),
            },
            "optional": {
                "convert_bgr_to_rgb": ("BOOLEAN", {"default": True, "tooltip": "是否将BGR格式转换为RGB格式 (OpenCV默认使用BGR)"}),
                "batch_mode": ("BOOLEAN", {"default": False, "tooltip": "是否启用批处理模式（多图加载）"}),
            }
        }

    CATEGORY = "api/image/efficient"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    
    def load_image(self, shm_name, shape, dtype, convert_bgr_to_rgb=True, batch_mode=False):
        if not shm_name or shm_name.strip() == "":
            raise ValueError("shm_name is required")

        try:
            # 解析参数
            shape_list = json.loads(shape)
            if len(shape_list) != 3:
                raise ValueError("Shape must be [height, width, channels]")

            height, width, channels = shape_list
            numpy_dtype = getattr(np, dtype)

            # 处理共享内存名称（支持多个名称）
            shm_names = [name.strip() for name in shm_name.split(',') if name.strip()]

            if not shm_names:
                raise ValueError("No valid shared memory names provided")

            print(f"=== LoadImageSharedMemory Debug Info ===")
            print(f"Batch mode: {batch_mode}")
            print(f"Loading {len(shm_names)} image(s) from shared memory")
            print(f"Expected shape per image: {shape_list}")

            loaded_images = []

            # 加载每个共享内存块
            for i, current_shm_name in enumerate(shm_names):
                print(f"Loading image {i+1}/{len(shm_names)}: {current_shm_name}")

                # 连接到共享内存
                shm = shared_memory.SharedMemory(name=current_shm_name)

                # 从共享内存重建numpy数组
                image_array = np.ndarray(shape_list, dtype=numpy_dtype, buffer=shm.buf)

                # 复制数据（避免共享内存被释放）
                image_copy = image_array.copy()

                # 关闭共享内存连接（不删除）
                shm.close()

                # BGR到RGB转换 (OpenCV默认使用BGR格式，ComfyUI期望RGB格式)
                if convert_bgr_to_rgb and channels == 3:
                    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
                    print(f"  ✓ Converted BGR to RGB for image {i+1}")

                # 标准化到0-1范围
                if image_copy.dtype == np.uint8:
                    image_normalized = image_copy.astype(np.float32) / 255.0
                else:
                    image_normalized = image_copy.astype(np.float32)

                loaded_images.append(image_normalized)
                print(f"  ✓ Successfully loaded image {i+1}: {image_copy.shape}")

            # 根据模式返回结果
            if batch_mode and len(loaded_images) > 1:
                # 批处理模式：将所有图像堆叠为一个批次
                batch_tensor = torch.from_numpy(np.stack(loaded_images, axis=0))
                batch_size = len(loaded_images)

                # 创建批次mask
                mask_tensor = torch.zeros((batch_size, height, width), dtype=torch.float32)

                print(f"✓ Batch mode: Created tensor with shape {batch_tensor.shape}")
                return (batch_tensor, mask_tensor)
            else:
                # 单图模式：只返回第一张图像（向后兼容）
                image_tensor = torch.from_numpy(loaded_images[0]).unsqueeze(0)
                mask_tensor = torch.zeros((1, height, width), dtype=torch.float32)

                if len(loaded_images) > 1:
                    print(f"⚠️  Multiple images provided but batch_mode=False, only returning first image")

                print(f"✓ Single mode: Created tensor with shape {image_tensor.shape}")
                return (image_tensor, mask_tensor)

        except Exception as e:
            raise ValueError(f"Error loading image from shared memory: {e}")

    @classmethod
    def IS_CHANGED(s, shm_name, shape, dtype, convert_bgr_to_rgb=True, batch_mode=False):
        if shm_name and shm_name.strip():
            m = hashlib.sha256()
            m.update(shm_name.strip().encode('utf-8'))
            m.update(shape.encode('utf-8'))
            m.update(dtype.encode('utf-8'))
            m.update(str(convert_bgr_to_rgb).encode('utf-8'))
            m.update(str(batch_mode).encode('utf-8'))
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

    def _safe_create_shared_memory(self, size, base_name, batch_number, max_retries=5):
        """
        Safely create shared memory with conflict resolution

        Args:
            size: Size of shared memory in bytes
            base_name: Base name for shared memory
            batch_number: Batch number for this image
            max_retries: Maximum number of retry attempts

        Returns:
            tuple: (SharedMemory object, actual name used)
        """
        for attempt in range(max_retries):
            try:
                # Generate unique name with timestamp and process ID for better uniqueness
                timestamp = int(time.time() * 1000000)  # microsecond precision
                process_id = os.getpid()

                if base_name and base_name.strip():
                    # Use provided base name with additional uniqueness
                    current_shm_name = f"{base_name.strip()}_{batch_number:03d}_{timestamp}_{process_id}"
                else:
                    # Generate completely unique name
                    current_shm_name = f"comfyui_output_{uuid.uuid4().hex[:16]}_{batch_number:03d}_{timestamp}_{process_id}"

                # Try to create shared memory
                shm = shared_memory.SharedMemory(create=True, size=size, name=current_shm_name)
                return shm, current_shm_name

            except FileExistsError:
                # Shared memory with this name already exists
                print(f"  ⚠️  Shared memory name conflict (attempt {attempt + 1}): {current_shm_name}")

                # Try to clean up the existing segment
                try:
                    existing_shm = shared_memory.SharedMemory(name=current_shm_name)
                    existing_shm.close()
                    existing_shm.unlink()
                    print(f"  🗑️  Cleaned up existing shared memory: {current_shm_name}")
                except Exception as cleanup_error:
                    print(f"  ⚠️  Failed to cleanup existing shared memory: {cleanup_error}")

                # Add small delay before retry
                time.sleep(0.001 * (attempt + 1))  # Progressive delay
                continue

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to create shared memory after {max_retries} attempts: {e}")
                print(f"  ⚠️  Shared memory creation error (attempt {attempt + 1}): {e}")
                time.sleep(0.001 * (attempt + 1))
                continue

        raise ValueError(f"Failed to create shared memory after {max_retries} attempts")

    def save_images(self, images, shm_name="", output_format="RGB", convert_rgb_to_bgr=False):
        try:
            # 处理批处理中的所有图像
            batch_size = images.shape[0]
            print(f"=== SaveImageSharedMemory Debug Info ===")
            print(f"Input images tensor shape: {images.shape}")
            print(f"Processing {batch_size} image(s) for shared memory storage")

            results = []

            # 为每张图像创建独立的共享内存块
            for batch_number, image_tensor in enumerate(images):
                # 转换为numpy数组 (0-255 uint8范围)
                image_numpy = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

                # 格式转换
                if convert_rgb_to_bgr and image_numpy.shape[2] == 3:
                    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
                    print(f"✓ Converted RGB to BGR for image {batch_number}")

                # 使用安全的共享内存创建方法
                shm_size = image_numpy.nbytes
                shm, current_shm_name = self._safe_create_shared_memory(
                    size=shm_size,
                    base_name=shm_name,
                    batch_number=batch_number
                )

                # 将数据写入共享内存
                shm_array = np.ndarray(image_numpy.shape, dtype=image_numpy.dtype, buffer=shm.buf)
                shm_array[:] = image_numpy[:]

                # 关闭共享内存连接（保持数据存在）
                shm.close()

                # 准备返回信息
                result_info = {
                    "shm_name": current_shm_name,
                    "shape": list(image_numpy.shape),
                    "dtype": str(image_numpy.dtype),
                    "format": output_format,
                    "size_bytes": shm_size,
                    "size_mb": round(shm_size / 1024 / 1024, 2),
                    "batch_number": batch_number
                }

                results.append(result_info)

                print(f"✓ Image {batch_number} saved to shared memory: {current_shm_name}")
                print(f"  - Shape: {image_numpy.shape}")
                print(f"  - Size: {result_info['size_mb']} MB")
                print(f"  - Format: {output_format}")

            print(f"✓ Successfully processed {len(results)} image(s) to shared memory")

            return {"ui": {"shared_memory_info": results}}

        except Exception as e:
            raise ValueError(f"Error saving images to shared memory: {e}")

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
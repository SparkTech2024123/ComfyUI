"""
ComfyUI Node for Volcano Engine Doubao SeedEdit 3.0 Image-to-Image model
Wraps the doubao-seededit-3-0-i2i-250628 model for use in ComfyUI workflows
"""

import os
import sys
import torch
import numpy as np
import subprocess
import importlib.util
from PIL import Image
import io
import base64
import requests

# Load .env file using python-dotenv
try:
    from dotenv import load_dotenv
    # Load .env file from ComfyUI root directory
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        # Try to load from current directory as fallback
        load_dotenv()
except ImportError:
    print("Warning: python-dotenv not available, .env file will not be loaded automatically")
except Exception as e:
    print(f"Warning: Failed to load .env file: {e}")

python = sys.executable

def is_installed(package, package_overwrite=None, auto_install=True):
    """Check if package is installed and install if needed"""
    is_has = False
    try:
        spec = importlib.util.find_spec(package)
        is_has = spec is not None
    except ModuleNotFoundError:
        pass

    package = package_overwrite or package

    if spec is None and auto_install:
        print(f"Installing {package}...")
        command = f'"{python}" -m pip install {package}'
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ)
        
        if result.returncode == 0:
            is_has = True
        else:
            print(f"Failed to install {package}: {result.stderr.decode()}")
            is_has = False
    elif spec is not None:
        print(f"{package} ## OK")
        is_has = True

    return is_has

# Import Volcano Engine ARK SDK
ark_client = None
try:
    if is_installed('volcengine-python-sdk[ark]', 'volcengine-python-sdk[ark]'):
        from volcenginesdkarkruntime import Ark
        print("Volcano Engine ARK SDK imported successfully")
    else:
        print("Failed to install volcengine-python-sdk[ark] package")
except Exception as e:
    print(f"Failed to import Volcano Engine ARK SDK: {e}")

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    if isinstance(tensor, torch.Tensor):
        # Handle batch dimension
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Convert to numpy
        image_np = tensor.cpu().numpy()
        
        # Ensure correct format (H, W, C)
        if image_np.shape[0] == 3:  # CHW to HWC
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # Scale to 0-255 if needed
        if image_np.dtype == np.float32 or image_np.dtype == np.float64:
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(image_np)
    return tensor

def pil_to_tensor(pil_image):
    """Convert PIL Image to tensor"""
    if isinstance(pil_image, Image.Image):
        image_array = np.array(pil_image.convert('RGB')).astype(np.float32) / 255.0
        return torch.from_numpy(image_array).unsqueeze(0)
    return pil_image

def pil_to_base64(pil_image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def is_url(string):
    """Check if string is a valid URL"""
    return isinstance(string, str) and (string.startswith('http://') or string.startswith('https://') or string.startswith('data:image/'))

class VolcanoDoubaoSeedEditNode:
    """ComfyUI Node for Volcano Engine Doubao SeedEdit 3.0 I2I model"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "改成爱心形状的泡泡", 
                    "multiline": True,
                    "tooltip": "Text prompt describing the desired changes to the image"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image to be edited (tensor format)"
                }),
            },
            "optional": {
                "model": ("STRING", {
                    "default": "doubao-seededit-3-0-i2i-250628",
                    "tooltip": "Model name for Volcano Engine Doubao SeedEdit"
                }),
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Image URL (if provided, takes priority over tensor image input)"
                }),
                "size": (["adaptive", "1024x1024", "1024x768", "768x1024", "1152x896", "896x1152"], {
                    "default": "adaptive",
                    "tooltip": "Output image size"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "Random seed for generation. Use -1 for random."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 5.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "How closely to follow the prompt (higher = more adherence to prompt)"
                }),
                "watermark": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to add watermark to the generated image"
                }),
                "ark_api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Volcano Engine ARK API Key (leave empty to use ARK_API_KEY environment variable)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "🌋 Volcano Engine"
    
    def generate_image(self, prompt, image, model="doubao-seededit-3-0-i2i-250628", image_url="", size="adaptive", seed=-1, guidance_scale=5.5, watermark=True, ark_api_key=""):
        """Generate edited image using Volcano Engine Doubao SeedEdit 3.0 I2I model"""
        
        # Validate inputs
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if not image_url and image is None:
            raise ValueError("Either image_url or image tensor is required")
        
        if not model or not model.strip():
            raise ValueError("Model name cannot be empty")
        
        # Set API key
        api_key = ark_api_key.strip() if ark_api_key else os.environ.get("ARK_API_KEY", "").strip()
        if not api_key:
            error_msg = "ARK API key required. Either:\n1. Set the ark_api_key parameter, or\n2. Set ARK_API_KEY environment variable"
            print(error_msg)
            raise ValueError(error_msg)
        
        try:
            print(f"Starting Volcano Engine Doubao SeedEdit generation...")
            print(f"Model: {model}")
            print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"Size: {size}, Seed: {seed}, Guidance Scale: {guidance_scale}, Watermark: {watermark}")
            
            # Initialize Ark client
            client = Ark(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=api_key,
            )
            
            # Determine image input format
            image_input = None
            if image_url and image_url.strip():
                # Use provided URL
                image_input = image_url.strip()
                print(f"Using image URL: {image_input[:100]}{'...' if len(image_input) > 100 else ''}")
            else:
                # Convert tensor to base64 data URL
                pil_image = tensor_to_pil(image)
                if pil_image is None:
                    raise ValueError("Failed to convert tensor to PIL image")
                
                img_base64 = pil_to_base64(pil_image)
                image_input = f"data:image/png;base64,{img_base64}"
                print("Using tensor image converted to base64")
            
            # Prepare API call arguments
            api_args = {
                "model": model,
                "prompt": prompt,
                "image": image_input,
                "size": size,
                "guidance_scale": guidance_scale,
                "watermark": watermark,
            }
            
            # Add seed if specified (not -1)
            if seed >= 0:
                api_args["seed"] = seed
                print(f"Using seed: {seed}")
            
            print(f"Calling Volcano Engine API with model: {model}")
            
            # Make the API call using official SDK
            response = client.images.generate(**api_args)
            
            print("API call successful, processing result...")
            
            # Validate response
            if not response or not hasattr(response, 'data') or len(response.data) == 0:
                raise RuntimeError("No images returned from API")
            
            # Extract image URL from result
            result_image_url = response.data[0].url
            print(f"Generated image URL: {result_image_url}")
            
            # Download the generated image
            try:
                download_response = requests.get(result_image_url, timeout=60)
                download_response.raise_for_status()
                
                if len(download_response.content) == 0:
                    raise RuntimeError("Downloaded image is empty")
                
                # Convert downloaded image to PIL then tensor
                generated_image = Image.open(io.BytesIO(download_response.content))
                result_tensor = pil_to_tensor(generated_image)
                
                print("Image editing completed successfully")
                return (result_tensor,)
                
            except requests.exceptions.RequestException as download_error:
                error_msg = f"Failed to download generated image: {str(download_error)}"
                print(error_msg)
                raise RuntimeError(error_msg) from download_error
            except Exception as conversion_error:
                error_msg = f"Failed to process generated image: {str(conversion_error)}"
                print(error_msg)
                raise RuntimeError(error_msg) from conversion_error
                
        except Exception as e:
            # Handle SDK-specific errors
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code
                if status_code == 401:
                    error_msg = f"Authentication failed: {str(e)}\nCheck your ARK API key is valid and has sufficient credits."
                elif status_code == 400:
                    error_msg = f"Bad request: {str(e)}\nCheck your prompt, image, and parameters are valid."
                elif status_code == 404:
                    error_msg = f"Model not found: {str(e)}\nModel '{model}' might not be available. Check your model access permissions."
                else:
                    error_msg = f"API error ({status_code}): {str(e)}"
                print(error_msg)
                raise RuntimeError(error_msg) from e
            
            # Re-raise known errors without wrapping
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            
            # Wrap unknown errors
            error_msg = f"Unexpected error in Volcano Engine Doubao SeedEdit generation: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e

# Node registration
NODE_CLASS_MAPPINGS = {
    "VolcanoDoubaoSeedEditNode": VolcanoDoubaoSeedEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VolcanoDoubaoSeedEditNode": "🌋 Volcano Doubao SeedEdit 3.0 I2I",
}
"""
ComfyUI Node for fal-ai/flux-pro/kontext API integration
Wraps the fal-ai flux pro kontext model for use in ComfyUI workflows
"""

import os
import sys
import torch
import numpy as np
import tempfile
import subprocess
import importlib.util
from PIL import Image

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

# Try to import fal_client
fal_client = None
try:
    if is_installed('fal_client', 'fal-client'):
        import fal_client
        print("fal_client imported successfully")
except Exception as e:
    print(f"Failed to import fal_client: {e}")

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

def upload_image_to_fal(image_tensor):
    """Upload image tensor to fal and return URL"""
    if fal_client is None:
        raise RuntimeError("fal_client not available. Please install fal-client package.")
    
    temp_file_path = None
    try:
        # Validate input
        if image_tensor is None:
            raise ValueError("Image tensor cannot be None")
        
        # Convert tensor to PIL
        pil_image = tensor_to_pil(image_tensor)
        if pil_image is None:
            raise ValueError("Failed to convert tensor to PIL image")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            pil_image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name
        
        if not os.path.exists(temp_file_path):
            raise RuntimeError("Failed to save temporary image file")
        
        # Upload to fal
        print(f"Uploading image file: {temp_file_path} ({os.path.getsize(temp_file_path)} bytes)")
        image_url = fal_client.upload_file(temp_file_path)
        
        if not image_url:
            raise RuntimeError("fal_client returned empty URL")
        
        print(f"Image uploaded successfully to: {image_url}")
        
        # Clean up
        os.unlink(temp_file_path)
        temp_file_path = None
        
        return image_url
    
    except Exception as e:
        error_msg = f"Failed to upload image to fal.ai: {str(e)}"
        print(error_msg)
        
        # Ensure cleanup happens even on error
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                print(f"Warning: Failed to clean up temp file {temp_file_path}: {cleanup_error}")
        
        raise RuntimeError(error_msg) from e

class FalFluxProKontextNode:
    """ComfyUI Node for fal-ai/flux-pro/kontext model"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "Put a donut next to the flour.", 
                    "multiline": True,
                    "tooltip": "Text prompt describing what to generate or modify"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image to modify with the prompt"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "🎨 Fal.ai/Flux"
    
    def generate_image(self, prompt, image):
        """Generate image using fal-ai/flux-pro/kontext model"""
        
        if fal_client is None:
            error_msg = "fal_client not available. Please install fal-client package:\npip install fal-client"
            print(error_msg)
            raise RuntimeError(error_msg)
        
        # Validate inputs
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if image is None:
            raise ValueError("Input image is required")
        
        # Handle optional parameters with defaults
        enable_safety_checker = True
        seed = -1
            
        # Set API key
        api_key = os.environ.get("FAL_KEY", "").strip()
        if not api_key:
            error_msg = "FAL API key required. Either:\n1. Set the fal_api_key parameter, or\n2. Set FAL_KEY environment variable"
            print(error_msg)
            raise ValueError(error_msg)
        
        # Temporarily set environment variable for this request
        original_key = os.environ.get("FAL_KEY")
        os.environ["FAL_KEY"] = api_key
        
        try:
            # Upload image to fal
            print(f"Starting fal-ai/flux-pro/kontext generation...")
            print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            
            image_url = upload_image_to_fal(image)
            
            # Prepare arguments
            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "enable_safety_checker": enable_safety_checker,
            }
            
            # Add seed if specified
            if seed >= 0:
                arguments["seed"] = seed
                print(f"Using seed: {seed}")
            
            print(f"Calling fal-ai/flux-pro/kontext API...")
            
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        print(f"[fal-ai] {log['message']}")
            
            # Call the API
            try:
                result = fal_client.subscribe(
                    "fal-ai/flux-pro/kontext",
                    arguments=arguments,
                    with_logs=True,
                    on_queue_update=on_queue_update,
                )
            except Exception as api_error:
                error_msg = f"fal.ai API call failed: {str(api_error)}"
                if "401" in str(api_error) or "unauthorized" in str(api_error).lower():
                    error_msg += "\nCheck your FAL API key is valid and has sufficient credits."
                elif "400" in str(api_error) or "bad request" in str(api_error).lower():
                    error_msg += "\nCheck your prompt and image are valid."
                print(error_msg)
                raise RuntimeError(error_msg) from api_error
            
            print("API call successful, processing result...")
            
            # Validate result structure
            if not result:
                raise RuntimeError("Empty result returned from fal.ai API")
            
            if "images" not in result:
                available_keys = list(result.keys()) if isinstance(result, dict) else "N/A"
                raise RuntimeError(f"No 'images' key in result. Available keys: {available_keys}")
            
            if not result["images"] or len(result["images"]) == 0:
                raise RuntimeError("No images returned from API")
            
            # Extract image from result
            image_data = result["images"][0]
            if "url" not in image_data:
                raise RuntimeError("No URL found in image data")
            
            result_image_url = image_data["url"]
            print(f"Generated image URL: {result_image_url}")
            
            # Download the generated image
            try:
                import requests
                response = requests.get(result_image_url, timeout=60)
                response.raise_for_status()
                
                if len(response.content) == 0:
                    raise RuntimeError("Downloaded image is empty")
                
            except requests.exceptions.RequestException as download_error:
                error_msg = f"Failed to download generated image: {str(download_error)}"
                print(error_msg)
                raise RuntimeError(error_msg) from download_error
            
            # Convert to PIL then tensor
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(response.content)
                    temp_file_path = temp_file.name
                
                generated_image = Image.open(temp_file_path)
                result_tensor = pil_to_tensor(generated_image)
                
                # Clean up
                os.unlink(temp_file_path)
                temp_file_path = None
                
            except Exception as conversion_error:
                error_msg = f"Failed to process generated image: {str(conversion_error)}"
                print(error_msg)
                
                # Clean up on error
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                
                raise RuntimeError(error_msg) from conversion_error
            
            print("Image generation completed successfully")
            return (result_tensor,)
                
        except Exception as e:
            # Re-raise known errors without wrapping
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            
            # Wrap unknown errors
            error_msg = f"Unexpected error in fal-ai/flux-pro/kontext generation: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
        
        finally:
            # Restore original API key
            if original_key is not None:
                os.environ["FAL_KEY"] = original_key
            elif "FAL_KEY" in os.environ:
                del os.environ["FAL_KEY"]

# Node registration
NODE_CLASS_MAPPINGS = {
    "FalFluxProKontextNode": FalFluxProKontextNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalFluxProKontextNode": "🎨 Fal Flux Pro Kontext",
}
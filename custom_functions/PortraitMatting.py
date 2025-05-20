import os
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    解析可选的extra_model_paths.yaml文件并将解析的路径添加到sys.path。
    """
    try:
        from utils.extra_config import load_extra_path_config
    except ImportError:
        print("Could not import load_extra_path_config from utils.extra_config")

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS, LoadImage

def init_portrait_matting_models():
    """初始化人像抠图所需的模型"""
    import_custom_nodes()
    
    comfyui_path = find_path("ComfyUI")
    model_root = os.path.join(comfyui_path, "models") if comfyui_path else "models"
    # 人像蒙版模型路径
    human_mask_path = os.path.join(model_root, "human_mask")
    
    with torch.inference_mode():
        # 加载人像模型
        portraitmodelloader = NODE_CLASS_MAPPINGS["PortraitModelLoader"]()
        portrait_models = portraitmodelloader.load_portrait_models(
            model_root=human_mask_path, device="cuda:0 (NVIDIA GeForce RTX 3090)"
        )
        
        return {
            "portrait_models": portrait_models,
        }

def portrait_matting_process(
    matting_models, 
    image_path,
    abs_path=True, 
    is_ndarray=False
):
    """处理人像抠图
    
    Args:
        matting_models: 预加载的模型
        image_path: 图像路径或numpy数组
        size: hivision参数 - 尺寸
        bgcolor: hivision参数 - 背景颜色
        render: hivision参数 - 渲染方式
        kb: hivision参数 - kb值
        dpi: hivision参数 - dpi值
        abs_path: 是否为绝对路径
        is_ndarray: 输入是否为numpy数组
        
    Returns:
        numpy.ndarray: 处理后的图像，带Alpha通道
    """
    with torch.inference_mode():
        # 加载图像
        loadimage = LoadImage()
        loadimage_14 = loadimage.load_image(image=image_path, abs_path=abs_path, is_ndarray=is_ndarray)

        # 设置参数
        zhhivisionparamsnode = NODE_CLASS_MAPPINGS["ZHHivisionParamsNode"]()
        zhhivisionparamsnode_15 = zhhivisionparamsnode.get_params(
            size="研究生考试\t\t(709, 531)", bgcolor="蓝色", render="上下渐变", kb=300, dpi=300
        )

        # 处理图像
        hivisionnode = NODE_CLASS_MAPPINGS["HivisionNode"]()
        hivisionnode_45 = hivisionnode.gen_img(
            face_alignment=True,
            change_bg_only=False,
            crop_only=True,
            matting_model="hivision_modnet",
            face_detect_model="retinaface-resnet50",
            head_measure_ratio=0.2,
            top_distance=0.12,
            whitening_strength=2,
            brightness_strength=0,
            contrast_strength=0,
            saturation_strength=0,
            sharpen_strength=0,
            input_img=get_value_at_index(loadimage_14, 0),
            normal_params=get_value_at_index(zhhivisionparamsnode_15, 0),
        )

        # 生成蒙版
        portraitmaskgenerator = NODE_CLASS_MAPPINGS["PortraitMaskGenerator"]()
        portraitmaskgenerator_48 = portraitmaskgenerator.generate_portrait_mask(
            conf_threshold=0.25,
            iou_threshold=0.5,
            human_targets="person",
            matting_threshold=0.1,
            min_box_area_rate=0.0012,
            image=get_value_at_index(hivisionnode_45, 1),
            portrait_models=get_value_at_index(matting_models["portrait_models"], 0),
        )

        # 应用蒙版
        imagealphamaskreplacer = NODE_CLASS_MAPPINGS["ImageAlphaMaskReplacer"]()
        imagealphamaskreplacer_49 = imagealphamaskreplacer.replace_alpha_with_mask(
            image=get_value_at_index(hivisionnode_45, 1),
            mask=get_value_at_index(portraitmaskgenerator_48, 0),
        )

        # 将处理后的图像转换为numpy数组并返回
        outimg = get_value_at_index(imagealphamaskreplacer_49, 0).squeeze(0) * 255.
        return outimg.cpu().numpy().astype(np.uint8)


if __name__ == "__main__":
    # 测试代码
    import cv2
    from PIL import Image
    
    # 图像路径
    img_path = 'input/4.jpg'
    
    # 两种输入方式
    # 1. 直接使用图像路径
    portrait_matting_models = init_portrait_matting_models()
    for i in range(0, 50):
        final_img = portrait_matting_process(
            portrait_matting_models, 
            img_path
        )
    
    # 2. 使用numpy数组
    # image = np.array(Image.open(img_path))[..., :3]
    # final_img = portrait_matting_process(
    #     portrait_matting_models, 
    #     image, 
    #     is_ndarray=True
    # )
    
    # 保存带Alpha通道的图像
    if final_img.shape[2] == 4:
        cv2.imwrite('output_matting.png', final_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        cv2.imwrite('output_matting.jpg', final_img)

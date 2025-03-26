import os
import random
import sys
import json
import argparse
import contextlib
from typing import Sequence, Mapping, Any, Union
import torch
from enum import Enum
import numpy as np

# 定义全局变量
args = None

class IDPhotoType(Enum):
    """证件照类型枚举类，用于定义不同的证件照规格和参数"""
    ONE_INCH = "one_inch"           # 一寸
    TWO_INCH = "two_inch"           # 二寸
    SMALL_TWO_INCH = "small_two_inch"  # 小二寸
    
    @classmethod
    def get_params(cls, photo_type):
        """获取指定证件照类型的参数
        
        Args:
            photo_type: 证件照类型
            
        Returns:
            dict: 包含尺寸和其他参数的字典
        """
        params = {
            cls.ONE_INCH: {"size": "一寸\t\t(413, 295)", "bgcolor": "蓝色", "kb": 300, "dpi": 300},
            cls.TWO_INCH: {"size": "二寸\t\t(626, 413)", "bgcolor": "蓝色", "kb": 300, "dpi": 300},
            cls.SMALL_TWO_INCH: {"size": "小二寸\t\t(531, 413)", "bgcolor": "蓝色", "kb": 300, "dpi": 300}
        }
        return params.get(photo_type, params[cls.TWO_INCH])
    
    @classmethod
    def get_background_colors(cls):
        """获取支持的背景颜色列表
        
        Returns:
            list: 支持的背景颜色列表
        """
        return ["蓝色", "白色", "红色", "灰色"]

class PromptStyle(Enum):
    """提示词风格枚举类，定义不同的人物形象提示词风格"""
    STANDARD = "standard"     # 标准证件照
    BUSINESS = "business"     # 商务风格
    ACADEMIC = "academic"     # 学术专业风格
    
    @classmethod
    def get_prompt_prefix(cls, style):
        """获取指定风格的提示词前缀
        
        Args:
            style: 提示词风格
            
        Returns:
            str: 该风格的提示词前缀
        """
        prefixes = {
            cls.STANDARD: "An individual in professional attire, with blue background, and ",
            cls.BUSINESS: "A confident business professional with blue background, and ",
            cls.ACADEMIC: "A scholarly individual with blue background, and "
        }
        return prefixes.get(style, prefixes[cls.STANDARD])
    
    @classmethod
    def get_prompt_suffix(cls, style):
        """获取指定风格的提示词后缀
        
        Args:
            style: 提示词风格
            
        Returns:
            str: 该风格的提示词后缀
        """
        suffixes = {
            cls.STANDARD: ", and wear a dark blazer over a white dress shirt, complemented by a blue tie with striped patterns. The person has a neutral facial expression suitable for formal identification.",
            cls.BUSINESS: ", and wear a charcoal suit over a white dress shirt, complemented by a burgundy tie with striped patterns. The background is a simple light gray, highlighting the neat and formal appearance of the individual.",
            cls.ACADEMIC: ", wearing scholarly attire including a well-pressed oxford shirt and understated tie. A thoughtful expression convey intellectual focus and academic credibility."
        }
        return suffixes.get(style, suffixes[cls.STANDARD])


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
        # path = os.getcwd()
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
        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")

        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    import yaml
    import folder_paths
    
    extra_model_paths_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extra_model_paths.yaml"
    )
    
    if not os.path.exists(extra_model_paths_file):
        return
    
    with open(extra_model_paths_file, "r") as file:
        extra_paths = yaml.safe_load(file)
    
    if extra_paths is None:
        return
    
    for path_type, paths in extra_paths.items():
        for path in paths:
            folder_paths.add_model_folder_path(path_type, path)

add_comfyui_directory_to_sys_path()
# add_extra_model_paths()

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
    init_extra_nodes(init_custom_nodes=True)

from nodes import LoadImage, NODE_CLASS_MAPPINGS
import folder_paths

def init_idphoto_models():
    """初始化并加载证件照生成所需的所有模型"""
    try:
        import_custom_nodes()
        
        # 获取ComfyUI根目录
        comfyui_path = find_path("ComfyUI")
        model_root = os.path.join(comfyui_path, "models") if comfyui_path else "models"
        
        with torch.inference_mode():
            # 加载UNET模型
            unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
            unetloader_31 = unetloader.load_unet(
                unet_name="Flux_Fill_dev_fp8_e4m3fn.safetensors", weight_dtype="fp8_e4m3fn"
            )

            # 加载差分扩散模型
            differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
            differentialdiffusion_39 = differentialdiffusion.apply(
                model=get_value_at_index(unetloader_31, 0)
            )

            # 加载CLIP模型
            dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
            dualcliploader_34 = dualcliploader.load_clip(
                clip_name1="clip_l.safetensors",
                clip_name2="t5xxl_fp8_e4m3fn.safetensors",
                type="flux",
            )

            # 加载Lora模型
            loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
            loraloader_76 = loraloader.load_lora(
                lora_name="flux1-portrait-lora.safetensors",
                strength_model=0.8,
                strength_clip=0.8,
                model=get_value_at_index(differentialdiffusion_39, 0),
                clip=get_value_at_index(dualcliploader_34, 0),
            )

            # 加载VAE模型
            vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
            vaeloader_32 = vaeloader.load_vae(vae_name="ae.safetensors")

            # 加载Florence2模型
            layermask_loadflorence2model = NODE_CLASS_MAPPINGS[
                "LayerMask: LoadFlorence2Model"
            ]()
            layermask_loadflorence2model_111 = layermask_loadflorence2model.load(
                version="CogFlorence-2.1-Large"
            )

            # 加载人脸模型
            facemodelloader = NODE_CLASS_MAPPINGS["FaceModelLoader"]()
            facemodelloader_72 = facemodelloader.load_face_models(model_root=model_root)

            # 加载核心文本编码组件
            cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            cr_text = NODE_CLASS_MAPPINGS["CR Text"]()
            cr_text_concatenate = NODE_CLASS_MAPPINGS["CR Text Concatenate"]()

            # 返回所有加载的模型
            return {
                "unetloader_31": unetloader_31,
                "differentialdiffusion_39": differentialdiffusion_39,
                "dualcliploader_34": dualcliploader_34,
                "loraloader_76": loraloader_76,
                "vaeloader_32": vaeloader_32,
                "layermask_loadflorence2model_111": layermask_loadflorence2model_111,
                "facemodelloader_72": facemodelloader_72,
                "cliptextencode": cliptextencode,
                "cr_text": cr_text,
                "cr_text_concatenate": cr_text_concatenate
            }
    except Exception as e:
        print(f"初始化模型时出错: {str(e)}")
        raise

def idphoto_process(inited_models, image_path, photo_type=IDPhotoType.TWO_INCH, bg_color="蓝色", 
                   prompt_style=PromptStyle.STANDARD, text_prompt=None, 
                   abs_path=True, is_ndarray=False, queue_size=1):
    """
    使用已初始化的模型处理图像生成证件照
    
    Args:
        inited_models: 初始化好的模型字典
        image_path: 输入图像路径或图像数组
        photo_type: 证件照类型，默认为TWO_INCH
        bg_color: 背景颜色，默认为蓝色（在输出RGBA时仅用于参数设置）
        prompt_style: 提示词风格，默认为STANDARD
        text_prompt: 可选的文本提示，用于AI生成的描述补充
        abs_path: 是否为绝对路径
        is_ndarray: 输入是否为numpy数组
        queue_size: 处理队列大小
        
    Returns:
        numpy.ndarray: 处理后的证件照图像数组（RGBA格式）
    """
    try:
        with torch.inference_mode(), ctx:
            # 解包预加载的模型
            unetloader_31 = inited_models["unetloader_31"]
            differentialdiffusion_39 = inited_models["differentialdiffusion_39"]
            dualcliploader_34 = inited_models["dualcliploader_34"]
            loraloader_76 = inited_models["loraloader_76"]
            vaeloader_32 = inited_models["vaeloader_32"]
            layermask_loadflorence2model_111 = inited_models["layermask_loadflorence2model_111"]
            facemodelloader_72 = inited_models["facemodelloader_72"]
            cliptextencode = inited_models["cliptextencode"]
            cr_text = inited_models["cr_text"]
            cr_text_concatenate = inited_models["cr_text_concatenate"]

            # 获取所选证件照类型的参数
            photo_params = IDPhotoType.get_params(photo_type)
            
            # 加载输入图像
            loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            loadimage_67 = loadimage.load_image(image=image_path, abs_path=abs_path, is_ndarray=is_ndarray)
            # 清理显存
            torch.cuda.empty_cache()
            # 设置证件照参数
            zhhivisionparamsnode = NODE_CLASS_MAPPINGS["ZHHivisionParamsNode"]()
            zhhivisionparamsnode_68 = zhhivisionparamsnode.get_params(
                size=photo_params["size"], 
                bgcolor=bg_color, 
                render="纯色", 
                kb=photo_params["kb"], 
                dpi=photo_params["dpi"]
            )
            # 清理显存
            torch.cuda.empty_cache()
            # 处理图像，去除背景并应用证件照参数
            hivisionnode = NODE_CLASS_MAPPINGS["HivisionNode"]()
            hivisionnode_66 = hivisionnode.gen_img(
                face_alignment=True,
                change_bg_only=False,
                crop_only=False,
                matting_model="rmbg2_fp16",
                face_detect_model="retinaface-resnet50",
                head_measure_ratio=0.2,
                top_distance=0.12,
                whitening_strength=2,
                brightness_strength=0,
                contrast_strength=0,
                saturation_strength=0,
                sharpen_strength=0,
                input_img=get_value_at_index(loadimage_67, 0),
                normal_params=get_value_at_index(zhhivisionparamsnode_68, 0),
            )
            
            # 将RGBA转换为RGB（为了分析）
            rgbatorgbconverter = NODE_CLASS_MAPPINGS["RGBAToRGBConverter"]()
            rgbatorgbconverter_69 = rgbatorgbconverter.convert_rgba_to_rgb(
                bg_color="白色",
                custom_r=0.5,
                custom_g=0.5,
                custom_b=0.5,
                image=get_value_at_index(hivisionnode_66, 1),
            )

            # 使用Florence2模型获取图像描述
            layerutility_florence2image2prompt = NODE_CLASS_MAPPINGS[
                "LayerUtility: Florence2Image2Prompt"
            ]()
            layerutility_florence2image2prompt_112 = (
                layerutility_florence2image2prompt.florence2_image2prompt(
                    task="detailed caption",
                    text_input="",
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                    fill_mask=False,
                    florence2_model=get_value_at_index(layermask_loadflorence2model_111, 0),
                    image=get_value_at_index(rgbatorgbconverter_69, 0),
                )
            )

            # 提取发型描述
            texthairextractor = NODE_CLASS_MAPPINGS["TextHairExtractor"]()
            texthairextractor_113 = texthairextractor.extract_hair_phrase(
                input_text=get_value_at_index(layerutility_florence2image2prompt_112, 0)
            )

            # 获取选定风格的提示词前缀和后缀
            base_prompt = PromptStyle.get_prompt_prefix(prompt_style)
            prompt_suffix = PromptStyle.get_prompt_suffix(prompt_style)
            
            # 组合提示文本
            if text_prompt:
                custom_text = text_prompt
            else:
                custom_text = get_value_at_index(texthairextractor_113, 0)
                custom_text += prompt_suffix
            
            # 负面提示词
            negative_prompt = "ng_deepnegative_v1_75t,(badhandv4:1.2),EasyNegative,(worst quality:2),"
            
            # 编码提示词
            cliptextencode_7 = cliptextencode.encode(
                text=negative_prompt,
                clip=get_value_at_index(loraloader_76, 1),
            )
            
            cr_text_115 = cr_text.text_multiline(text=base_prompt)
            
            showtextpysssss = NODE_CLASS_MAPPINGS["ShowText|pysssss"]()
            showtextpysssss_114 = showtextpysssss.notify(
                text=custom_text,
                unique_id=14453620076813389785,
            )
            
            cr_text_concatenate_116 = cr_text_concatenate.concat_text(
                text1=get_value_at_index(cr_text_115, 0),
                text2=get_value_at_index(showtextpysssss_114, 0),
                separator="",
            )
            
            cliptextencode_23 = cliptextencode.encode(
                text=get_value_at_index(cr_text_concatenate_116, 0),
                clip=get_value_at_index(loraloader_76, 1),
            )

            # 设置FluxGuidance
            fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
            fluxguidance_26 = fluxguidance.append(
                guidance=30, conditioning=get_value_at_index(cliptextencode_23, 0)
            )

            # 调整图像大小
            imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
            imageresizekj_49 = imageresizekj.resize(
                width=768,
                height=1024,
                upscale_method="lanczos",
                keep_proportion=True,
                divisible_by=4,
                crop="disabled",
                image=get_value_at_index(rgbatorgbconverter_69, 0),
            )

            # 生成人脸蒙版
            facemaskgenerator = NODE_CLASS_MAPPINGS["FaceMaskGenerator"]()
            facemaskgenerator_71 = facemaskgenerator.generate_faces_mask(
                max_face=False,
                num_faces=1,
                dilate_pixels=5,
                image=get_value_at_index(imageresizekj_49, 0),
                face_models=get_value_at_index(facemodelloader_72, 0),
            )

            # 为Inpaint准备条件
            inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
            inpaintmodelconditioning_38 = inpaintmodelconditioning.encode(
                noise_mask=True,
                positive=get_value_at_index(fluxguidance_26, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                vae=get_value_at_index(vaeloader_32, 0),
                pixels=get_value_at_index(imageresizekj_49, 0),
                mask=get_value_at_index(facemaskgenerator_71, 0),
            )

            # 创建处理节点对象
            ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            imagemaskblender = NODE_CLASS_MAPPINGS["ImageMaskBlender"]()
            
            # 执行处理循环
            final_result = None
            # 清理显存
            torch.cuda.empty_cache()
            for q in range(queue_size):
                # KSampler生成
                ksampler_3 = ksampler.sample(
                    seed=777,
                    steps=10,
                    cfg=2,
                    sampler_name="euler",
                    scheduler="normal",
                    denoise=1,
                    model=get_value_at_index(loraloader_76, 0),
                    positive=get_value_at_index(inpaintmodelconditioning_38, 0),
                    negative=get_value_at_index(inpaintmodelconditioning_38, 1),
                    latent_image=get_value_at_index(inpaintmodelconditioning_38, 2),
                )

                # VAE解码
                vaedecode_8 = vaedecode.decode(
                    samples=get_value_at_index(ksampler_3, 0),
                    vae=get_value_at_index(vaeloader_32, 0),
                )

                # 调整图像大小
                imageresizekj_92 = imageresizekj.resize(
                    width=1024,
                    height=512,
                    upscale_method="lanczos",
                    keep_proportion=True,
                    divisible_by=2,
                    crop="disabled",
                    image=get_value_at_index(vaedecode_8, 0),
                    get_image_size=get_value_at_index(rgbatorgbconverter_69, 0),
                )

                # 生成人脸蒙版
                facemaskgenerator_93 = facemaskgenerator.generate_faces_mask(
                    max_face=False,
                    num_faces=1,
                    dilate_pixels=5,
                    image=get_value_at_index(rgbatorgbconverter_69, 0),
                    face_models=get_value_at_index(facemodelloader_72, 0),
                )

                # 混合图像
                imagemaskblender_89 = imagemaskblender.blend_images(
                    feather=20,
                    image_fg=get_value_at_index(imageresizekj_92, 0),
                    image_bg=get_value_at_index(rgbatorgbconverter_69, 0),
                    mask=get_value_at_index(facemaskgenerator_93, 0),
                )

                # 生成最终证件照（保留Alpha通道）
                hivisionnode_84 = hivisionnode.gen_img(
                    face_alignment=True,
                    change_bg_only=True,
                    crop_only=False,
                    matting_model="rmbg2_fp16",
                    face_detect_model="retinaface-resnet50",
                    head_measure_ratio=0.2,
                    top_distance=0.12,
                    whitening_strength=2,
                    brightness_strength=0,
                    contrast_strength=0,
                    saturation_strength=0,
                    sharpen_strength=0,
                    input_img=get_value_at_index(imagemaskblender_89, 0),
                    normal_params=get_value_at_index(zhhivisionparamsnode_68, 0),
                )
                
                # 添加背景节点处理
                addbackgroundnode = NODE_CLASS_MAPPINGS["AddBackgroundNode"]()
                addbackgroundnode_85 = addbackgroundnode.gen_img(
                    input_img=get_value_at_index(hivisionnode_84, 1),
                    normal_params=get_value_at_index(zhhivisionparamsnode_68, 0),
                )

                # 后期处理节点
                laterprocessnode = NODE_CLASS_MAPPINGS["LaterProcessNode"]()
                laterprocessnode_86 = laterprocessnode.gen_img(
                    input_img=get_value_at_index(addbackgroundnode_85, 0),
                    normal_params=get_value_at_index(zhhivisionparamsnode_68, 0),
                )
                
                # 直接使用带透明通道的图像（第二个输出）
                final_result = get_value_at_index(laterprocessnode_86, 0)
            # 清理显存
            torch.cuda.empty_cache()
            # 清理显存
            torch.cuda.empty_cache()
            # 返回结果图像（带Alpha通道）
            if final_result is not None:
                outimg = final_result * 255.0
                return outimg.cpu().numpy().astype(np.uint8)[0]
            return None
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        raise


ctx = contextlib.nullcontext()

def parse_arg(s: Any):
    """Parses a JSON string, returning it unchanged if the parsing fails."""
    if __name__ == "__main__" or not isinstance(s, str):
        return s

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


parser = argparse.ArgumentParser(
    description="证件照生成工作流。支持不同类型的证件照和提示词风格。"
)
parser.add_argument(
    "--photo-type",
    "-t",
    dest="photo_type",
    type=str,
    choices=[t.value for t in IDPhotoType],
    default=IDPhotoType.TWO_INCH.value,
    help=f"指定证件照类型，可选值: {', '.join([t.value for t in IDPhotoType])} (默认: {IDPhotoType.TWO_INCH.value})",
)

parser.add_argument(
    "--bg-color",
    "-b",
    dest="bg_color",
    type=str,
    choices=IDPhotoType.get_background_colors(),
    default="蓝色",
    help=f"指定背景参数设置，可选值: {', '.join(IDPhotoType.get_background_colors())} (默认: 蓝色)",
)

parser.add_argument(
    "--prompt-style",
    "-s",
    dest="prompt_style",
    type=str,
    choices=[s.value for s in PromptStyle],
    default=PromptStyle.STANDARD.value,
    help=f"指定提示词风格，可选值: {', '.join([s.value for s in PromptStyle])} (默认: {PromptStyle.STANDARD.value})",
)

parser.add_argument(
    "--image-path",
    "-i",
    dest="image_path",
    type=str,
    default="input.jpg",
    help="输入图像路径 (默认: input.jpg)",
)

parser.add_argument(
    "--custom-prompt",
    "-p",
    dest="custom_prompt",
    type=str,
    default=None,
    help="自定义文本提示，用于描述照片中的人物形象 (默认: 自动生成)",
)

parser.add_argument(
    "--queue-size",
    "-q",
    type=int,
    default=1,
    help="工作流执行次数 (默认: 1)",
)

parser.add_argument(
    "--comfyui-directory",
    "-c",
    default=None,
    help="ComfyUI的目录位置 (默认: 当前目录)",
)

parser.add_argument(
    "--output",
    "-o",
    default=None,
    help="输出图像保存位置. 可以是文件路径, 目录, 或 - 表示标准输出 (默认: ComfyUI输出目录)",
)

parser.add_argument(
    "--disable-metadata",
    action="store_true",
    help="禁用在输出中写入工作流元数据",
)

def main(*func_args, **func_kwargs):
    global args
    if __name__ == "__main__":
        if args is None:
            args = parser.parse_args()
    else:
        defaults = dict(
            (arg, parser.get_default(arg))
            for arg in ["queue_size", "comfyui_directory", "output", "disable_metadata"]
        )
        ordered_args = dict(zip([], func_args))

        all_args = dict()
        all_args.update(defaults)
        all_args.update(ordered_args)
        all_args.update(func_kwargs)

        args = argparse.Namespace(**all_args)

    # 初始化模型
    inited_models = init_idphoto_models()
    
    # 检查是否传入了photo_type参数
    photo_type = None
    if hasattr(args, 'photo_type') and args.photo_type:
        # 尝试从字符串获取对应的IDPhotoType枚举
        try:
            photo_type = IDPhotoType(args.photo_type)
        except (ValueError, KeyError):
            # 如果无效，使用默认值
            photo_type = IDPhotoType.TWO_INCH
    else:
        # 默认使用二寸证件照
        photo_type = IDPhotoType.TWO_INCH
    
    # 检查提示词风格
    prompt_style = None
    if hasattr(args, 'prompt_style') and args.prompt_style:
        # 尝试从字符串获取对应的PromptStyle枚举
        try:
            prompt_style = PromptStyle(args.prompt_style)
        except (ValueError, KeyError):
            # 如果无效，使用默认值
            prompt_style = PromptStyle.STANDARD
    else:
        # 默认使用标准风格
        prompt_style = PromptStyle.STANDARD
    
    # 检查图像路径
    image_path = "input/4.jpg"
    if hasattr(args, 'image_path') and args.image_path:
        image_path = args.image_path
    
    # 检查背景颜色
    bg_color = "蓝色"
    if hasattr(args, 'bg_color') and args.bg_color:
        bg_color = args.bg_color
        
    # 检查自定义提示
    custom_prompt = None
    if hasattr(args, 'custom_prompt') and args.custom_prompt:
        custom_prompt = args.custom_prompt
    
    # 使用预加载的模型进行处理
    return idphoto_process(
        inited_models=inited_models,
        image_path=image_path,
        photo_type=photo_type,
        bg_color=bg_color,
        prompt_style=prompt_style,
        text_prompt=custom_prompt,
        queue_size=args.queue_size
    )

import cv2
if __name__ == "__main__":
    result = main()
    # 将结果保存为图像文件（带Alpha通道）
    output_path = "output/idphoto_result.png"
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))
    print(f"结果已保存到 {output_path}")

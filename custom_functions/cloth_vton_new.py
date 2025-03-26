import os
import random
import sys
import json
import argparse
import contextlib
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np
import datetime
import cv2
from enum import Enum
from PIL import Image

# 设置上下文管理器
ctx = contextlib.nullcontext()


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """返回序列或映射中给定索引处的值。"""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """递归查找指定名称的文件或目录路径。"""
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))

    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} 已找到: {path_name}")
        return path_name

    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None

    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """将'ComfyUI'添加到sys.path"""
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")

        print(f"'{comfyui_path}' 已添加到sys.path")


def add_extra_model_paths() -> None:
    """解析extra_model_paths.yaml文件并添加路径到sys.path。"""
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("无法找到extra_model_paths配置文件。")

add_comfyui_directory_to_sys_path()
# add_extra_model_paths()

def import_custom_nodes() -> None:
    """导入所有自定义节点"""
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()

from nodes import NODE_CLASS_MAPPINGS, LoadImage

def init_vton_models():
    """初始化并加载虚拟换装所需的所有模型"""
    import_custom_nodes()
    
    # 获取ComfyUI根目录
    comfyui_path = find_path("ComfyUI")
    model_root = os.path.join(comfyui_path, "models") if comfyui_path else "models"
    human_mask_path = os.path.join(model_root, "human_mask")
    with torch.inference_mode():
        # 加载CLIP模型
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_result = dualcliploader.load_clip(
            clip_name1="clip_l.safetensors",
            clip_name2="t5xxl_fp8_e4m3fn.safetensors",
            type="flux",
        )
        
        # 将CLIP模型放在指定GPU上
        overrideclipdevice = NODE_CLASS_MAPPINGS["OverrideCLIPDevice"]()
        overrided_clip = overrideclipdevice.patch(
            device="cuda:1", 
            clip=get_value_at_index(dualcliploader_result, 0)
        )
        
        # 加载UNET模型
        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_result = unetloader.load_unet(
            unet_name="Flux_Fill_dev_fp8_e4m3fn.safetensors",
            weight_dtype="fp8_e4m3fn",
        )
        
        # 加载VAE模型
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_result = vaeloader.load_vae(vae_name="ae.safetensors")
        
        # 加载人脸模型
        facemodelloader = NODE_CLASS_MAPPINGS["FaceModelLoader"]()
        facemodelloader_result = facemodelloader.load_face_models(model_root=model_root)
        
        # 加载人像模型
        portraitmodelloader = NODE_CLASS_MAPPINGS["PortraitModelLoader"]()
        portraitmodelloader_result = portraitmodelloader.load_portrait_models(
            model_root=human_mask_path
        )
        
        # 加载ControlNet模型
        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        controlnetloader_result = controlnetloader.load_controlnet(
            control_net_name="flux1_instantx_union_control_pro.safetensors"
        )
        
        # 加载SR Blend模型
        srblendbuildpipe = NODE_CLASS_MAPPINGS["SrBlendBuildPipe"]()
        srblendbuildpipe_result = srblendbuildpipe.load_models(
            lut_path="lut", gpu_choose="cuda:0", sr_type="ESRGAN", half=True
        )
        
        # 加载Lora模型
        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        loraloadermodelonly_result = loraloadermodelonly.load_lora_model_only(
            lora_name="flux1_catvton_lora.safetensors",
            strength_model=1,
            model=get_value_at_index(unetloader_result, 0),
        )
        
        return {
            "clip": overrided_clip,  # 使用重定向后的CLIP模型
            "unet": unetloader_result,
            "vae": vaeloader_result,
            "face_model": facemodelloader_result,
            "portrait_model": portraitmodelloader_result,
            "controlnet": controlnetloader_result,
            "srblend": srblendbuildpipe_result,
            "lora": loraloadermodelonly_result
        }

def cloth_vton_process(models, clothing_name, model_image_path, seed=None, abs_path=True, is_ndarray=False):
    """处理图像，执行虚拟换装
    
    Args:
        models: 初始化的模型字典
        clothing_name: 服装图片名称（不含.jpg后缀，如'Tchaoxianfu'）
        model_image_path: 模特图像的路径或数组
        seed: 随机种子，默认为None（随机生成）
        abs_path: 是否为绝对路径
        is_ndarray: 输入是否为numpy数组
        
    Returns:
        处理后的图像（numpy数组）
    """
    
    with torch.inference_mode():
        # 准备文本编码
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        negative_prompt = cliptextencode.encode(
            text="ng_deepnegative_v1_75t,(badhandv4:1.2),EasyNegative,(worst quality:2),",
            clip=get_value_at_index(models["clip"], 0),
        )
        
        positive_prompt = cliptextencode.encode(
            text="The pair of images highlights a clothing and its styling on a model, high resolution, 4K, 8K; [IMAGE1] Detailed product shot of a clothing [IMAGE2] The same cloth is worn by a model in a lifestyle setting.",
            clip=get_value_at_index(models["clip"], 0),
        )
        
        # 添加Flux引导
        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        fluxguidance_result = fluxguidance.append(
            guidance=30, conditioning=get_value_at_index(positive_prompt, 0)
        )
        
        # 加载图像
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        # 从templates/vton_clothes/目录加载服装图片
        clothing_image = loadimage.load_image(
            image=f"../templates/vton_clothes/{clothing_name}.jpg",
            abs_path=False
        )
        model_image = loadimage.load_image(image=model_image_path, abs_path=abs_path, is_ndarray=is_ndarray)
        
        # 调整图像大小
        imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
        resized_model = imageresizekj.resize(
            width=768,
            height=1024,
            upscale_method="lanczos",
            keep_proportion=True,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(model_image, 0),
        )
        torch.cuda.empty_cache()
        # 生成人脸蒙版
        facemaskgenerator = NODE_CLASS_MAPPINGS["FaceMaskGenerator"]()
        face_mask = facemaskgenerator.generate_faces_mask(
            max_face=False,
            num_faces=1,
            dilate_pixels=10,
            image=get_value_at_index(resized_model, 0),
            face_models=get_value_at_index(models["face_model"], 0),
        )
        torch.cuda.empty_cache()
        # 生成人像蒙版
        portraitmaskgenerator = NODE_CLASS_MAPPINGS["PortraitMaskGenerator"]()
        portrait_mask = portraitmaskgenerator.generate_portrait_mask(
            conf_threshold=0.25,
            iou_threshold=0.5,
            human_targets="person",
            matting_threshold=0.1,
            min_box_area_rate=0.0012,
            image=get_value_at_index(resized_model, 0),
            portrait_models=get_value_at_index(models["portrait_model"], 0),
        )
        
        # 蒙版处理
        maskmorphology = NODE_CLASS_MAPPINGS["MaskMorphology"]()
        morphed_mask = maskmorphology.process_mask(
            pixels=100,
            use_split_mode=False,
            upper_pixels=5,
            lower_pixels=5,
            feather_split=0,
            mask=get_value_at_index(portrait_mask, 0),
        )
        
        # 合成蒙版
        maskcomposite = NODE_CLASS_MAPPINGS["MaskComposite"]()
        combined_mask = maskcomposite.combine(
            x=0,
            y=0,
            operation="and",
            destination=get_value_at_index(face_mask, 0),
            source=get_value_at_index(morphed_mask, 0),
        )
        
        # 为ICLora添加蒙版
        addmaskforiclora = NODE_CLASS_MAPPINGS["AddMaskForICLora"]()
        masked_images = addmaskforiclora.add_mask(
            patch_mode="patch_right",
            output_length=1536,
            patch_color="#FF0000",
            first_image=get_value_at_index(clothing_image, 0),
            second_image=get_value_at_index(resized_model, 0),
            second_mask=get_value_at_index(combined_mask, 0),
        )
        
        # Inpaint模型条件设置
        inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
        inpaint_conditioning = inpaintmodelconditioning.encode(
            noise_mask=True,
            positive=get_value_at_index(fluxguidance_result, 0),
            negative=get_value_at_index(negative_prompt, 0),
            vae=get_value_at_index(models["vae"], 0),
            pixels=get_value_at_index(masked_images, 0),
            mask=get_value_at_index(masked_images, 1),
        )
        
        # 设置ControlNet类型
        setunioncontrolnettype = NODE_CLASS_MAPPINGS["SetUnionControlNetType"]()
        controlnet_type = setunioncontrolnettype.set_controlnet_type(
            type="depth", control_net=get_value_at_index(models["controlnet"], 0)
        )
        torch.cuda.empty_cache()
        # 深度预处理
        depthanythingv2preprocessor = NODE_CLASS_MAPPINGS["DepthAnythingV2Preprocessor"]()
        depth_result = depthanythingv2preprocessor.execute(
            ckpt_name="depth_anything_v2_vitl.pth",
            resolution=1024,
            image=get_value_at_index(resized_model, 0),
        )
        torch.cuda.empty_cache()
        # 应用ControlNet
        controlnetapplysd3 = NODE_CLASS_MAPPINGS["ControlNetApplySD3"]()
        controlnet_applied = controlnetapplysd3.apply_controlnet(
            strength=0.6,
            start_percent=0,
            end_percent=0.8,
            positive=get_value_at_index(inpaint_conditioning, 0),
            negative=get_value_at_index(inpaint_conditioning, 1),
            control_net=get_value_at_index(controlnet_type, 0),
            vae=get_value_at_index(models["vae"], 0),
            image=get_value_at_index(depth_result, 0),
        )
        torch.cuda.empty_cache()
        # 设置种子
        if seed is None:
            seed = random.randint(1, 2 ** 64)
            
        # 采样过程
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        sampler_result = ksampler.sample(
            seed=710874769520552,
            steps=15,
            cfg=3.5,
            sampler_name="euler",
            scheduler="normal",
            denoise=1,
            model=get_value_at_index(models["lora"], 0),
            positive=get_value_at_index(controlnet_applied, 0),
            negative=get_value_at_index(controlnet_applied, 1),
            latent_image=get_value_at_index(inpaint_conditioning, 2),
        )
        
        # VAE解码
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        decoded_image = vaedecode.decode(
            samples=get_value_at_index(sampler_result, 0),
            vae=get_value_at_index(models["vae"], 0),
        )
        
        # 提取右半部分图像
        imagehalfextractor = NODE_CLASS_MAPPINGS["ImageHalfExtractor"]()
        right_half = imagehalfextractor.extract_right_half(
            image=get_value_at_index(decoded_image, 0)
        )
        
        # 转换为NumPy格式
        converttensortonumpy = NODE_CLASS_MAPPINGS["ConvertTensorToNumpy"]()
        numpy_image = converttensortonumpy.convert(
            image=get_value_at_index(right_half, 0)
        )
        
        # 应用SR增强
        srblendprocess = NODE_CLASS_MAPPINGS["SrBlendProcess"]()
        enhanced_image = srblendprocess.enhance_process(
            is_front="False",
            model=get_value_at_index(models["srblend"], 0),
            src_img=get_value_at_index(numpy_image, 0),
        )
        
        # 调整到目标尺寸
        resized_result = imageresizekj.resize(
            width=512,
            height=512,
            upscale_method="lanczos",
            keep_proportion=True,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(enhanced_image, 1),
            get_image_size=get_value_at_index(model_image, 0),
        )
        
        # 生成最终的人脸蒙版
        final_face_mask = facemaskgenerator.generate_faces_mask(
            max_face=False,
            num_faces=1,
            dilate_pixels=5,
            image=get_value_at_index(model_image, 0),
            face_models=get_value_at_index(models["face_model"], 0),
        )
        
        # 图像混合
        imagemaskblender = NODE_CLASS_MAPPINGS["ImageMaskBlender"]()
        final_result = imagemaskblender.blend_images(
            feather=30,
            image_fg=get_value_at_index(resized_result, 0),
            image_bg=get_value_at_index(model_image, 0),
            mask=get_value_at_index(final_face_mask, 0),
        )
        torch.cuda.empty_cache()
        # 获取numpy格式的最终结果
        final_img = get_value_at_index(final_result, 0) * 255.0
        return final_img.cpu().numpy().astype(np.uint8)[0]

# 命令行入口函数
def main():
    # 参数解析
    parser = argparse.ArgumentParser(
        description="虚拟换装工具。以下是必需的输入参数。"
    )
    parser.add_argument(
        "--clothing", required=True, help="服装图片名称（不含.jpg后缀，如'Tchaoxianfu'）"
    )
    parser.add_argument(
        "--model", required=True, help="模特图像路径"
    )
    parser.add_argument(
        "--output", required=True, help="输出图像路径"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="随机种子（可选）"
    )
    
    args = parser.parse_args()
    
    # 初始化模型
    print("正在加载模型...")
    models = init_vton_models()
    
    # 处理图像
    print("开始处理图像...")
    result = cloth_vton_process(
        models,
        args.clothing,
        args.model,
        seed=args.seed,
        abs_path=True
    )
    
    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"结果已保存至: {args.output}")

if __name__ == "__main__":
    main()

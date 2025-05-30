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
    """导入所有自定义节点"""
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # 初始化自定义节点
    init_extra_nodes(init_custom_nodes=True)


def init_vton_models():
    """初始化并加载虚拟换装所需的所有模型"""
    import_custom_nodes()
    
    from nodes import NODE_CLASS_MAPPINGS
    
    # 获取ComfyUI根目录
    comfyui_path = find_path("ComfyUI")
    model_root = os.path.join(comfyui_path, "models") if comfyui_path else "models"
    human_mask_path = os.path.join(model_root, "human_mask")
    
    with torch.inference_mode():
        # 加载CLIP模型
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_result = dualcliploader.load_clip(
            clip_name1="ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors",
            clip_name2="t5xxl_fp16.safetensors",
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
        
        # 重定向VAE设备
        overridevaedevice = NODE_CLASS_MAPPINGS["OverrideVAEDevice"]()
        overrided_vae = overridevaedevice.patch(
            device="cuda:1", 
            vae=get_value_at_index(vaeloader_result, 0)
        )
        
        # 加载人脸模型
        facemodelloader = NODE_CLASS_MAPPINGS["FaceModelLoader"]()
        facemodelloader_result = facemodelloader.load_face_models(
            model_root=model_root,
            device="cuda:1 (NVIDIA GeForce RTX 3090)"
        )
        
        # 加载人像模型
        portraitmodelloader = NODE_CLASS_MAPPINGS["PortraitModelLoader"]()
        portraitmodelloader_result = portraitmodelloader.load_portrait_models(
            model_root=human_mask_path,
            device="cuda:1 (NVIDIA GeForce RTX 3090)"
        )
        
        # 加载UNET GGUF模型
        unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unetloadergguf_result = unetloadergguf.load_unet(unet_name="flux1-dev-Q8_0.gguf")
        
        # 加载SR Blend模型
        srblendbuildpipe = NODE_CLASS_MAPPINGS["SrBlendBuildPipe"]()
        srblendbuildpipe_result = srblendbuildpipe.load_models(
            lut_path="lut", gpu_choose="cuda:1", sr_type="ESRGAN", half=True
        )
        
        # # 加载Florence2模型
        # layermask_loadflorence2model = NODE_CLASS_MAPPINGS["LayerMask: LoadFlorence2Model"]()
        # layermask_loadflorence2model_result = layermask_loadflorence2model.load(
        #     version="CogFlorence-2.1-Large"
        # )
        
        # 加载Lora模型
        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        catvton_lora_result = loraloadermodelonly.load_lora_model_only(
            lora_name="flux1_catvton_lora.safetensors",
            strength_model=1,
            model=get_value_at_index(unetloader_result, 0),
        )
        
        # 添加FluxForwardOverrider节点
        fluxforwardoverrider = NODE_CLASS_MAPPINGS["FluxForwardOverrider"]()
        fluxforwardoverrider_result = fluxforwardoverrider.apply_patch(
            model=get_value_at_index(catvton_lora_result, 0)
        )
        
        # 添加ApplyTeaCachePatch节点
        applyteacachepatch = NODE_CLASS_MAPPINGS["ApplyTeaCachePatch"]()
        applyteacachepatch_result = applyteacachepatch.apply_patch(
            rel_l1_thresh=0.25,
            cache_device="main_device",
            wan_coefficients="disabled",
            model=get_value_at_index(fluxforwardoverrider_result, 0),
        )
        
        # 加载增强Lora模型
        enhance_lora_result = loraloadermodelonly.load_lora_model_only(
            lora_name="flux1_enhance_lora.safetensors",
            strength_model=0.35,
            model=get_value_at_index(unetloadergguf_result, 0),
        )
        
        # 为增强Lora模型添加FluxForwardOverrider
        fluxforwardoverrider_enhance = fluxforwardoverrider.apply_patch(
            model=get_value_at_index(enhance_lora_result, 0)
        )
        
        # 为增强Lora模型添加ApplyTeaCachePatch
        applyteacachepatch_enhance = applyteacachepatch.apply_patch(
            rel_l1_thresh=0.25,
            cache_device="main_device",
            wan_coefficients="disabled",
            model=get_value_at_index(fluxforwardoverrider_enhance, 0),
        )
        
        return {
            "clip": overrided_clip,
            "unet": unetloader_result,
            "vae": overrided_vae,
            "face_model": facemodelloader_result,
            "portrait_model": portraitmodelloader_result,
            "unet_gguf": unetloadergguf_result,
            "srblend": srblendbuildpipe_result,
            # "florence2_model": layermask_loadflorence2model_result,
            "catvton_lora": catvton_lora_result,
            "enhance_lora": enhance_lora_result,
            "catvton_lora_patched": applyteacachepatch_result,
            "enhance_lora_patched": applyteacachepatch_enhance
        }


def cloth_vton_process(models, clothing_path, model_image_path, is_clothing_template=True, seed=None, abs_path=True, is_ndarray=False,):
    """处理图像，执行增强版虚拟换装
    
    Args:
        models: 初始化的模型字典
        clothing_name: 服装图片名称（不含.jpg后缀，如'Tmiaofu'），当clothing_image_path为None时使用
        model_image_path: 模特图像的路径或数组
        seed: 随机种子，默认为None（使用固定值710874769520552）
        abs_path: 是否为绝对路径
        is_ndarray: 输入是否为numpy数组
        clothing_image_path: 服装图像的路径或数组，指定时将忽略clothing_name
        
    Returns:
        处理后的图像（numpy数组）
    """
    
    from nodes import NODE_CLASS_MAPPINGS
    
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
        
        # 服装图像加载，支持两种方式
        if is_clothing_template:
            # 从本地模板目录加载服装图片
            clothing_image = loadimage.load_image(
                image=f"../templates/vton_clothes/{clothing_path}.jpg",
            )
        else:
            # 从外部路径或数组加载服装图片
            clothing_image = loadimage.load_image(
                image=clothing_path, 
                abs_path=abs_path, 
                is_ndarray=is_ndarray
            )
        
        model_image = loadimage.load_image(image=model_image_path, abs_path=abs_path, is_ndarray=is_ndarray)
        
        # 调整图像大小
        imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
        resized_clothing = imageresizekj.resize(
            width=1536,
            height=2048,
            upscale_method="lanczos",
            keep_proportion=True,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(clothing_image, 0),
        )
        
        # 图像填充扩展
        imagepadforoutpainttargetsize = NODE_CLASS_MAPPINGS["ImagePadForOutpaintTargetSize"]()
        padded_clothing = imagepadforoutpainttargetsize.expand_image(
            target_width=1536,
            target_height=2048,
            feathering=0,
            upscale_method="lanczos",
            image=get_value_at_index(resized_clothing, 0),
        )
        
        portraitmaskgenerator = NODE_CLASS_MAPPINGS["PortraitMaskGenerator"]()
        maskmorphology = NODE_CLASS_MAPPINGS["MaskMorphology"]()
        invertmask = NODE_CLASS_MAPPINGS["InvertMask"]()
        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        imagemaskblender = NODE_CLASS_MAPPINGS["ImageMaskBlender"]()
        
        blended_clothing = padded_clothing
        
        # 处理模特图像
        resized_model_input = imageresizekj.resize(
            width=1536,
            height=2048,
            upscale_method="lanczos",
            keep_proportion=True,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(model_image, 0),
        )
        
        padded_model = imagepadforoutpainttargetsize.expand_image(
            target_width=1536,
            target_height=2048,
            feathering=0,
            upscale_method="lanczos",
            image=get_value_at_index(resized_model_input, 0),
        )
        
        resized_model = imageresizekj.resize(
            width=768,
            height=1024,
            upscale_method="lanczos",
            keep_proportion=True,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(padded_model, 0),
        )
        
        # 生成人脸蒙版
        facemaskgenerator = NODE_CLASS_MAPPINGS["FaceMaskGenerator"]()
        face_mask = facemaskgenerator.generate_faces_mask(
            max_face=False,
            num_faces=1,
            dilate_pixels=10,
            image=get_value_at_index(resized_model, 0),
            face_models=get_value_at_index(models["face_model"], 0),
        )
        
        # 生成人像蒙版
        portrait_mask_model = portraitmaskgenerator.generate_portrait_mask(
            conf_threshold=0.25,
            iou_threshold=0.5,
            human_targets="person",
            matting_threshold=0.1,
            min_box_area_rate=0.0012,
            image=get_value_at_index(resized_model, 0),
            portrait_models=get_value_at_index(models["portrait_model"], 0),
        )
        
        # 蒙版处理
        morphed_portrait_mask = maskmorphology.process_mask(
            pixels=200,
            use_split_mode=True,
            upper_pixels=150,
            lower_pixels=250,
            feather_split=0,
            mask=get_value_at_index(portrait_mask_model, 0),
        )
        
        # 合成蒙版
        maskcomposite = NODE_CLASS_MAPPINGS["MaskComposite"]()
        combined_mask = maskcomposite.combine(
            x=0,
            y=0,
            operation="and",
            destination=get_value_at_index(face_mask, 0),
            source=get_value_at_index(morphed_portrait_mask, 0),
        )
        
        # 为ICLora添加蒙版
        addmaskforiclora = NODE_CLASS_MAPPINGS["AddMaskForICLora"]()
        masked_images = addmaskforiclora.add_mask(
            patch_mode="patch_right",
            output_length=1536,
            patch_color="#FF0000",
            first_image=get_value_at_index(blended_clothing, 0),
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
        
        # 设置种子
        if seed is None:
            # 当seed为None时使用固定值
            seed = 710874769520552
        else:
            # 当seed不为None时使用随机值
            seed = random.randint(1, 2 ** 64)
            
        # 第一次采样 - 使用应用了补丁的模型
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        sampler_result1 = ksampler.sample(
            seed=seed,
            steps=7,
            cfg=1.5,
            sampler_name="dpmpp_2m",
            scheduler="simple",
            denoise=1,
            model=get_value_at_index(models["catvton_lora_patched"], 0),
            positive=get_value_at_index(inpaint_conditioning, 0),
            negative=get_value_at_index(inpaint_conditioning, 1),
            latent_image=get_value_at_index(inpaint_conditioning, 2),
        )
        
        # VAE解码
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        decoded_image1 = vaedecode.decode(
            samples=get_value_at_index(sampler_result1, 0),
            vae=get_value_at_index(models["vae"], 0),
        )
        
        # 提取右半部分图像
        imagehalfextractor = NODE_CLASS_MAPPINGS["ImageHalfExtractor"]()
        right_half1 = imagehalfextractor.extract_right_half(
            image=get_value_at_index(decoded_image1, 0)
        )
        
        # 第二次处理的人像蒙版
        portrait_mask2 = portraitmaskgenerator.generate_portrait_mask(
            conf_threshold=0.25,
            iou_threshold=0.5,
            human_targets="person",
            matting_threshold=0.1,
            min_box_area_rate=0.0012,
            image=get_value_at_index(right_half1, 0),
            portrait_models=get_value_at_index(models["portrait_model"], 0),
        )
        
        # 合并蒙版
        combined_mask2 = maskcomposite.combine(
            x=0,
            y=0,
            operation="or",
            destination=get_value_at_index(portrait_mask_model, 0),
            source=get_value_at_index(portrait_mask2, 0),
        )
        
        morphed_combined_mask = maskmorphology.process_mask(
            pixels=100,
            use_split_mode=False,
            upper_pixels=50,
            lower_pixels=50,
            feather_split=0,
            mask=get_value_at_index(combined_mask2, 0),
        )
        
        final_mask = maskcomposite.combine(
            x=0,
            y=0,
            operation="multiply",
            destination=get_value_at_index(face_mask, 0),
            source=get_value_at_index(morphed_combined_mask, 0),
        )
        
        # 第二次蒙版处理
        masked_images2 = addmaskforiclora.add_mask(
            patch_mode="patch_right",
            output_length=1536,
            patch_color="#FF0000",
            first_image=get_value_at_index(blended_clothing, 0),
            second_image=get_value_at_index(resized_model, 0),
            second_mask=get_value_at_index(final_mask, 0),
        )
        
        # 第二次条件设置
        inpaint_conditioning2 = inpaintmodelconditioning.encode(
            noise_mask=True,
            positive=get_value_at_index(fluxguidance_result, 0),
            negative=get_value_at_index(negative_prompt, 0),
            vae=get_value_at_index(models["vae"], 0),
            pixels=get_value_at_index(masked_images2, 0),
            mask=get_value_at_index(masked_images2, 1),
        )
        
        # 第二次采样 - 使用应用了补丁的模型
        sampler_result2 = ksampler.sample(
            seed=seed,
            steps=17,
            cfg=3.5,
            sampler_name="dpmpp_2m",
            scheduler="simple",
            denoise=1,
            model=get_value_at_index(models["catvton_lora_patched"], 0),
            positive=get_value_at_index(inpaint_conditioning2, 0),
            negative=get_value_at_index(inpaint_conditioning2, 1),
            latent_image=get_value_at_index(inpaint_conditioning2, 2),
        )
        
        # 第二次解码
        decoded_image2 = vaedecode.decode(
            samples=get_value_at_index(sampler_result2, 0),
            vae=get_value_at_index(models["vae"], 0),
        )
        
        # 提取右半部分
        right_half2 = imagehalfextractor.extract_right_half(
            image=get_value_at_index(decoded_image2, 0)
        )
        
        # # 使用Florence2模型生成详细描述
        # layerutility_florence2image2prompt = NODE_CLASS_MAPPINGS["LayerUtility: Florence2Image2Prompt"]()
        # florence2_caption = layerutility_florence2image2prompt.florence2_image2prompt(
        #     task="more detailed caption",
        #     text_input="",
        #     max_new_tokens=1024,
        #     num_beams=3,
        #     do_sample=False,
        #     fill_mask=False,
        #     florence2_model=get_value_at_index(models["florence2_model"], 0),
        #     image=get_value_at_index(right_half2, 0),
        # )
        
        # 准备高质量提示
        textinput_ = NODE_CLASS_MAPPINGS["TextInput_"]()
        quality_prompt = textinput_.run(
            text="high quality, detailed, photograph , hd, 8k , 4k , sharp, highly detailed"
        )
        
        # # 合并提示
        # text_concatenate_jps = NODE_CLASS_MAPPINGS["Text Concatenate (JPS)"]()
        # combined_prompt = text_concatenate_jps.get_contxt(
        #     delimiter="comma",
        #     text1=get_value_at_index(quality_prompt, 0),
        #     text2=get_value_at_index(florence2_caption, 0),
        # )
        
        # Flux文本编码
        cliptextencodeflux = NODE_CLASS_MAPPINGS["CLIPTextEncodeFlux"]()
        positive_flux = cliptextencodeflux.encode(
            clip_l=get_value_at_index(quality_prompt, 0),
            t5xxl=get_value_at_index(quality_prompt, 0),
            guidance=3.5,
            clip=get_value_at_index(models["clip"], 0),
        )
        
        negative_flux = cliptextencodeflux.encode(
            clip_l="ng_deepnegative_v1_75t,(badhandv4:1.2),EasyNegative,(worst quality:2),",
            t5xxl="ng_deepnegative_v1_75t,(badhandv4:1.2),EasyNegative,(worst quality:2),",
            guidance=3.5,
            clip=get_value_at_index(models["clip"], 0),
        )
        
        # 转换为NumPy
        converttensortonumpy = NODE_CLASS_MAPPINGS["ConvertTensorToNumpy"]()
        numpy_image = converttensortonumpy.convert(
            image=get_value_at_index(right_half2, 0)
        )
        
        # 应用SR增强
        srblendprocess = NODE_CLASS_MAPPINGS["SrBlendProcess"]()
        enhanced_image = srblendprocess.enhance_process(
            is_front="False",
            model=get_value_at_index(models["srblend"], 0),
            src_img=get_value_at_index(numpy_image, 0),
        )
        
        # 准备最终尺寸
        resized_model_final = imageresizekj.resize(
            width=1536,
            height=2048,
            upscale_method="lanczos",
            keep_proportion=True,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(padded_model, 0),
        )
        
        # 调整增强图像尺寸
        resized_enhanced = imageresizekj.resize(
            width=512,
            height=512,
            upscale_method="lanczos",
            keep_proportion=True,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(enhanced_image, 1),
            get_image_size=get_value_at_index(resized_model_final, 0),
        )
        
        # 准备蒙版
        mask_image_final = masktoimage.mask_to_image(
            mask=get_value_at_index(combined_mask, 0)
        )
        
        resized_mask = imageresizekj.resize(
            width=512,
            height=512,
            upscale_method="lanczos",
            keep_proportion=True,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(mask_image_final, 0),
            get_image_size=get_value_at_index(resized_model_final, 0),
        )
        
        # 图像转蒙版
        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        final_process_mask = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(resized_mask, 0)
        )
        
        # 最终增强处理的条件设置
        inpaint_conditioning_final = inpaintmodelconditioning.encode(
            noise_mask=True,
            positive=get_value_at_index(positive_flux, 0),
            negative=get_value_at_index(negative_flux, 0),
            vae=get_value_at_index(models["vae"], 0),
            pixels=get_value_at_index(resized_enhanced, 0),
            mask=get_value_at_index(final_process_mask, 0),
        )
        
        # 最终增强采样 - 使用应用了补丁的增强模型
        enhanced_sampler_result = ksampler.sample(
            seed=seed,
            steps=9,
            cfg=1,
            sampler_name="dpmpp_2m",
            scheduler="sgm_uniform",
            denoise=0.35,
            model=get_value_at_index(models["enhance_lora_patched"], 0),
            positive=get_value_at_index(inpaint_conditioning_final, 0),
            negative=get_value_at_index(inpaint_conditioning_final, 1),
            latent_image=get_value_at_index(inpaint_conditioning_final, 2),
        )
        
        # 解码增强结果
        enhanced_decoded = vaedecode.decode(
            samples=get_value_at_index(enhanced_sampler_result, 0),
            vae=get_value_at_index(models["vae"], 0),
        )
        
        # 生成最终的人脸蒙版
        final_face_mask = facemaskgenerator.generate_faces_mask(
            max_face=False,
            num_faces=1,
            dilate_pixels=5,
            image=get_value_at_index(resized_model_final, 0),
            face_models=get_value_at_index(models["face_model"], 0),
        )
        
        # 最终混合
        final_result = imagemaskblender.blend_images(
            feather=30,
            image_fg=get_value_at_index(enhanced_decoded, 0),
            image_bg=get_value_at_index(resized_model_final, 0),
            mask=get_value_at_index(final_face_mask, 0),
        )
        
        # 获取numpy格式的最终结果
        final_img = get_value_at_index(final_result, 0) * 255.0
        return final_img.cpu().numpy().astype(np.uint8)[0]


# 命令行入口函数
def main():
    # 参数解析
    parser = argparse.ArgumentParser(
        description="增强版虚拟换装工具。以下是必需的输入参数。"
    )
    parser.add_argument(
        "--clothing", required=True, help="服装图片名称（含.jpg后缀，如'Tmiaofu.jpg'）"
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
    
    # 处理服装名称，移除.jpg后缀
    clothing_name = args.clothing
    if clothing_name.endswith('.jpg'):
        clothing_name = clothing_name[:-4]
    
    # 初始化模型
    print("正在加载模型...")
    models = init_vton_models()
    
    # 处理图像
    print("开始处理图像...")
    result = cloth_vton_process(
        models,
        clothing_name,
        args.model,
        seed=args.seed,
        abs_path=True
    )
    
    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"结果已保存至: {args.output}")

if __name__ == "__main__":
    main()

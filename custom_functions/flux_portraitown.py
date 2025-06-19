import os
import random
import sys
import json
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np
import importlib.util


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """返回序列或映射中给定索引处的值。

    如果对象是序列（如列表或字符串），则返回给定索引处的值。
    如果对象是映射（如字典），则返回索引处键的值。

    有些返回字典，在这些情况下，我们查找"results"键

    Args:
        obj (Union[Sequence, Mapping]): 要从中检索值的对象。
        index (int): 要检索的值的索引。

    Returns:
        Any: 给定索引处的值。

    Raises:
        IndexError: 如果索引超出对象边界且对象不是映射。
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    从给定路径开始递归查看父文件夹，直到找到给定名称。
    如果找到，则返回路径，否则返回None。
    """
    # 如果没有给定路径，使用当前工作目录
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))

    # 检查当前目录是否包含该名称
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # 获取父目录
    parent_directory = os.path.dirname(path)

    # 如果父目录与当前目录相同，则我们已达到根目录并停止搜索
    if parent_directory == path:
        return None

    # 递归调用函数和父目录
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


def import_custom_nodes() -> None:
    """找到custom_nodes文件夹中的所有自定义节点，并将这些节点对象添加到NODE_CLASS_MAPPINGS

    此函数设置新的asyncio事件循环，初始化PromptServer，
    创建PromptQueue，并初始化自定义节点。
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # 创建新的事件循环并将其设置为默认循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 使用循环创建PromptServer实例
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # 初始化自定义节点
    init_extra_nodes(init_custom_nodes=True)


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def init_flux_portraitown_models():
    """初始化所有需要的模型并返回它们"""
    import_custom_nodes()
    from nodes import NODE_CLASS_MAPPINGS
    
    comfyui_path = find_path("ComfyUI")
    model_root = os.path.join(comfyui_path, "models") if comfyui_path else "models"
    human_mask_path = os.path.join(model_root, "human_mask")
    
    with torch.inference_mode():
        facewarppipebuilder = NODE_CLASS_MAPPINGS["FaceWarpPipeBuilder"]()
        facewarppipebuilder_31 = facewarppipebuilder.load_models(
            detect_model_path="facedetect/scrfd_10g_bnkps_shape640x640.onnx",
            deca_dir="deca",
            gpu_choose="cuda:1",
        )

        faceswappipebuilder = NODE_CLASS_MAPPINGS["FaceSwapPipeBuilder"]()
        faceswappipebuilder_51 = faceswappipebuilder.load_models(
            swap_own_model="faceswap/swapper_own.pth",
            arcface_model="faceswap/arcface_checkpoint.tar",
            facealign_config_dir="face_align",
            phase1_model="facealign/p1.pt",
            phase2_model="facealign/p2.pt",
            device="cuda:1",
        )

        srblendbuildpipe = NODE_CLASS_MAPPINGS["SrBlendBuildPipe"]()
        srblendbuildpipe_58 = srblendbuildpipe.load_models(
            lut_path="lut", gpu_choose="cuda:1", sr_type="RealESRGAN_X2", half=True
        )

        gpenpbuildpipeline = NODE_CLASS_MAPPINGS["GPENPBuildpipeline"]()
        gpenpbuildpipeline_164 = gpenpbuildpipeline.load_model(
            model="GPEN-BFR-1024.pth",
            in_size=1024,
            channel_multiplier=2,
            narrow=1,
            alpha=0.7000000000000001,
            device="cuda:1",
        )

        preprocnewbuildpipe = NODE_CLASS_MAPPINGS["PreprocNewBuildPipe"]()
        preprocnewbuildpipe_67 = preprocnewbuildpipe.load_models(
            wd14_cgpath="wd14_tagger"
        )
        
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_155_1 = dualcliploader.load_clip(
            clip_name1="ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors",
            clip_name2="t5xxl_fp8_e4m3fn.safetensors",
            type="flux",
            device="default",
        )
        
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_99 = vaeloader.load_vae(vae_name="ae.safetensors")
        
        # 设置CLIP和VAE设备
        overrideclipdevice = NODE_CLASS_MAPPINGS["OverrideCLIPDevice"]()
        overrideclipdevice_155_3 = overrideclipdevice.patch(
            device="cuda:1", clip=get_value_at_index(dualcliploader_155_1, 0)
        )

        overridevaedevice = NODE_CLASS_MAPPINGS["OverrideVAEDevice"]()
        overridevaedevice_106 = overridevaedevice.patch(
            device="cuda:1", vae=get_value_at_index(vaeloader_99, 0)
        )
        
        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        controlnetloader_155_0 = controlnetloader.load_controlnet(
            control_net_name="flux-canny-controlnet-v3.safetensors"
        )
        
        pulidfluxmodelloader = NODE_CLASS_MAPPINGS["PulidFluxModelLoader"]()
        pulidfluxmodelloader_82_0 = pulidfluxmodelloader.load_model(
            pulid_file="pulid_flux_v0.9.1.safetensors"
        )
        
        pulidfluxinsightfaceloader = NODE_CLASS_MAPPINGS["PulidFluxInsightFaceLoader"]()
        pulidfluxinsightfaceloader_82_1 = pulidfluxinsightfaceloader.load_insightface(
            provider="CUDA"
        )
        
        pulidfluxevacliploader = NODE_CLASS_MAPPINGS["PulidFluxEvaClipLoader"]()
        pulidfluxevacliploader_82_2 = pulidfluxevacliploader.load_eva_clip()
        
        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_82_3 = unetloader.load_unet(
            unet_name="flux1-dev-fp8.safetensors", weight_dtype="fp8_e4m3fn"
        )
        
        portraitmodelloader = NODE_CLASS_MAPPINGS["PortraitModelLoader"]()
        portraitmodelloader_168 = portraitmodelloader.load_portrait_models(
            model_root=human_mask_path, device="cuda:1 (NVIDIA GeForce RTX 3090)"
        )

        facemodelloader = NODE_CLASS_MAPPINGS["FaceModelLoader"]()
        facemodelloader_169 = facemodelloader.load_face_models(
            model_root=model_root, device="cuda:1 (NVIDIA GeForce RTX 3090)"
        )
        
        return {
            "facewarppipebuilder_31": facewarppipebuilder_31,
            "faceswappipebuilder_51": faceswappipebuilder_51,
            "srblendbuildpipe_58": srblendbuildpipe_58,
            "gpenpbuildpipeline_164": gpenpbuildpipeline_164,
            "preprocnewbuildpipe_67": preprocnewbuildpipe_67,
            "overrideclipdevice_155_3": overrideclipdevice_155_3,
            "overridevaedevice_106": overridevaedevice_106,
            "controlnetloader_155_0": controlnetloader_155_0,
            "pulidfluxmodelloader_82_0": pulidfluxmodelloader_82_0,
            "pulidfluxinsightfaceloader_82_1": pulidfluxinsightfaceloader_82_1,
            "pulidfluxevacliploader_82_2": pulidfluxevacliploader_82_2,
            "unetloader_82_3": unetloader_82_3,
            "portraitmodelloader_168": portraitmodelloader_168,
            "facemodelloader_169": facemodelloader_169,
        }


def flux_portraitown_process(flux_inited_models, image_path, template_id="IDphotos/IDphotos_female_5", negative_prompt=None, abs_path=True, is_ndarray=False):
    """
    使用初始化好的模型处理图像
    
    Args:
        flux_inited_models: 初始化好的模型字典
        image_path: 图像路径或numpy数组
        template_id: 模板ID
        negative_prompt: 负面提示词，默认为None
        abs_path: 是否是绝对路径
        is_ndarray: 图像是否是numpy数组
        
    Returns:
        处理后的图像（numpy数组）
    """
    from nodes import NODE_CLASS_MAPPINGS, LoadImage
    
    with torch.inference_mode():
        loadimage = LoadImage()
        loadimage_24 = loadimage.load_image(image=image_path, abs_path=abs_path, is_ndarray=is_ndarray)

        # 准备负面提示词
        if negative_prompt is None:
            textinput_ = NODE_CLASS_MAPPINGS["TextInput_"]()
            textinput___97 = textinput_.run(
                text="(smile:2),(laugh:2),(teeth:2),(grin:2),ng_deepnegative_v1_75t,(badhandv4:1.2),EasyNegative,(worst quality:2),"
            )
            negative_prompt_text = get_value_at_index(textinput___97, 0)
        else:
            negative_prompt_text = negative_prompt

        facewarpdetectfacesimginput = NODE_CLASS_MAPPINGS["FaceWarpDetectFacesImgInput"]()
        facewarpdetectfacesimginput_77 = facewarpdetectfacesimginput.detect_faces(
            is_bgr="True",
            model=get_value_at_index(flux_inited_models["facewarppipebuilder_31"], 0),
            image=get_value_at_index(loadimage_24, 0),
        )

        preprocnewgetconds = NODE_CLASS_MAPPINGS["PreprocNewGetConds"]()
        preprocnewgetconds_68 = preprocnewgetconds.prepare(
            template_id=template_id,
            is_front="False",
            is_bgr="True",
            model=get_value_at_index(flux_inited_models["preprocnewbuildpipe_67"], 0),
            src_img=get_value_at_index(loadimage_24, 0),
            faces=get_value_at_index(facewarpdetectfacesimginput_77, 0),
        )

        preprocnewsplitconds = NODE_CLASS_MAPPINGS["PreprocNewSplitConds"]()
        preprocnewsplitconds_69 = preprocnewsplitconds.split(
            pipe_conditions=get_value_at_index(preprocnewgetconds_68, 0)
        )

        gpenprocess = NODE_CLASS_MAPPINGS["GPENProcess"]()
        gpenprocess_79 = gpenprocess.enhance_face(
            aligned=False,
            model=get_value_at_index(flux_inited_models["gpenpbuildpipeline_164"], 0),
            image=get_value_at_index(preprocnewsplitconds_69, 0),
        )

        facewarpdetectfacesmethod = NODE_CLASS_MAPPINGS["FaceWarpDetectFacesMethod"]()
        facewarpdetectfacesmethod_33 = facewarpdetectfacesmethod.detect_faces(
            model=get_value_at_index(flux_inited_models["facewarppipebuilder_31"], 0),
            image=get_value_at_index(gpenprocess_79, 0),
        )

        facewarpgetfaces3dinfomethod = NODE_CLASS_MAPPINGS["FaceWarpGetFaces3DinfoMethod"]()
        facewarpgetfaces3dinfomethod_32 = facewarpgetfaces3dinfomethod.get_faces_3dinfo(
            model=get_value_at_index(flux_inited_models["facewarppipebuilder_31"], 0),
            image=get_value_at_index(gpenprocess_79, 0),
            faces=get_value_at_index(facewarpdetectfacesmethod_33, 0),
        )

        facewarpwarp3dfaceimgmaskmethod = NODE_CLASS_MAPPINGS["FaceWarpWarp3DfaceImgMaskMethod"]()
        facewarpwarp3dfaceimgmaskmethod_36 = facewarpwarp3dfaceimgmaskmethod.warp_3d_face(
            model=get_value_at_index(flux_inited_models["facewarppipebuilder_31"], 0),
            user_dict=get_value_at_index(facewarpgetfaces3dinfomethod_32, 0),
            template_image=get_value_at_index(preprocnewsplitconds_69, 3),
            template_mask_img=get_value_at_index(preprocnewsplitconds_69, 11),
            template_canny_img=get_value_at_index(preprocnewsplitconds_69, 4),
        )

        # 准备控制网络
        # setunioncontrolnettype = NODE_CLASS_MAPPINGS["SetUnionControlNetType"]()
        # setunioncontrolnettype_155_2 = setunioncontrolnettype.set_controlnet_type(
        #     type="canny/lineart/anime_lineart/mlsd",
        #     control_net=get_value_at_index(flux_inited_models["controlnetloader_155_0"], 0),
        # )
        
        # 处理正面和负面提示词
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_155_6 = cliptextencode.encode(
            text=get_value_at_index(preprocnewsplitconds_69, 1),
            clip=get_value_at_index(flux_inited_models["overrideclipdevice_155_3"], 0),
        )

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        fluxguidance_155_7 = fluxguidance.append(
            guidance=4.5, conditioning=get_value_at_index(cliptextencode_155_6, 0)
        )

        cliptextencode_155_4 = cliptextencode.encode(
            text=negative_prompt_text,
            clip=get_value_at_index(flux_inited_models["overrideclipdevice_155_3"], 0),
        )

        fluxguidance_155_5 = fluxguidance.append(
            guidance=3.5, conditioning=get_value_at_index(cliptextencode_155_4, 0)
        )

        # 转换图像为tensor
        convertnumpytotensor = NODE_CLASS_MAPPINGS["ConvertNumpyToTensor"]()
        convertnumpytotensor_103 = convertnumpytotensor.convert(
            image=get_value_at_index(facewarpwarp3dfaceimgmaskmethod_36, 2)
        )

        convertnumpytotensor_107 = convertnumpytotensor.convert(
            image=get_value_at_index(facewarpwarp3dfaceimgmaskmethod_36, 0)
        )

        convertnumpytotensor_102 = convertnumpytotensor.convert(
            image=get_value_at_index(facewarpwarp3dfaceimgmaskmethod_36, 1)
        )

        convertnumpytotensor_100 = convertnumpytotensor.convert(
            image=get_value_at_index(gpenprocess_79, 0)
        )

        # 应用控制网络
        controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
        controlnetapplyadvanced_155_8 = controlnetapplyadvanced.apply_controlnet(
            strength=0.7000000000000002,
            start_percent=0,
            end_percent=0.8000000000000002,
            positive=get_value_at_index(fluxguidance_155_7, 0),
            negative=get_value_at_index(fluxguidance_155_5, 0),
            control_net=get_value_at_index(flux_inited_models["controlnetloader_155_0"], 0),
            image=get_value_at_index(convertnumpytotensor_103, 0),
            vae=get_value_at_index(flux_inited_models["overridevaedevice_106"], 0),
        )

        # 创建蒙版
        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        imagetomask_105 = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(convertnumpytotensor_102, 0)
        )

        # 准备修复模型条件
        inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
        inpaintmodelconditioning_104 = inpaintmodelconditioning.encode(
            noise_mask=True,
            positive=get_value_at_index(controlnetapplyadvanced_155_8, 0),
            negative=get_value_at_index(controlnetapplyadvanced_155_8, 1),
            vae=get_value_at_index(flux_inited_models["overridevaedevice_106"], 0),
            pixels=get_value_at_index(convertnumpytotensor_107, 0),
            mask=get_value_at_index(imagetomask_105, 0),
        )

        # 应用PulidFlux
        applypulidflux = NODE_CLASS_MAPPINGS["ApplyPulidFlux"]()
        applypulidflux_82_4 = applypulidflux.apply_pulid_flux(
            weight=0.7500000000000002,
            start_at=0,
            end_at=1,
            model=get_value_at_index(flux_inited_models["unetloader_82_3"], 0),
            pulid_flux=get_value_at_index(flux_inited_models["pulidfluxmodelloader_82_0"], 0),
            eva_clip=get_value_at_index(flux_inited_models["pulidfluxevacliploader_82_2"], 0),
            face_analysis=get_value_at_index(flux_inited_models["pulidfluxinsightfaceloader_82_1"], 0),
            image=get_value_at_index(convertnumpytotensor_100, 0),
            unique_id=random.randint(1, 2**64),
        )

        # 应用其他模型补丁
        fluxforwardoverrider = NODE_CLASS_MAPPINGS["FluxForwardOverrider"]()
        fluxforwardoverrider_82_5 = fluxforwardoverrider.apply_patch(
            model=get_value_at_index(applypulidflux_82_4, 0)
        )

        applyteacachepatch = NODE_CLASS_MAPPINGS["ApplyTeaCachePatch"]()
        applyteacachepatch_82_6 = applyteacachepatch.apply_patch(
            rel_l1_thresh=0.4,
            cache_device="offload_device",
            wan_coefficients="disabled",
            model=get_value_at_index(fluxforwardoverrider_82_5, 0),
        )

        # 采样器
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        ksampler_123 = ksampler.sample(
            seed=710874769520552,
            steps=20,
            cfg=1.0,
            sampler_name="dpmpp_2m",
            scheduler="sgm_uniform",
            denoise=0.8600000000000002,
            model=get_value_at_index(applyteacachepatch_82_6, 0),
            positive=get_value_at_index(inpaintmodelconditioning_104, 0),
            negative=get_value_at_index(inpaintmodelconditioning_104, 1),
            latent_image=get_value_at_index(inpaintmodelconditioning_104, 2),
        )

        # 解码VAE
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        vaedecode_114 = vaedecode.decode(
            samples=get_value_at_index(ksampler_123, 0),
            vae=get_value_at_index(flux_inited_models["overridevaedevice_106"], 0),
        )

        # 转换回numpy
        converttensortonumpy = NODE_CLASS_MAPPINGS["ConvertTensorToNumpy"]()
        converttensortonumpy_109 = converttensortonumpy.convert(
            image=get_value_at_index(vaedecode_114, 0)
        )

        # 第二阶段：面部对齐和交换
        facewarpwarp3dfaceimgmethod = NODE_CLASS_MAPPINGS["FaceWarpWarp3DfaceImgMethod"]()
        facewarpwarp3dfaceimgmethod_70 = facewarpwarp3dfaceimgmethod.warp_3d_face(
            model=get_value_at_index(flux_inited_models["facewarppipebuilder_31"], 0),
            user_dict=get_value_at_index(facewarpgetfaces3dinfomethod_32, 0),
            template_image=get_value_at_index(converttensortonumpy_109, 0),
        )

        facewarpdetectfacesmethod_55 = facewarpdetectfacesmethod.detect_faces(
            model=get_value_at_index(flux_inited_models["facewarppipebuilder_31"], 0),
            image=get_value_at_index(facewarpwarp3dfaceimgmethod_70, 0),
        )

        faceswapdetectpts = NODE_CLASS_MAPPINGS["FaceSwapDetectPts"]()
        faceswapdetectpts_52 = faceswapdetectpts.detect_face_pts(
            ptstype="256",
            model=get_value_at_index(flux_inited_models["faceswappipebuilder_51"], 0),
            src_image=get_value_at_index(facewarpwarp3dfaceimgmethod_70, 0),
            src_faces=get_value_at_index(facewarpdetectfacesmethod_55, 0),
        )

        faceswapdetectpts_56 = faceswapdetectpts.detect_face_pts(
            ptstype="5",
            model=get_value_at_index(flux_inited_models["faceswappipebuilder_51"], 0),
            src_image=get_value_at_index(gpenprocess_79, 0),
            src_faces=get_value_at_index(facewarpdetectfacesmethod_33, 0),
        )

        faceswapdetectpts_54 = faceswapdetectpts.detect_face_pts(
            ptstype="5",
            model=get_value_at_index(flux_inited_models["faceswappipebuilder_51"], 0),
            src_image=get_value_at_index(facewarpwarp3dfaceimgmethod_70, 0),
            src_faces=get_value_at_index(facewarpdetectfacesmethod_55, 0),
        )

        faceswapmethod = NODE_CLASS_MAPPINGS["FaceSwapMethod"]()
        faceswapmethod_53 = faceswapmethod.swap_face(
            model=get_value_at_index(flux_inited_models["faceswappipebuilder_51"], 0),
            src_image=get_value_at_index(gpenprocess_79, 0),
            two_stage_image=get_value_at_index(facewarpwarp3dfaceimgmethod_70, 0),
            source_5pts=get_value_at_index(faceswapdetectpts_56, 0),
            target_5pts=get_value_at_index(faceswapdetectpts_54, 0),
            target_256pts=get_value_at_index(faceswapdetectpts_52, 0),
        )

        gpenprocess_60 = gpenprocess.enhance_face(
            aligned=False,
            model=get_value_at_index(flux_inited_models["gpenpbuildpipeline_164"], 0),
            image=get_value_at_index(faceswapmethod_53, 0),
        )

        # 最终图像处理
        srblendprocess = NODE_CLASS_MAPPINGS["SrBlendProcess"]()
        srblendprocess_71 = srblendprocess.enhance_process(
            is_front="False",
            model=get_value_at_index(flux_inited_models["srblendbuildpipe_58"], 0),
            src_img=get_value_at_index(gpenprocess_60, 0),
        )

        # 使用人像蒙版生成器处理图像
        if template_id.startswith("IDphotos/"):
            portraitmaskgenerator = NODE_CLASS_MAPPINGS["PortraitMaskGenerator"]()
            portraitmaskgenerator_169 = portraitmaskgenerator.generate_portrait_mask(
                conf_threshold=0.25,
                iou_threshold=0.5,
                human_targets="person",
                matting_threshold=0.1,
                min_box_area_rate=0.0012,
                image=get_value_at_index(srblendprocess_71, 1),
                portrait_models=get_value_at_index(flux_inited_models["portraitmodelloader_168"], 0),
            )

            facemaskgenerator = NODE_CLASS_MAPPINGS["FaceMaskGenerator"]()
            facemaskgenerator_169 = facemaskgenerator.generate_faces_mask(
                max_face=False,
                num_faces=1,
                dilate_pixels=5,
                image=get_value_at_index(srblendprocess_71, 1),
                face_models=get_value_at_index(flux_inited_models["facemodelloader_169"], 0),
            )

            invertmask = NODE_CLASS_MAPPINGS["InvertMask"]()
            invertmask_170 = invertmask.invert(
                mask=get_value_at_index(facemaskgenerator_169, 0),
            )

            maskmathoperations = NODE_CLASS_MAPPINGS["MaskMathOperations"]()
            maskmathoperations_171 = maskmathoperations.perform_mask_operation(
                operation="max",
                clamp_result=True,
                mask_a=get_value_at_index(invertmask_170, 0),
                mask_b=get_value_at_index(portraitmaskgenerator_169, 0),
            )

            imagealphamaskreplacer = NODE_CLASS_MAPPINGS["ImageAlphaMaskReplacer"]()
            imagealphamaskreplacer_171 = imagealphamaskreplacer.replace_alpha_with_mask(
                image=get_value_at_index(srblendprocess_71, 1),
                mask=get_value_at_index(maskmathoperations_171, 0),
            )
            
            final_img = get_value_at_index(imagealphamaskreplacer_171, 0)
        else:
            final_img = get_value_at_index(srblendprocess_71, 1)
        
        # 将处理后的图像转换为numpy数组并返回
        outimg = get_value_at_index(final_img, 0)
        if isinstance(outimg, torch.Tensor):
            outimg = outimg.squeeze(0) * 255.
            return outimg.cpu().numpy().astype(np.uint8)
        else:
            return outimg


if __name__ == "__main__":
    import argparse
    import cv2
    from PIL import Image
    
    parser = argparse.ArgumentParser(description="Flux Portrait Own 处理工具")
    parser.add_argument("--image", "-i", type=str, default='/data/projs/WebServer/distributed-server-node/submodules/VisualForge/ComfyUI/input/example.jpg', help="输入图像路径")
    parser.add_argument("--output", "-o", type=str, default="output.png", help="输出图像路径")
    parser.add_argument("--template", "-t", type=str, default="IDphotos/IDphotos_female_5", help="模板ID")
    parser.add_argument("--negative-prompt", "-n", type=str, default=None, help="负面提示词")
    
    args = parser.parse_args()
    
    # 初始化模型
    print("正在初始化模型...")
    flux_inited_models = init_flux_portraitown_models()
    
    # 加载图像
    print(f"正在处理图像: {args.image}")
    input_img = np.array(Image.open(args.image))[..., :3]
    
    # 处理图像
    for i in range(10):
        final_img = flux_portraitown_process(
            flux_inited_models, 
            input_img, 
            template_id=args.template,
            is_ndarray=True
        )
    
    # 保存结果
    print(f"正在保存结果到: {args.output}")
    if final_img.shape[2] == 4:  # 有Alpha通道
        cv2.imwrite(args.output, final_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        cv2.imwrite(args.output, final_img)
    
    print("处理完成!") 
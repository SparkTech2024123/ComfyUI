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

class ClothType(Enum):
    """服装类型枚举类，用于定义不同的虚拟试衣选项"""
    HANFU = "hanfu"
    JK_UNIFORM = "jk_uniform"
    
    @classmethod
    def get_text_prompt(cls, cloth_type):
        """获取指定服装类型的文本提示词
        
        Args:
            cloth_type: 服装类型
            
        Returns:
            str: 该服装类型的文本提示词
        """
        prompts = {
            cls.HANFU: "1girl, hanfu, (wide sleeves:1.2), floral embroidery, long skirt, silk texture, flowing fabric, traditional Chinese beauty, soft lighting, dynamic pose, (delicate features:1.1), wind-blown hair, historical style, pastel colors, cinematic atmosphere.",
            cls.JK_UNIFORM: "1girl, JK uniform, white short-sleeve shirt with brown trim, brown plaid tie, brown plaid pleated skirt, white stripe details, school ID badge element, youthful style, modern fashion."
        }
        return prompts.get(cloth_type, prompts[cls.HANFU])
    
    @classmethod
    def get_region_mappings(cls, cloth_type):
        """获取指定服装类型的区域映射配置
        
        Args:
            cloth_type: 服装类型
            
        Returns:
            List[Tuple[str, str]]: 包含(region_text, target_text)对的列表
        """
        mappings = {
            cls.HANFU: [
                ("hanfu", "hanfu"),
                ("wide sleeves", "wide sleeves"),
                ("floral embroidery", "floral embroidery"),
                ("long skirt", "long skirt"),
                ("silk texture", "silk texture"),
                ("flowing fabric", "flowing fabric")
            ],
            cls.JK_UNIFORM: [
                ("white short-sleeve shirt", "white"),
                ("brown trim", "brown"),
                ("brown plaid tie", "brown"),
                ("brown plaid pleated skirt", "brown"),
                ("white stripe details", "white")
            ]
        }
        return mappings.get(cloth_type, mappings[cls.HANFU])
    
    @classmethod
    def get_negative_prompt(cls, cloth_type=None):
        """获取通用负面提示词
        
        Returns:
            str: 负面提示词
        """
        return "(worst quality:1.6), (low quality:1.5), blurry, deformed hands, mutated fingers, bad anatomy, extra limbs, (disfigured face:1.4), asymmetrical eyes, fused limbs, watermark, text, logo, 3D render, mutation, cloned features, floating objects, double image, low-poly, wax skin, anime style, nsfw, (overexposed:1.3), chromatic aberration"


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the given index in the sequence of keys.
    
    Args:
        obj: The object to retrieve the value from.
        index: The index of the value to retrieve.
        
    Returns:
        The value at the given index.
        
    Raises:
        IndexError: If the index is out of bounds for the object.
        TypeError: If the object is not a sequence or mapping.
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
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """Add extra model paths to the sys.path.
    
    The function reads the 'extra_model_paths.yaml' file and adds the paths to the sys.path.
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

def init_cloth_vton_models():
    """初始化并加载虚拟试衣所需的所有模型"""
    try:
        import_custom_nodes()
        with torch.inference_mode():
            # from nodes import NODE_CLASS_MAPPINGS
            
            # 加载基础模型
            checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            checkpointloadersimple_47 = checkpointloadersimple.load_checkpoint(
                ckpt_name="realisticVisionV60B1_v51HyperVAE.safetensors"
            )

            # 加载CLIP模型
            cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            
            # 加载PowerPaint和BrushNet模型
            powerpaintcliploader = NODE_CLASS_MAPPINGS["PowerPaintCLIPLoader"]()
            # 确保在使用前初始化属性
            powerpaintcliploader.INPUT_TYPES()
            powerpaintcliploader_66 = powerpaintcliploader.ppclip_loading(
                base="sd15_clip.safetensors", powerpaint="powerpaint/pytorch_model.bin"
            )

            brushnetloader = NODE_CLASS_MAPPINGS["BrushNetLoader"]()
            brushnetloader.INPUT_TYPES()
            brushnetloader_90 = brushnetloader.brushnet_loading(
                brushnet="powerpaint/diffusion_pytorch_model.safetensors", dtype="float16"
            )

            # 加载ControlNet模型
            controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
            controlnetloader_97 = controlnetloader.load_controlnet(
                control_net_name="control_v11f1p_sd15_depth.safetensors"
            )

            controlnetloader_120 = controlnetloader.load_controlnet(
                control_net_name="control_v11p_sd15_openpose.safetensors"
            )

            # 加载SR模型
            srblendbuildpipe = NODE_CLASS_MAPPINGS["SrBlendBuildPipe"]()
            srblendbuildpipe_126 = srblendbuildpipe.load_models(
                lut_path="lut", gpu_choose="cuda:0", sr_type="RealESRGAN", half=True
            )

            # 加载Florence2模型
            layermask_loadflorence2model = NODE_CLASS_MAPPINGS[
                "LayerMask: LoadFlorence2Model"
            ]()
            layermask_loadflorence2model_131 = layermask_loadflorence2model.load(
                version="CogFlorence-2.1-Large"
            )

            # 加载人像和面部模型
            portraitmodelloader = NODE_CLASS_MAPPINGS["PortraitModelLoader"]()
            portraitmodelloader_134 = portraitmodelloader.load_portrait_models(
                model_root="models/human_mask"
            )

            facemodelloader = NODE_CLASS_MAPPINGS["FaceModelLoader"]()
            facemodelloader_136 = facemodelloader.load_face_models(model_root="models")

            # 返回所有加载的模型
            return {
                "checkpointloadersimple_47": checkpointloadersimple_47,
                "powerpaintcliploader_66": powerpaintcliploader_66,
                "brushnetloader_90": brushnetloader_90,
                "controlnetloader_97": controlnetloader_97,
                "controlnetloader_120": controlnetloader_120,
                "srblendbuildpipe_126": srblendbuildpipe_126,
                "layermask_loadflorence2model_131": layermask_loadflorence2model_131,
                "portraitmodelloader_134": portraitmodelloader_134,
                "facemodelloader_136": facemodelloader_136,
                "cliptextencode": cliptextencode,
            }
    except Exception as e:
        print(f"初始化模型时出错: {str(e)}")
        raise

def cloth_vton_process(inited_models, image_path, cloth_type=ClothType.HANFU, text_prompt=None, \
    clip=None, negative_prompt=None, abs_path=True, is_ndarray=False, queue_size=1):
    """
    使用已初始化的模型处理图像进行虚拟试衣
    
    Args:
        inited_models: 初始化好的模型字典
        image_path: 输入图像路径或图像数组
        cloth_type: 服装类型，默认为HANFU
        text_prompt: 文本提示，描述要生成的服装，如果为None则使用cloth_type的预设提示
        clip: CLIP模型，如未提供则使用检查点中的CLIP
        negative_prompt: 负面提示，如果为None则使用默认负面提示
        abs_path: 是否为绝对路径
        is_ndarray: 输入是否为numpy数组
        queue_size: 处理队列大小
        
    Returns:
        numpy.ndarray: 处理后的图像数组
    """
    try:
        with torch.inference_mode(), ctx:
            # from nodes import NODE_CLASS_MAPPINGS
            
            # 解包预加载的模型
            checkpointloadersimple_47 = inited_models["checkpointloadersimple_47"]
            powerpaintcliploader_66 = inited_models["powerpaintcliploader_66"]
            brushnetloader_90 = inited_models["brushnetloader_90"]
            controlnetloader_97 = inited_models["controlnetloader_97"]
            controlnetloader_120 = inited_models["controlnetloader_120"]
            srblendbuildpipe_126 = inited_models["srblendbuildpipe_126"]
            layermask_loadflorence2model_131 = inited_models["layermask_loadflorence2model_131"]
            portraitmodelloader_134 = inited_models["portraitmodelloader_134"]
            facemodelloader_136 = inited_models["facemodelloader_136"]
            cliptextencode = inited_models["cliptextencode"]
            
            # 处理默认提示
            if text_prompt is None:
                text_prompt = ClothType.get_text_prompt(cloth_type)
            
            if negative_prompt is None:
                negative_prompt = ClothType.get_negative_prompt()
            
            cliptextencode_50 = cliptextencode.encode(
                text=negative_prompt,
                clip=get_value_at_index(checkpointloadersimple_47, 1),
            )

            # 加载输入图像
            loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            loadimage_58 = loadimage.load_image(image=image_path, abs_path=abs_path, is_ndarray=is_ndarray)

            # 初始化节点
            cr_text = NODE_CLASS_MAPPINGS["CR Text"]()
            cr_text_144 = cr_text.text_multiline(text=text_prompt)

            # 创建处理节点对象
            imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
            facemaskgenerator = NODE_CLASS_MAPPINGS["FaceMaskGenerator"]()
            portraitmaskgenerator = NODE_CLASS_MAPPINGS["PortraitMaskGenerator"]()
            maskmorphology = NODE_CLASS_MAPPINGS["MaskMorphology"]()
            maskcomposite = NODE_CLASS_MAPPINGS["MaskComposite"]()
            layerutility_lama = NODE_CLASS_MAPPINGS["LayerUtility: LaMa"]()
            layerutility_florence2image2prompt = NODE_CLASS_MAPPINGS[
                "LayerUtility: Florence2Image2Prompt"
            ]()
            cr_text_concatenate = NODE_CLASS_MAPPINGS["CR Text Concatenate"]()
            bnk_cutoffbaseprompt = NODE_CLASS_MAPPINGS["BNK_CutoffBasePrompt"]()
            bnk_cutoffsetregions = NODE_CLASS_MAPPINGS["BNK_CutoffSetRegions"]()
            bnk_cutoffregionstoconditioning = NODE_CLASS_MAPPINGS[
                "BNK_CutoffRegionsToConditioning"
            ]()
            powerpaint = NODE_CLASS_MAPPINGS["PowerPaint"]()
            depthanythingv2preprocessor = NODE_CLASS_MAPPINGS[
                "DepthAnythingV2Preprocessor"
            ]()
            controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
            openposepreprocessor = NODE_CLASS_MAPPINGS["OpenposePreprocessor"]()
            ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            converttensortonumpy = NODE_CLASS_MAPPINGS["ConvertTensorToNumpy"]()
            srblendprocess = NODE_CLASS_MAPPINGS["SrBlendProcess"]()
            imagemaskblender = NODE_CLASS_MAPPINGS["ImageMaskBlender"]()
            
            # 执行处理流程
            for q in range(queue_size):
                imageresizekj_114 = imageresizekj.resize(
                    width=768,
                    height=1024,
                    upscale_method="nearest-exact",
                    keep_proportion=True,
                    divisible_by=4,
                    crop="disabled",
                    image=get_value_at_index(loadimage_58, 0),
                )

                facemaskgenerator_137 = facemaskgenerator.generate_faces_mask(
                    max_face=False,
                    num_faces=1,
                    dilate_pixels=10,
                    image=get_value_at_index(imageresizekj_114, 0),
                    face_models=get_value_at_index(facemodelloader_136, 0),
                )

                portraitmaskgenerator_135 = portraitmaskgenerator.generate_portrait_mask(
                    conf_threshold=0.25,
                    iou_threshold=0.5,
                    human_targets="person",
                    matting_threshold=0.1,
                    min_box_area_rate=0.0012,
                    image=get_value_at_index(imageresizekj_114, 0),
                    portrait_models=get_value_at_index(portraitmodelloader_134, 0),
                )

                maskmorphology_146 = maskmorphology.process_mask(
                    pixels=50, mask=get_value_at_index(portraitmaskgenerator_135, 0)
                )

                maskcomposite_140 = maskcomposite.combine(
                    x=0,
                    y=0,
                    operation="and",
                    destination=get_value_at_index(facemaskgenerator_137, 0),
                    source=get_value_at_index(maskmorphology_146, 0),
                )

                layerutility_lama_138 = layerutility_lama.lama(
                    lama_model="lama",
                    device="cuda",
                    invert_mask=False,
                    mask_grow=26,
                    mask_blur=8,
                    image=get_value_at_index(imageresizekj_114, 0),
                    mask=get_value_at_index(portraitmaskgenerator_135, 0),
                )

                layerutility_florence2image2prompt_132 = (
                    layerutility_florence2image2prompt.florence2_image2prompt(
                        task="more detailed caption",
                        text_input="",
                        max_new_tokens=1024,
                        num_beams=3,
                        do_sample=False,
                        fill_mask=False,
                        florence2_model=get_value_at_index(
                            layermask_loadflorence2model_131, 0
                        ),
                        image=get_value_at_index(layerutility_lama_138, 0),
                    )
                )

                cr_text_concatenate_143 = cr_text_concatenate.concat_text(
                    text1=get_value_at_index(cr_text_144, 0),
                    text2=get_value_at_index(layerutility_florence2image2prompt_132, 0),
                    separator="",
                )

                bnk_cutoffbaseprompt_159 = bnk_cutoffbaseprompt.init_prompt(
                    text=get_value_at_index(cr_text_concatenate_143, 0),
                    clip=get_value_at_index(checkpointloadersimple_47, 1),
                )
                
                # 获取当前服装类型的区域映射
                region_mappings = ClothType.get_region_mappings(cloth_type)
                
                # 使用统一的for循环处理区域映射
                # 首先初始化clip_regions
                clip_regions = get_value_at_index(bnk_cutoffbaseprompt_159, 0)
                # 确保从元组中提取字典
                if isinstance(clip_regions, tuple) and len(clip_regions) > 0:
                    clip_regions = clip_regions[0]
                
                # 遍历所有区域映射并添加
                for region_text, target_text in region_mappings:
                    result = bnk_cutoffsetregions.add_clip_region(
                        region_text=region_text,
                        target_text=target_text,
                        weight=1,
                        clip_regions=clip_regions,
                    )
                    # 从返回的元组中提取字典
                    if isinstance(result, tuple) and len(result) > 0:
                        clip_regions = result[0]
                    else:
                        clip_regions = result
                
                # 继续处理流程，使用更新后的clip_regions
                # 检查 clip_regions 是否为元组，若是则提取第一个元素
                if isinstance(clip_regions, tuple) and len(clip_regions) > 0:
                    clip_regions = clip_regions[0]
                    
                bnk_cutoffregionstoconditioning_168 = bnk_cutoffregionstoconditioning.finalize(
                    clip_regions=clip_regions,
                    mask_token="",
                    strict_mask=1.0,
                    start_from_masked=1.0
                )

                powerpaint_65 = powerpaint.model_update(
                    fitting=1,
                    function="context aware",
                    scale=1,
                    start_at=0,
                    end_at=10000,
                    save_memory="none",
                    model=get_value_at_index(checkpointloadersimple_47, 0),
                    vae=get_value_at_index(checkpointloadersimple_47, 2),
                    image=get_value_at_index(imageresizekj_114, 0),
                    mask=get_value_at_index(maskcomposite_140, 0),
                    powerpaint=get_value_at_index(brushnetloader_90, 0),
                    clip=get_value_at_index(powerpaintcliploader_66, 0),
                    positive=get_value_at_index(bnk_cutoffregionstoconditioning_168, 0),
                    negative=get_value_at_index(cliptextencode_50, 0),
                )

                depthanythingv2preprocessor_101 = depthanythingv2preprocessor.execute(
                    ckpt_name="depth_anything_v2_vitl.pth",
                    resolution=1024,
                    image=get_value_at_index(imageresizekj_114, 0),
                )

                controlnetapplyadvanced_96 = controlnetapplyadvanced.apply_controlnet(
                    strength=0.1,
                    start_percent=0,
                    end_percent=1,
                    positive=get_value_at_index(powerpaint_65, 1),
                    negative=get_value_at_index(powerpaint_65, 2),
                    control_net=get_value_at_index(controlnetloader_97, 0),
                    image=get_value_at_index(depthanythingv2preprocessor_101, 0),
                )

                openposepreprocessor_116 = openposepreprocessor.estimate_pose(
                    detect_hand="disable",
                    detect_body="enable",
                    detect_face="enable",
                    resolution=1024,
                    scale_stick_for_xinsr_cn="disable",
                    image=get_value_at_index(imageresizekj_114, 0),
                )

                controlnetapplyadvanced_119 = controlnetapplyadvanced.apply_controlnet(
                    strength=0.4,
                    start_percent=0,
                    end_percent=1,
                    positive=get_value_at_index(controlnetapplyadvanced_96, 0),
                    negative=get_value_at_index(controlnetapplyadvanced_96, 1),
                    control_net=get_value_at_index(controlnetloader_120, 0),
                    image=get_value_at_index(openposepreprocessor_116, 0),
                )

                ksampler_52 = ksampler.sample(
                    seed=777,
                    steps=15,
                    cfg=7.5,
                    sampler_name="euler",
                    scheduler="normal",
                    denoise=1,
                    model=get_value_at_index(powerpaint_65, 0),
                    positive=get_value_at_index(controlnetapplyadvanced_119, 0),
                    negative=get_value_at_index(controlnetapplyadvanced_119, 1),
                    latent_image=get_value_at_index(powerpaint_65, 3),
                )

                vaedecode_54 = vaedecode.decode(
                    samples=get_value_at_index(ksampler_52, 0),
                    vae=get_value_at_index(checkpointloadersimple_47, 2),
                )

                converttensortonumpy_128 = converttensortonumpy.convert(
                    image=get_value_at_index(vaedecode_54, 0)
                )

                srblendprocess_127 = srblendprocess.enhance_process(
                    is_front="False",
                    model=get_value_at_index(srblendbuildpipe_126, 0),
                    src_img=get_value_at_index(converttensortonumpy_128, 0),
                )

                facemaskgenerator_149 = facemaskgenerator.generate_faces_mask(
                    max_face=True,
                    num_faces=1,
                    dilate_pixels=0,
                    image=get_value_at_index(loadimage_58, 0),
                    face_models=get_value_at_index(facemodelloader_136, 0),
                )

                imageresizekj_153 = imageresizekj.resize(
                    width=512,
                    height=512,
                    upscale_method="lanczos",
                    keep_proportion=True,
                    divisible_by=1,
                    crop="disabled",
                    image=get_value_at_index(srblendprocess_127, 1),
                    get_image_size=get_value_at_index(loadimage_58, 0),
                )

                imagemaskblender_150 = imagemaskblender.blend_images(
                    feather=30,
                    image_fg=get_value_at_index(imageresizekj_153, 0),
                    image_bg=get_value_at_index(loadimage_58, 0),
                    mask=get_value_at_index(facemaskgenerator_149, 0),
                )

                # 返回最终处理结果
                outimg = get_value_at_index(imagemaskblender_150, 0) * 255.0
                return outimg.cpu().numpy().astype(np.uint8)[0]
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
    description="虚拟试衣工作流。可选择不同服装类型进行试穿。"
)
parser.add_argument(
    "text1",
    nargs="?",
    default=None,
    help='可选参数: 自定义文本提示，覆盖预设的服装提示词',
)

parser.add_argument(
    "clip2",
    nargs="?",
    default=None,
    help='可选参数: 自定义CLIP模型',
)

parser.add_argument(
    "--cloth-type",
    "-t",
    dest="cloth_type",
    type=str,
    choices=[t.value for t in ClothType],
    default=ClothType.HANFU.value,
    help=f"指定服装类型，可选值: {', '.join([t.value for t in ClothType])} (默认: {ClothType.HANFU.value})",
)

parser.add_argument(
    "--image-path",
    "-i",
    dest="image_path",
    type=str,
    default="input/lixy_outdoor.jpeg",
    help="输入图像路径 (默认: input/lixy_outdoor.jpeg)",
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
        ordered_args = dict(zip(["text1", "clip2"], func_args))

        all_args = dict()
        all_args.update(defaults)
        all_args.update(ordered_args)
        all_args.update(func_kwargs)

        args = argparse.Namespace(**all_args)

    # with ctx:
    #     if not _custom_path_added:
    #         add_comfyui_directory_to_sys_path()
    #         add_extra_model_paths()
    #         _custom_path_added = True

    #     if not _custom_nodes_imported:
    #         import_custom_nodes()
    #         _custom_nodes_imported = True

    # 初始化模型
    inited_models = init_cloth_vton_models()
    
    # 检查是否传入了cloth_type参数
    cloth_type = None
    if hasattr(args, 'cloth_type') and args.cloth_type:
        # 尝试从字符串获取对应的ClothType枚举
        try:
            cloth_type = ClothType(args.cloth_type)
        except (ValueError, KeyError):
            # 如果无效，使用默认值
            cloth_type = ClothType.HANFU
    else:
        # 默认使用汉服
        cloth_type = ClothType.HANFU
    
    # 检查图像路径
    image_path = "lixy_outdoor.jpeg"
    if hasattr(args, 'image_path') and args.image_path:
        image_path = args.image_path
    
    # 使用预加载的模型进行处理
    return cloth_vton_process(
        inited_models=inited_models,
        image_path=image_path,
        cloth_type=cloth_type,
        text_prompt=parse_arg(args.text1) if hasattr(args, 'text1') and args.text1 else None,
        clip=parse_arg(args.clip2) if hasattr(args, 'clip2') and args.clip2 else None,
        queue_size=args.queue_size
    )
import cv2
if __name__ == "__main__":
    result = main()
    # 将结果保存为图像文件
    output_path = "output/cloth_vton_result.png"
    cv2.imwrite(output_path, result)
    print(f"结果已保存到 {output_path}")

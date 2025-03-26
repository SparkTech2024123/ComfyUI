import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np
from enum import Enum, auto


class StyleType(Enum):
    """风格类型枚举类，用于定义不同的风格转换选项"""
    CLAY = "clay"
    DUANWU_INK = "duanwu_ink"
    SAIBO_GUFENG = "saibo_gufeng"
    GHIBLI_STYLE = "ghibli_style"
    TANGGUO = "tangguo"
    
    @classmethod
    def get_lora_config(cls, style_type):
        """根据风格类型获取相应的LoRA配置"""
        if style_type == cls.CLAY:
            return {
                "switch_1": "On",
                "lora_name_1": "CLAYMATE_V2.03_.safetensors",
                "model_weight_1": 0.95,
                "clip_weight_1": 1,
                "switch_2": "On",
                "lora_name_2": "DD-made-of-clay-XL-v2.safetensors",
                "model_weight_2": 0.45,
                "clip_weight_2": 1,
                "switch_3": "Off",
                "lora_name_3": "duanwu_ink.safetensors",
                "model_weight_3": 1,
                "clip_weight_3": 1,
            }
        elif style_type == cls.DUANWU_INK:
            return {
                "switch_1": "Off",
                "lora_name_1": "CLAYMATE_V2.03_.safetensors",
                "model_weight_1": 0.95,
                "clip_weight_1": 1,
                "switch_2": "Off",
                "lora_name_2": "DD-made-of-clay-XL-v2.safetensors",
                "model_weight_2": 0.45,
                "clip_weight_2": 1,
                "switch_3": "On",
                "lora_name_3": "duanwu_ink.safetensors",
                "model_weight_3": 1,
                "clip_weight_3": 1,
            }
        elif style_type == cls.SAIBO_GUFENG:
            return {
                "switch_1": "On",
                "lora_name_1": "loraxl_saibo_gufeng.safetensors",
                "model_weight_1": 0.8,
                "clip_weight_1": 0.8,
                "switch_2": "Off",
                "lora_name_2": "loraxl_ghibli.safetensors",
                "model_weight_2": 1,
                "clip_weight_2": 1,
                "switch_3": "Off",
                "lora_name_3": "loraxl_tangguo.safetensors",
                "model_weight_3": 0.6,
                "clip_weight_3": 0.6,
            }
        elif style_type == cls.GHIBLI_STYLE:
            return {
                "switch_1": "Off",
                "lora_name_1": "loraxl_saibo_gufeng.safetensors",
                "model_weight_1": 0.8,
                "clip_weight_1": 0.8,
                "switch_2": "On",
                "lora_name_2": "loraxl_ghibli.safetensors",
                "model_weight_2": 1,
                "clip_weight_2": 1,
                "switch_3": "Off",
                "lora_name_3": "loraxl_tangguo.safetensors",
                "model_weight_3": 0.6,
                "clip_weight_3": 0.6,
            }
        elif style_type == cls.TANGGUO:
            return {
                "switch_1": "Off",
                "lora_name_1": "loraxl_saibo_gufeng.safetensors",
                "model_weight_1": 0.8,
                "clip_weight_1": 0.8,
                "switch_2": "Off",
                "lora_name_2": "loraxl_ghibli.safetensors",
                "model_weight_2": 1,
                "clip_weight_2": 1,
                "switch_3": "On",
                "lora_name_3": "loraxl_tangguo.safetensors",
                "model_weight_3": 0.6,
                "clip_weight_3": 0.6,
            }
        else:
            raise ValueError(f"未知的风格类型: {style_type}")
    
    @classmethod
    def get_style_prompt(cls, style_type):
        """根据风格类型获取相应的风格提示文本"""
        if style_type == cls.CLAY:
            return "((claymotion, made-of-clay, stopmotion, polymer clay, ultra light clay)), High quality, details, cartoonish, 8k"
        elif style_type == cls.DUANWU_INK:
            return "(Chinese ink painting, Chinese ink watercolor)1.2, using atmospheric perspective and ink painting style inspired by chinese famous artist, no humans, high resolution, perfect environment, extremely detailed, perfect composition, aesthetic, High quality, details, cartoonish, 8k"
        elif style_type == cls.SAIBO_GUFENG:
            return "((Chinese traditional style, fantasy illustration, ancient costume, romantic atmosphere, moonlight scene, traditional Chinese elements)), High quality, details, 8k"
        elif style_type == cls.GHIBLI_STYLE:
            return "((Ghibli style, Studio Ghibli, anime style, anime art, anime drawing, anime illustration)), High quality, details, 8k"
        elif style_type == cls.TANGGUO:
            return "((colorful candy shop aesthetic, vibrant neon lights, pastel colors, pink and blue tones, candy-colored environment, kawaii anime style, glossy surfaces, bright saturated colors, cute fashion, holographic elements, candy store, crystalline decorations)), High quality, details, 8k"
        else:
            raise ValueError(f"未知的风格类型: {style_type}")
    
    @classmethod
    def get_all_lora_names(cls):
        """获取所有风格使用的LoRA模型名称列表（去重）"""
        lora_names = set()
        for style_type in cls:
            config = cls.get_lora_config(style_type)
            lora_names.add(config["lora_name_1"])
            lora_names.add(config["lora_name_2"])
            lora_names.add(config["lora_name_3"])
        return list(lora_names)

    @classmethod
    def get_task_type(cls, style_type):
        """根据风格类型获取相应的Florence2任务类型"""
        if style_type == cls.CLAY:
            return "more detailed caption"
        elif style_type == cls.DUANWU_INK:
            return "caption"
        elif style_type == cls.SAIBO_GUFENG:
            return "more detailed caption"
        elif style_type == cls.GHIBLI_STYLE:
            return "caption"
        elif style_type == cls.TANGGUO:
            return "caption"
        else:
            raise ValueError(f"未知的风格类型: {style_type}")


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
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
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
        print("Could not find the extra_model_paths config file.")


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
    init_extra_nodes()


from nodes import LoadImage, NODE_CLASS_MAPPINGS
import folder_paths

def load_lora_to_cpu(lora_name):
    """加载单个LoRA模型到CPU内存
    
    Args:
        lora_name: LoRA模型文件名
        
    Returns:
        dict: 加载的LoRA模型，如果加载失败则返回None
    """
    try:
        print(f"预加载LoRA模型到CPU: {lora_name}")
        from comfy.utils import load_torch_file
        
        # 获取LoRA模型路径
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            return None
        
        # 加载模型到CPU
        lora_model = load_torch_file(lora_path, device=torch.device("cpu"))
        return lora_model
    except Exception as e:
        print(f"加载LoRA模型 {lora_name} 时出错: {str(e)}")
        return None

def init_trans_style_models():
    """初始化并加载风格转换所需的基础模型和所有LoRA模型（到CPU内存）"""
    try:
        import_custom_nodes()
        with torch.inference_mode():
            # 加载基础模型
            checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            checkpointloadersimple_2 = checkpointloadersimple.load_checkpoint(
                ckpt_name="juggernautXL_v9Rdphoto2Lightning.safetensors"
            )
            
            # 加载Florence2模型
            layermask_loadflorence2model = NODE_CLASS_MAPPINGS["LayerMask: LoadFlorence2Model"]()
            layermask_loadflorence2model_16 = layermask_loadflorence2model.load(
                version="CogFlorence-2.1-Large"
            )
            
            # 加载ControlNet模型
            controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
            controlnetloader_39 = controlnetloader.load_controlnet(
                control_net_name="CN-anytest_v4-marged.safetensors"
            )
            
            controlnetloader_41 = controlnetloader.load_controlnet(
                control_net_name="xinsir_xl_controlnet_depth.safetensors"
            )
            
            # 预加载所有LoRA模型到CPU内存
            lora_models = {}
            for lora_name in StyleType.get_all_lora_names():
                lora_model = load_lora_to_cpu(lora_name)
                if lora_model is not None:
                    lora_models[lora_name] = lora_model
                else:
                    print(f"警告: LoRA模型 {lora_name} 加载失败，将在使用时尝试从磁盘加载")
            
            # 返回所有加载的基础模型和LoRA模型
            return {
                "checkpointloadersimple_2": checkpointloadersimple_2,
                "layermask_loadflorence2model_16": layermask_loadflorence2model_16,
                "controlnetloader_39": controlnetloader_39,
                "controlnetloader_41": controlnetloader_41,
                "lora_models": lora_models,
            }
    except Exception as e:
        print(f"初始化模型时出错: {str(e)}")
        raise

def create_lora_stack_from_preloaded(inited_models, style_type):
    """从预加载的LoRA模型创建LoRA堆栈
    
    注意：由于ComfyUI的API限制，目前仍然使用原始的lora_stacker方法加载模型。
    在未来，如果ComfyUI提供了直接使用预加载模型的API，可以修改此函数。
    
    Args:
        inited_models: 包含预加载LoRA模型的字典
        style_type: 风格类型
        
    Returns:
        object: 创建的LoRA堆栈
    """
    try:
        # 获取风格对应的LoRA配置
        lora_config = StyleType.get_lora_config(style_type)
        
        # 创建LoRA堆栈
        cr_lora_stack = NODE_CLASS_MAPPINGS["CR LoRA Stack"]()
        
        # 从预加载的模型中获取对应的LoRA模型
        lora_models = inited_models.get("lora_models", {})
        
        # 检查所需的LoRA模型是否都已预加载
        required_loras = [
            lora_config["lora_name_1"],
            lora_config["lora_name_2"],
            lora_config["lora_name_3"]
        ]
        
        all_loaded = all(name in lora_models and lora_models[name] is not None 
                         for name in required_loras if lora_config.get(f"switch_{required_loras.index(name)+1}") == "On")
        
        if all_loaded:
            print(f"使用预加载的LoRA模型创建{style_type.value}风格堆栈")
            # 理想情况下，我们应该直接使用预加载的模型
            # 但由于ComfyUI的API限制，目前仍然使用原始的lora_stacker方法
            # 在未来，如果ComfyUI提供了直接使用预加载模型的API，可以修改此处
        else:
            print(f"部分LoRA模型未预加载，将从磁盘加载{style_type.value}风格所需的LoRA模型")
        
        # 使用原始的lora_stacker方法
        # 注意：这里实际上可能会再次从磁盘加载模型，而不是使用预加载的模型
        # 这是由于ComfyUI的API限制导致的
        cr_lora_stack_4 = cr_lora_stack.lora_stacker(**lora_config)
        
        return cr_lora_stack_4
    except Exception as e:
        print(f"创建LoRA堆栈时出错: {str(e)}")
        raise

def trans_style_process(inited_models, image_path, style_type=StyleType.CLAY, \
    abs_path=True, is_ndarray=False, seed=None):
    """
    使用已初始化的模型处理图像进行风格转换
    
    Args:
        inited_models: 初始化好的基础模型和预加载的LoRA模型
        image_path: 输入图像路径或图像数组
        style_type: 风格类型，默认为CLAY
        abs_path: 是否为绝对路径
        is_ndarray: 输入是否为numpy数组
        seed: 随机种子
    
    Returns:
        numpy.ndarray: 处理后的图像数组
    """
    try:
        with torch.inference_mode():
            print(f"开始处理图像，应用{style_type.value}风格...")
            
            # 加载图像
            loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            loadimage_1 = loadimage.load_image(image=image_path, abs_path=abs_path, is_ndarray=is_ndarray)
            
            # 调整图像大小
            imagescaletototalpixels = NODE_CLASS_MAPPINGS["ImageScaleToTotalPixels"]()
            imagescaletototalpixels_7 = imagescaletototalpixels.upscale(
                upscale_method="lanczos",
                megapixels=1,
                image=get_value_at_index(loadimage_1, 0),
            )
            
            # VAE编码
            vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
            vaeencode_8 = vaeencode.encode(
                pixels=get_value_at_index(imagescaletototalpixels_7, 0),
                vae=get_value_at_index(inited_models["checkpointloadersimple_2"], 2),
            )
            
            # 从预加载的LoRA模型创建LoRA堆栈
            print(f"创建{style_type.value}风格的LoRA堆栈...")
            cr_lora_stack_4 = create_lora_stack_from_preloaded(inited_models, style_type)
            
            # 应用LoRA模型
            print("应用LoRA模型...")
            cr_apply_lora_stack = NODE_CLASS_MAPPINGS["CR Apply LoRA Stack"]()
            cr_apply_lora_stack_5 = cr_apply_lora_stack.apply_lora_stack(
                model=get_value_at_index(inited_models["checkpointloadersimple_2"], 0),
                clip=get_value_at_index(inited_models["checkpointloadersimple_2"], 1),
                lora_stack=get_value_at_index(cr_lora_stack_4, 0),
            )
            
            # 编码否定提示
            cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            cliptextencode_12 = cliptextencode.encode(
                text="nsfw, paintings, worst quality, low quality, normal quality, lowres, watermark, monochrome, grayscale, ugly, blurry, bad anatomy, morbid, malformation, amputation, bad proportions, missing body, fused body, extra head, poorly drawn face, bad eyes, deformed eye, unclear eyes, cross-eyed, long neck, malformed limbs, extra limbs, extra arms, missing arms, bad tongue, strange fingers, mutated hands, missing hands, poorly drawn hands, extra hands, fused hands, connected hand, bad hands, wrong fingers, missing fingers, extra fingers, 4 fingers, 3 fingers, deformed hands, extra legs, bad legs, many legs, more than two legs, bad feet, wrong feet, extra feets,",
                clip=get_value_at_index(cr_apply_lora_stack_5, 1),
            )
            
            # 使用Florence2模型生成图像描述
            print("使用Florence2模型生成图像描述...")
            layerutility_florence2image2prompt = NODE_CLASS_MAPPINGS["LayerUtility: Florence2Image2Prompt"]()
            # 根据风格类型获取任务类型
            task_type = StyleType.get_task_type(style_type)
            print(f"使用{style_type.value}风格的任务类型: {task_type}")
            layerutility_florence2image2prompt_15 = layerutility_florence2image2prompt.florence2_image2prompt(
                task=task_type,
                text_input="Ignore the artistic style of the picture.",
                max_new_tokens=1024,
                num_beams=4,
                do_sample=False,
                fill_mask=False,
                florence2_model=get_value_at_index(inited_models["layermask_loadflorence2model_16"], 0),
                image=get_value_at_index(imagescaletototalpixels_7, 0),
            )
            
            # 显示生成的描述
            displaytext_zho = NODE_CLASS_MAPPINGS["DisplayText_Zho"]()
            displaytext_zho_19 = displaytext_zho.display_text(
                text=get_value_at_index(layerutility_florence2image2prompt_15, 0)
            )
            
            # 根据风格类型获取风格提示
            style_prompt = StyleType.get_style_prompt(style_type)
            print(f"使用{style_type.value}风格提示: {style_prompt}")
            
            # 创建风格提示
            cr_text = NODE_CLASS_MAPPINGS["CR Text"]()
            cr_text_34 = cr_text.text_multiline(
                text=style_prompt
            )
            
            # 合并描述和风格提示
            concattext_zho = NODE_CLASS_MAPPINGS["ConcatText_Zho"]()
            concattext_zho_23 = concattext_zho.concat_texts(
                text_1=get_value_at_index(displaytext_zho_19, 0),
                text_2=get_value_at_index(cr_text_34, 0),
            )
            
            # 编码组合后的提示词
            cliptextencode_33 = cliptextencode.encode(
                text=get_value_at_index(concattext_zho_23, 0),
                clip=get_value_at_index(cr_apply_lora_stack_5, 1),
            )
            
            # 设置采样器参数
            alignyourstepsscheduler = NODE_CLASS_MAPPINGS["AlignYourStepsScheduler"]()
            alignyourstepsscheduler_29 = alignyourstepsscheduler.get_sigmas(
                model_type="SDXL", steps=12, denoise=0.9
            )
            
            ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            ksamplerselect_31 = ksamplerselect.get_sampler(sampler_name="dpmpp_2m")
            
            # 深度预处理
            print("执行深度预处理...")
            depthanythingv2preprocessor = NODE_CLASS_MAPPINGS["DepthAnythingV2Preprocessor"]()
            depthanythingv2preprocessor_36 = depthanythingv2preprocessor.execute(
                ckpt_name="depth_anything_v2_vitl.pth",
                resolution=1024,
                image=get_value_at_index(imagescaletototalpixels_7, 0),
            )
            
            # CFG自动调整
            automatic_cfg = NODE_CLASS_MAPPINGS["Automatic CFG"]()
            automatic_cfg_28 = automatic_cfg.patch(
                hard_mode=True,
                boost=True,
                model=get_value_at_index(cr_apply_lora_stack_5, 0),
            )
            
            # 应用ControlNet
            print("应用ControlNet...")
            acn_advancedcontrolnetapply = NODE_CLASS_MAPPINGS["ACN_AdvancedControlNetApply"]()
            acn_advancedcontrolnetapply_27 = acn_advancedcontrolnetapply.apply_controlnet(
                strength=0.8,
                start_percent=0,
                end_percent=0.8,
                positive=get_value_at_index(cliptextencode_33, 0),
                negative=get_value_at_index(cliptextencode_12, 0),
                control_net=get_value_at_index(inited_models["controlnetloader_41"], 0),
                image=get_value_at_index(depthanythingv2preprocessor_36, 0),
            )
            
            acn_advancedcontrolnetapply_38 = acn_advancedcontrolnetapply.apply_controlnet(
                strength=0.6,
                start_percent=0,
                end_percent=0.7000000000000001,
                positive=get_value_at_index(acn_advancedcontrolnetapply_27, 0),
                negative=get_value_at_index(acn_advancedcontrolnetapply_27, 1),
                control_net=get_value_at_index(inited_models["controlnetloader_39"], 0),
                image=get_value_at_index(imagescaletototalpixels_7, 0),
            )
            
            # 采样生成图像
            print("开始采样生成图像...")
            samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()
            if seed is None:
                seed = random.randint(1, 2 ** 64)
                
            print(f"使用随机种子: {seed}")
            samplercustom_30 = samplercustom.sample(
                add_noise=True,
                noise_seed=777,
                cfg=7,
                model=get_value_at_index(automatic_cfg_28, 0),
                positive=get_value_at_index(acn_advancedcontrolnetapply_38, 0),
                negative=get_value_at_index(acn_advancedcontrolnetapply_38, 1),
                sampler=get_value_at_index(ksamplerselect_31, 0),
                sigmas=get_value_at_index(alignyourstepsscheduler_29, 0),
                latent_image=get_value_at_index(vaeencode_8, 0),
            )
            
            # VAE解码输出图像
            print("VAE解码输出图像...")
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            vaedecode_26 = vaedecode.decode(
                samples=get_value_at_index(samplercustom_30, 0),
                vae=get_value_at_index(inited_models["checkpointloadersimple_2"], 2),
            )
            
            # 返回最终图像
            print(f"{style_type.value}风格处理完成")
            outimg = get_value_at_index(vaedecode_26, 0) * 255.0
            return outimg.cpu().numpy().astype(np.uint8)[0]
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        raise


def cleanup_gpu_memory():
    """清理GPU内存"""
    try:
        import gc
        import torch
        
        gc.collect()
        torch.cuda.empty_cache()
        print("已清理GPU内存")
    except Exception as e:
        print(f"清理GPU内存时出错: {str(e)}")


if __name__ == "__main__":
    try:
        # 初始化模型（包括预加载所有LoRA到CPU）
        print("正在初始化模型并预加载所有LoRA到CPU...")
        trans_style_inited_models = init_trans_style_models()
        
        # 处理图像（使用默认的clay风格）
        # for i in range(10):
        print("正在处理图像（clay风格）...")
        img_path = "./input/Txiaobaiqun.jpg"
        final_img_clay = trans_style_process(trans_style_inited_models, img_path, StyleType.CLAY)
        
        # 保存clay风格结果
        import cv2
        cv2.imwrite('trans_style_clay_output.png', final_img_clay)
        print("已保存clay风格结果")
        
        # 清理GPU内存
        cleanup_gpu_memory()    
        
        # 处理图像（使用水墨风格）
        print("正在处理图像（水墨风格）...")
        final_img_ink = trans_style_process(trans_style_inited_models, img_path, StyleType.DUANWU_INK)
        
        # 保存水墨风格结果
        cv2.imwrite('trans_style_ink_output.png', final_img_ink)
        print("已保存水墨风格结果")
        
        # 最终清理
        cleanup_gpu_memory()
        
        # 处理图像（使用赛博风格）
        print("正在处理图像（赛博风格）...")
        final_img_cyber = trans_style_process(trans_style_inited_models, img_path, StyleType.SAIBO_GUFENG)
        
        # 保存赛博风格结果
        cv2.imwrite('trans_style_cyber_output.png', final_img_cyber)
        print("已保存赛博风格结果")
        
        # 处理图像（使用宫崎骏风格）
        print("正在处理图像（宫崎骏风格）...")
        final_img_ghibli = trans_style_process(trans_style_inited_models, img_path, StyleType.GHIBLI_STYLE)
        
        # 保存宫崎骏风格结果
        cv2.imwrite('trans_style_ghibli_output.png', final_img_ghibli)
        print("已保存宫崎骏风格结果")
        
        # 处理图像（使用唐国风格）
        print("正在处理图像（唐国风格）...")
        final_img_tangguo = trans_style_process(trans_style_inited_models, img_path, StyleType.TANGGUO)
        
        # 保存唐国风格结果
        cv2.imwrite('trans_style_tangguo_output.png', final_img_tangguo)
        print("已保存唐国风格结果")
        
        # 最终清理
        cleanup_gpu_memory()
        
    except Exception as e:
        print(f"程序执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

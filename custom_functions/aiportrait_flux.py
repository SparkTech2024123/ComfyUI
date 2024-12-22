from ast import main
import os
import random
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
        # path = os.getcwd()
        path = os.path.dirname(__file__)
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

def init_flux_models(device1, device2):
    import_custom_nodes()
    with torch.inference_mode():
        pulidfluxmodelloader = NODE_CLASS_MAPPINGS["PulidFluxModelLoader"]()
        pulidfluxmodelloader_45 = pulidfluxmodelloader.load_model(
            pulid_file="pulid_flux_v0.9.1.safetensors"
        )

        pulidfluxevacliploader = NODE_CLASS_MAPPINGS["PulidFluxEvaClipLoader"]()
        pulidfluxevacliploader_51 = pulidfluxevacliploader.load_eva_clip()

        pulidfluxinsightfaceloader = NODE_CLASS_MAPPINGS["PulidFluxInsightFaceLoader"]()
        pulidfluxinsightfaceloader_53 = pulidfluxinsightfaceloader.load_insightface(
            provider="CUDA"
        )

        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_132 = dualcliploader.load_clip(
            clip_name1="t5xxl_fp16.safetensors",
            clip_name2="clip_l.safetensors",
            type="flux",
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_135 = vaeloader.load_vae(vae_name="ae.safetensors")

        preprocnewbuildpipe = NODE_CLASS_MAPPINGS["PreprocNewBuildPipe"]()
        preprocnewbuildpipe_167 = preprocnewbuildpipe.load_models(
            wd14_cgpath="wd14_tagger"
        )

        overrideclipdevice = NODE_CLASS_MAPPINGS["OverrideCLIPDevice"]()
        overrideclipdevice_133 = overrideclipdevice.patch(
            device=device2, clip=get_value_at_index(dualcliploader_132, 0)
        )

        unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unetloadergguf_142 = unetloadergguf.load_unet(unet_name="flux1-dev-Q8_0.gguf")

        overridevaedevice = NODE_CLASS_MAPPINGS["OverrideVAEDevice"]()
        overridevaedevice_134 = overridevaedevice.patch(
            device=device1, vae=get_value_at_index(vaeloader_135, 0)
        )

        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        loraloadermodelonly_152 = loraloadermodelonly.load_lora_model_only(
            lora_name="Hyper-FLUX.1-dev-8steps-lora.safetensors",
            strength_model=0.13,
            model=get_value_at_index(unetloadergguf_142, 0),
        )

        loraloadermodelonly_153 = loraloadermodelonly.load_lora_model_only(
            lora_name="flux1-depth-dev-lora.safetensors",
            strength_model=1,
            model=get_value_at_index(loraloadermodelonly_152, 0),
        )

        srblendbuildpipe = NODE_CLASS_MAPPINGS["SrBlendBuildPipe"]()
        srblendbuildpipe_162 = srblendbuildpipe.load_models(
            lut_path="lut", gpu_choose=device2, sr_type="RealESRGAN_X2", half=True
        )

        gpenpbuildipeline = NODE_CLASS_MAPPINGS["GPENPBuildipeline"]()
        gpenpbuildipeline_164 = gpenpbuildipeline.load_model(
            model="GPEN-BFR-1024.pth",
            in_size=1024,
            channel_multiplier=2,
            narrow=1,
            alpha=1,
            device=device2,
        )
        
        facewarppipebuilder = NODE_CLASS_MAPPINGS["FaceWarpPipeBuilder"]()
        facewarppipebuilder_177 = facewarppipebuilder.load_models(
            detect_model_path="facedetect/scrfd_10g_bnkps_shape640x640.onnx",
            deca_dir="deca",
            gpu_choose=device2,
        )

        faceswappipebuilder = NODE_CLASS_MAPPINGS["FaceSwapPipeBuilder"]()
        faceswappipebuilder_180 = faceswappipebuilder.load_models(
            swap_own_model="faceswap/swapper_own.pth",
            arcface_model="faceswap/arcface_checkpoint.tar",
            facealign_config_dir="face_align",
            phase1_model="facealign/p1.pt",
            phase2_model="facealign/p2.pt",
            device=device2,
        )
        
        # return loaded models
        return {
            'pulidfluxmodelloader_45' : pulidfluxmodelloader_45,
            'pulidfluxevacliploader_51' : pulidfluxevacliploader_51,
            'pulidfluxinsightfaceloader_53' : pulidfluxinsightfaceloader_53,
            'dualcliploader_132' : dualcliploader_132,
            'vaeloader_135' : vaeloader_135,
            'preprocnewbuildpipe_167' : preprocnewbuildpipe_167,
            'overrideclipdevice_133' : overrideclipdevice_133,
            'unetloadergguf_142' : unetloadergguf_142,
            'overridevaedevice_134' : overridevaedevice_134,
            'loraloadermodelonly_152' : loraloadermodelonly_152,
            'loraloadermodelonly_153' : loraloadermodelonly_153,
            'srblendbuildpipe_162' : srblendbuildpipe_162,
            'gpenpbuildpipeline_164' : gpenpbuildipeline_164,
            "facewarppipebuilder_177" : facewarppipebuilder_177,
            "faceswappipebuilder_180" : faceswappipebuilder_180,
        }

# flux_inited_models = init_flux_models()

def ai_portrait_flux(flux_inited_models,image_path, template_id, abs_path=True, is_ndarray=False):
    # import_custom_nodes()
    with torch.inference_mode():
        loadimage = LoadImage()
        loadimage_104 = loadimage.load_image(image=image_path, abs_path=abs_path, is_ndarray=is_ndarray)

        preprocnewgetconds = NODE_CLASS_MAPPINGS["PreprocNewGetConds"]()
        preprocnewgetconds_168 = preprocnewgetconds.prepare(
            template_id=template_id,
            is_front="False",
            model=get_value_at_index(flux_inited_models['preprocnewbuildpipe_167'], 0),
            src_img=get_value_at_index(loadimage_104, 0),
        )

        preprocnewsplitconds = NODE_CLASS_MAPPINGS["PreprocNewSplitConds"]()
        preprocnewsplitconds_169 = preprocnewsplitconds.split(
            pipe_conditions=get_value_at_index(preprocnewgetconds_168, 0)
        )


        cliptextencodeflux = NODE_CLASS_MAPPINGS["CLIPTextEncodeFlux"]()
        cliptextencodeflux_137 = cliptextencodeflux.encode(
            clip_l=get_value_at_index(preprocnewsplitconds_169, 1),
            t5xxl=get_value_at_index(preprocnewsplitconds_169, 1),
            guidance=4.0,
            clip=get_value_at_index(flux_inited_models['overrideclipdevice_133'], 0),
        )

        cliptextencodeflux_138 = cliptextencodeflux.encode(
            clip_l=get_value_at_index(preprocnewsplitconds_169, 2),
            t5xxl=get_value_at_index(preprocnewsplitconds_169, 2),
            guidance=4.0,
            clip=get_value_at_index(flux_inited_models['overrideclipdevice_133'], 0),
        )

        convertnumpytotensor = NODE_CLASS_MAPPINGS["ConvertNumpyToTensor"]()
        convertnumpytotensor_176 = convertnumpytotensor.convert(
            image=get_value_at_index(preprocnewsplitconds_169, 14)
        )

        instructpixtopixconditioning = NODE_CLASS_MAPPINGS[
            "InstructPixToPixConditioning"
        ]()
        instructpixtopixconditioning_151 = instructpixtopixconditioning.encode(
            positive=get_value_at_index(cliptextencodeflux_137, 0),
            negative=get_value_at_index(cliptextencodeflux_138, 0),
            vae=get_value_at_index(flux_inited_models['overridevaedevice_134'], 0),
            pixels=get_value_at_index(convertnumpytotensor_176, 0),
        )


        applypulidflux = NODE_CLASS_MAPPINGS["ApplyPulidFlux"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        converttensortonumpy = NODE_CLASS_MAPPINGS["ConvertTensorToNumpy"]()
        facewarpdetectfacesmethod = NODE_CLASS_MAPPINGS["FaceWarpDetectFacesMethod"]()
        faceswapdetectpts = NODE_CLASS_MAPPINGS["FaceSwapDetectPts"]()
        faceswapmethod = NODE_CLASS_MAPPINGS["FaceSwapMethod"]()
        gpenprocess = NODE_CLASS_MAPPINGS["GPENProcess"]()
        srblendprocess = NODE_CLASS_MAPPINGS["SrBlendProcess"]()

        # for q in range(1):
        convertnumpytotensor_170 = convertnumpytotensor.convert(
            image=get_value_at_index(preprocnewsplitconds_169, 0)
        )

        applypulidflux_62 = applypulidflux.apply_pulid_flux(
            weight=0.8,
            start_at=0,
            end_at=1,
            fusion="mean",
            fusion_weight_max=1,
            fusion_weight_min=0,
            train_step=1000,
            use_gray=True,
            model=get_value_at_index(flux_inited_models['loraloadermodelonly_153'], 0),
            pulid_flux=get_value_at_index(flux_inited_models['pulidfluxmodelloader_45'], 0),
            eva_clip=get_value_at_index(flux_inited_models['pulidfluxevacliploader_51'], 0),
            face_analysis=get_value_at_index(flux_inited_models['pulidfluxinsightfaceloader_53'], 0),
            image=get_value_at_index(convertnumpytotensor_170, 0),
            unique_id=16129776131516065149,
        )

        ksampler_73 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=10,
            cfg=1,
            sampler_name="euler",
            scheduler="beta",
            denoise=1,
            model=get_value_at_index(applypulidflux_62, 0),
            positive=get_value_at_index(instructpixtopixconditioning_151, 0),
            negative=get_value_at_index(instructpixtopixconditioning_151, 1),
            latent_image=get_value_at_index(instructpixtopixconditioning_151, 2),
        )

        vaedecode_69 = vaedecode.decode(
            samples=get_value_at_index(ksampler_73, 0),
            vae=get_value_at_index(flux_inited_models['overridevaedevice_134'], 0),
        )

        converttensortonumpy_166 = converttensortonumpy.convert(
            image=get_value_at_index(vaedecode_69, 0)
        )
        
        # 检查template_id是否包含boy或girl
        is_child = 'boy' in template_id.lower() or 'girl' in template_id.lower()
        
        if not is_child:
            facewarpdetectfacesmethod_179 = facewarpdetectfacesmethod.detect_faces(
                model=get_value_at_index(flux_inited_models['facewarppipebuilder_177'], 0),
                image=get_value_at_index(preprocnewsplitconds_169, 0),
            )

            faceswapdetectpts_185 = faceswapdetectpts.detect_face_pts(
                ptstype="5",
                model=get_value_at_index(flux_inited_models['faceswappipebuilder_180'], 0),
                src_image=get_value_at_index(preprocnewsplitconds_169, 0),
                src_faces=get_value_at_index(facewarpdetectfacesmethod_179, 0),
            )

            facewarpdetectfacesmethod_184 = facewarpdetectfacesmethod.detect_faces(
                model=get_value_at_index(flux_inited_models['facewarppipebuilder_177'], 0),
                image=get_value_at_index(converttensortonumpy_166, 0),
            )

            faceswapdetectpts_183 = faceswapdetectpts.detect_face_pts(
                ptstype="5",
                model=get_value_at_index(flux_inited_models['faceswappipebuilder_180'], 0),
                src_image=get_value_at_index(converttensortonumpy_166, 0),
                src_faces=get_value_at_index(facewarpdetectfacesmethod_184, 0),
            )

            faceswapdetectpts_181 = faceswapdetectpts.detect_face_pts(
                ptstype="256",
                model=get_value_at_index(flux_inited_models['faceswappipebuilder_180'], 0),
                src_image=get_value_at_index(converttensortonumpy_166, 0),
                src_faces=get_value_at_index(facewarpdetectfacesmethod_184, 0),
            )

            faceswapmethod_182 = faceswapmethod.swap_face(
                model=get_value_at_index(flux_inited_models['faceswappipebuilder_180'], 0),
                src_image=get_value_at_index(preprocnewsplitconds_169, 0),
                two_stage_image=get_value_at_index(converttensortonumpy_166, 0),
                source_5pts=get_value_at_index(faceswapdetectpts_185, 0),
                target_5pts=get_value_at_index(faceswapdetectpts_183, 0),
                target_256pts=get_value_at_index(faceswapdetectpts_181, 0),
            )
            
            gpen_input = get_value_at_index(faceswapmethod_182, 0)
        else:
            gpen_input = get_value_at_index(converttensortonumpy_166, 0)

        gpenprocess_165 = gpenprocess.enhance_face(
            aligned=False,
            model=get_value_at_index(flux_inited_models['gpenpbuildpipeline_164'], 0),
            image=gpen_input,
        )

        _, final_img = srblendprocess.enhance_process(
            is_front="False",
            model=get_value_at_index(flux_inited_models['srblendbuildpipe_162'], 0),
            src_img=get_value_at_index(gpenprocess_165, 0),
        )
        outimg = get_value_at_index(final_img, 0) * 255.
        return outimg.cpu().numpy().astype(np.uint8)

if __name__ == "__main__":
    # main()
    pass
    pass
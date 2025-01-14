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

def init_portrait_own_models(device1, device2):
    import_custom_nodes()
    with torch.inference_mode():
        facewarppipebuilder = NODE_CLASS_MAPPINGS["FaceWarpPipeBuilder"]()
        facewarppipebuilder_31 = facewarppipebuilder.load_models(
            detect_model_path="facedetect/scrfd_10g_bnkps_shape640x640.onnx",
            deca_dir="deca",
            gpu_choose=device1,
        )

        pulidmodelsloader = NODE_CLASS_MAPPINGS["PulidModelsLoader"]()
        pulidmodelsloader_34 = pulidmodelsloader.load_models(
            base_model_path="checkpoints/LEOSAM_SDXL_v5.0.safetensors",
            canny_model_dir="controlnet/Canny",
            unet_trt_path="trts/sdxl_unet_trtV11.engine",
            controlnet_trt_path="trts/sdxl_ctrlnet_trtV11.engine",
            pulid_model_dir="pulid",
            eva_clip_path="eva_clip/EVA02_CLIP_L_336_psz14_s6B.pt",
            insightface_dir="insightface",
            facexlib_dir="facexlib",
            gpu_choose=device2,
        )

        faceswappipebuilder = NODE_CLASS_MAPPINGS["FaceSwapPipeBuilder"]()
        faceswappipebuilder_51 = faceswappipebuilder.load_models(
            swap_own_model="faceswap/swapper_own.pth",
            arcface_model="faceswap/arcface_checkpoint.tar",
            facealign_config_dir="face_align",
            phase1_model="facealign/p1.pt",
            phase2_model="facealign/p2.pt",
            device=device1,
        )

        srblendbuildpipe = NODE_CLASS_MAPPINGS["SrBlendBuildPipe"]()
        srblendbuildpipe_58 = srblendbuildpipe.load_models(
            lut_path="lut", gpu_choose=device1, sr_type="RealESRGAN_X2", half=True
        )

        gpenpbuildpipeline = NODE_CLASS_MAPPINGS["GPENPBuildipeline"]()
        gpenpbuildpipeline_59 = gpenpbuildpipeline.load_model(
            model="GPEN-BFR-1024.pth",
            in_size=1024,
            channel_multiplier=2,
            narrow=1,
            alpha=1,
            device=device1,
        )

        preprocnewbuildpipe = NODE_CLASS_MAPPINGS["PreprocNewBuildPipe"]()
        preprocnewbuildpipe_67 = preprocnewbuildpipe.load_models(
            wd14_cgpath="wd14_tagger"
        )
        return {
            "facewarppipebuilder_31" : facewarppipebuilder_31,
            "pulidmodelsloader_34" : pulidmodelsloader_34,
            "faceswappipebuilder_51" : faceswappipebuilder_51,
            "srblendbuildpipe_58" : srblendbuildpipe_58,
            "gpenpbuildpipeline_59" : gpenpbuildpipeline_59,
            "preprocnewbuildpipe_67" : preprocnewbuildpipe_67,
        }

# portrait_own_inited_models = init_portrait_own_models()

def ai_portrait_own(portrait_own_inited_models, image_path, template_id, abs_path=True, is_ndarray=False):
    # import_custom_nodes()
    with torch.inference_mode():
        loadimage = LoadImage()
        loadimage_24 = loadimage.load_image(image=image_path, abs_path=abs_path, is_ndarray=is_ndarray)

        preprocnewgetconds = NODE_CLASS_MAPPINGS["PreprocNewGetConds"]()
        preprocnewsplitconds = NODE_CLASS_MAPPINGS["PreprocNewSplitConds"]()
        facewarpdetectfacesmethod = NODE_CLASS_MAPPINGS["FaceWarpDetectFacesMethod"]()
        facewarpgetfaces3dinfomethod = NODE_CLASS_MAPPINGS[
            "FaceWarpGetFaces3DinfoMethod"
        ]()
        facewarpwarp3dfaceimgmaskmethod = NODE_CLASS_MAPPINGS[
            "FaceWarpWarp3DfaceImgMaskMethod"
        ]()
        pulidinferclass = NODE_CLASS_MAPPINGS["PulidInferClass"]()
        facewarpwarp3dfaceimgmethod = NODE_CLASS_MAPPINGS[
            "FaceWarpWarp3DfaceImgMethod"
        ]()
        faceswapdetectpts = NODE_CLASS_MAPPINGS["FaceSwapDetectPts"]()
        faceswapmethod = NODE_CLASS_MAPPINGS["FaceSwapMethod"]()
        gpenprocess = NODE_CLASS_MAPPINGS["GPENProcess"]()
        srblendprocess = NODE_CLASS_MAPPINGS["SrBlendProcess"]()
        previewnumpy = NODE_CLASS_MAPPINGS["PreviewNumpy"]()

        # for q in range(1):
        facewarpdetectfacesimginput = NODE_CLASS_MAPPINGS[
            "FaceWarpDetectFacesImgInput"
        ]()
        facewarpdetectfacesimginput_77 = facewarpdetectfacesimginput.detect_faces(
            model=get_value_at_index(portrait_own_inited_models['facewarppipebuilder_31'], 0),
            image=get_value_at_index(loadimage_24, 0),
        )
        preprocnewgetconds_68 = preprocnewgetconds.prepare(
            template_id=template_id,
            is_front="False",
            model=get_value_at_index(portrait_own_inited_models['preprocnewbuildpipe_67'], 0),
            src_img=get_value_at_index(loadimage_24, 0),
            faces=get_value_at_index(facewarpdetectfacesimginput_77, 0),
        )

        preprocnewsplitconds_69 = preprocnewsplitconds.split(
            pipe_conditions=get_value_at_index(preprocnewgetconds_68, 0)
        )

        gpenprocess_79 = gpenprocess.enhance_face(
            aligned=False,
            model=get_value_at_index(portrait_own_inited_models['gpenpbuildpipeline_59'], 0),
            image=get_value_at_index(preprocnewsplitconds_69, 0),
        )

        facewarpdetectfacesmethod_33 = facewarpdetectfacesmethod.detect_faces(
            model=get_value_at_index(portrait_own_inited_models['facewarppipebuilder_31'], 0),
            image=get_value_at_index(gpenprocess_79, 0),
        )

        facewarpgetfaces3dinfomethod_32 = (
            facewarpgetfaces3dinfomethod.get_faces_3dinfo(
                model=get_value_at_index(portrait_own_inited_models['facewarppipebuilder_31'], 0),
                image=get_value_at_index(gpenprocess_79, 0),
                faces=get_value_at_index(facewarpdetectfacesmethod_33, 0),
            )
        )

        facewarpwarp3dfaceimgmaskmethod_36 = (
            facewarpwarp3dfaceimgmaskmethod.warp_3d_face(
                model=get_value_at_index(portrait_own_inited_models['facewarppipebuilder_31'], 0),
                user_dict=get_value_at_index(facewarpgetfaces3dinfomethod_32, 0),
                template_image=get_value_at_index(preprocnewsplitconds_69, 3),
                template_mask_img=get_value_at_index(preprocnewsplitconds_69, 11),
                template_canny_img=get_value_at_index(preprocnewsplitconds_69, 4),
            )
        )

        pulidinferclass_35 = pulidinferclass.pulid_infer(
            prompt=get_value_at_index(preprocnewsplitconds_69, 1),
            negative_prompt=get_value_at_index(preprocnewsplitconds_69, 2),
            strength=0.7000000000000001,
            num_steps=14,
            model=get_value_at_index(portrait_own_inited_models['pulidmodelsloader_34'], 0),
            template_image=get_value_at_index(
                facewarpwarp3dfaceimgmaskmethod_36, 0
            ),
            canny_control=get_value_at_index(facewarpwarp3dfaceimgmaskmethod_36, 2),
            user_image=get_value_at_index(gpenprocess_79, 0),
            mask=get_value_at_index(facewarpwarp3dfaceimgmaskmethod_36, 1),
        )

        facewarpwarp3dfaceimgmethod_70 = facewarpwarp3dfaceimgmethod.warp_3d_face(
            model=get_value_at_index(portrait_own_inited_models['facewarppipebuilder_31'], 0),
            user_dict=get_value_at_index(facewarpgetfaces3dinfomethod_32, 0),
            template_image=get_value_at_index(pulidinferclass_35, 0),
        )

        facewarpdetectfacesmethod_55 = facewarpdetectfacesmethod.detect_faces(
            model=get_value_at_index(portrait_own_inited_models['facewarppipebuilder_31'], 0),
            image=get_value_at_index(facewarpwarp3dfaceimgmethod_70, 0),
        )

        faceswapdetectpts_52 = faceswapdetectpts.detect_face_pts(
            ptstype="256",
            model=get_value_at_index(portrait_own_inited_models['faceswappipebuilder_51'], 0),
            src_image=get_value_at_index(facewarpwarp3dfaceimgmethod_70, 0),
            src_faces=get_value_at_index(facewarpdetectfacesmethod_55, 0),
        )

        faceswapdetectpts_56 = faceswapdetectpts.detect_face_pts(
            ptstype="5",
            model=get_value_at_index(portrait_own_inited_models['faceswappipebuilder_51'], 0),
            src_image=get_value_at_index(gpenprocess_79, 0),
            src_faces=get_value_at_index(facewarpdetectfacesmethod_33, 0),
        )

        faceswapdetectpts_54 = faceswapdetectpts.detect_face_pts(
            ptstype="5",
            model=get_value_at_index(portrait_own_inited_models['faceswappipebuilder_51'], 0),
            src_image=get_value_at_index(facewarpwarp3dfaceimgmethod_70, 0),
            src_faces=get_value_at_index(facewarpdetectfacesmethod_55, 0),
        )

        faceswapmethod_53 = faceswapmethod.swap_face(
            model=get_value_at_index(portrait_own_inited_models['faceswappipebuilder_51'], 0),
            src_image=get_value_at_index(gpenprocess_79, 0),
            two_stage_image=get_value_at_index(facewarpwarp3dfaceimgmethod_70, 0),
            source_5pts=get_value_at_index(faceswapdetectpts_56, 0),
            target_5pts=get_value_at_index(faceswapdetectpts_54, 0),
            target_256pts=get_value_at_index(faceswapdetectpts_52, 0),
        )

        gpenprocess_60 = gpenprocess.enhance_face(
            aligned=False,
            model=get_value_at_index(portrait_own_inited_models['gpenpbuildpipeline_59'], 0),
            image=get_value_at_index(faceswapmethod_53, 0),
        )

        _, final_image = srblendprocess.enhance_process(
            is_front="False",
            model=get_value_at_index(portrait_own_inited_models['srblendbuildpipe_58'], 0),
            src_img=get_value_at_index(gpenprocess_60, 0),
        )
        outimg = get_value_at_index(final_image, 0) * 255.
        return outimg.cpu().numpy().astype(np.uint8)

if __name__ == "__main__":
    pass

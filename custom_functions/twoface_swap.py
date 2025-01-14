import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


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

def init_twoface_swap_models():
    import_custom_nodes()
    with torch.inference_mode():
        faceswappipebuilder = NODE_CLASS_MAPPINGS["FaceSwapPipeBuilder"]()
        faceswappipebuilder_3 = faceswappipebuilder.load_models(
            swap_own_model="faceswap/swapper_own.pth",
            arcface_model="faceswap/arcface_checkpoint.tar",
            facealign_config_dir="face_align",
            phase1_model="facealign/p1.pt",
            phase2_model="facealign/p2.pt",
            device="cuda:0",
        )

        facewarppipebuilder = NODE_CLASS_MAPPINGS["FaceWarpPipeBuilder"]()
        facewarppipebuilder_23 = facewarppipebuilder.load_models(
            detect_model_path="facedetect/scrfd_10g_bnkps_shape640x640.onnx",
            deca_dir="deca",
            gpu_choose="cuda:0",
        )

        preproctwofacesswapbuildpipe = NODE_CLASS_MAPPINGS[
            "PreprocTwoFacesSwapBuildPipe"
        ]()
        preproctwofacesswapbuildpipe_27 = preproctwofacesswapbuildpipe.load_models()

        return {
            'faceswappipebuilder_3': faceswappipebuilder_3,
            'facewarppipebuilder_23': facewarppipebuilder_23,
            'preproctwofacesswapbuildpipe_27': preproctwofacesswapbuildpipe_27,
        }

def twoface_swap_process(twoface_swap_inited_models, image_left, image_right, template_id, abs_path=True, is_ndarray=False):
    import_custom_nodes()
    with torch.inference_mode():
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_12 = loadimage.load_image(image=image_left, abs_path=abs_path, is_ndarray=is_ndarray)
        loadimage_1 = loadimage.load_image(image=image_right, abs_path=abs_path, is_ndarray=is_ndarray)        

        facewarpdetectfacesimginput = NODE_CLASS_MAPPINGS[
            "FaceWarpDetectFacesImgInput"
        ]()
        preproctwofacesswapgetconds = NODE_CLASS_MAPPINGS[
            "PreprocTwoFacesSwapGetConds"
        ]()
        facewarpdetectfacesmethod = NODE_CLASS_MAPPINGS["FaceWarpDetectFacesMethod"]()
        facewarpgetfaces3dinfomethod = NODE_CLASS_MAPPINGS[
            "FaceWarpGetFaces3DinfoMethod"
        ]()
        twofaceswarpimgonly = NODE_CLASS_MAPPINGS["TwoFacesWarpImgOnly"]()
        faceswapdetectpts = NODE_CLASS_MAPPINGS["FaceSwapDetectPts"]()
        twofacesdetectpts = NODE_CLASS_MAPPINGS["TwoFacesDetectPts"]()
        twofacesfaceswap = NODE_CLASS_MAPPINGS["TwoFacesFaceSwap"]()
        previewnumpy = NODE_CLASS_MAPPINGS["PreviewNumpy"]()

        # for q in range(1):
        facewarpdetectfacesimginput_38 = facewarpdetectfacesimginput.detect_faces(
            model=get_value_at_index(twoface_swap_inited_models['facewarppipebuilder_23'], 0),
            image=get_value_at_index(loadimage_12, 0),
        )

        facewarpdetectfacesimginput_39 = facewarpdetectfacesimginput.detect_faces(
            model=get_value_at_index(twoface_swap_inited_models['facewarppipebuilder_23'], 0),
            image=get_value_at_index(loadimage_1, 0),
        )

        preproctwofacesswapgetconds_28 = preproctwofacesswapgetconds.prepare(
            template_id=template_id,
            is_front="False",
            model=get_value_at_index(twoface_swap_inited_models['preproctwofacesswapbuildpipe_27'], 0),
            src_img1=get_value_at_index(loadimage_12, 0),
            src_img2=get_value_at_index(loadimage_1, 0),
            faces1=get_value_at_index(facewarpdetectfacesimginput_38, 0),
            faces2=get_value_at_index(facewarpdetectfacesimginput_39, 0),
        )

        facewarpdetectfacesmethod_24 = facewarpdetectfacesmethod.detect_faces(
            model=get_value_at_index(twoface_swap_inited_models['facewarppipebuilder_23'], 0),
            image=get_value_at_index(preproctwofacesswapgetconds_28, 0),
        )

        facewarpgetfaces3dinfomethod_30 = (
            facewarpgetfaces3dinfomethod.get_faces_3dinfo(
                model=get_value_at_index(twoface_swap_inited_models['facewarppipebuilder_23'], 0),
                image=get_value_at_index(preproctwofacesswapgetconds_28, 0),
                faces=get_value_at_index(facewarpdetectfacesmethod_24, 0),
            )
        )

        facewarpdetectfacesmethod_25 = facewarpdetectfacesmethod.detect_faces(
            model=get_value_at_index(twoface_swap_inited_models['facewarppipebuilder_23'], 0),
            image=get_value_at_index(preproctwofacesswapgetconds_28, 1),
        )

        facewarpgetfaces3dinfomethod_31 = (
            facewarpgetfaces3dinfomethod.get_faces_3dinfo(
                model=get_value_at_index(twoface_swap_inited_models['facewarppipebuilder_23'], 0),
                image=get_value_at_index(preproctwofacesswapgetconds_28, 1),
                faces=get_value_at_index(facewarpdetectfacesmethod_25, 0),
            )
        )

        twofaceswarpimgonly_29 = twofaceswarpimgonly.warp_3d_two_faces(
            model=get_value_at_index(twoface_swap_inited_models['facewarppipebuilder_23'], 0),
            user_dict1=get_value_at_index(facewarpgetfaces3dinfomethod_30, 0),
            user_dict2=get_value_at_index(facewarpgetfaces3dinfomethod_31, 0),
            template_image=get_value_at_index(preproctwofacesswapgetconds_28, 2),
        )

        facewarpdetectfacesmethod_4 = facewarpdetectfacesmethod.detect_faces(
            model=get_value_at_index(twoface_swap_inited_models['facewarppipebuilder_23'], 0),
            image=get_value_at_index(twofaceswarpimgonly_29, 0),
        )

        faceswapdetectpts_5 = faceswapdetectpts.detect_face_pts(
            ptstype="5",
            model=get_value_at_index(twoface_swap_inited_models['faceswappipebuilder_3'], 0),  
            src_image=get_value_at_index(preproctwofacesswapgetconds_28, 0),
            src_faces=get_value_at_index(facewarpdetectfacesmethod_24, 0),
        )

        faceswapdetectpts_17 = faceswapdetectpts.detect_face_pts(
            ptstype="5",
            model=get_value_at_index(twoface_swap_inited_models['faceswappipebuilder_3'], 0),
            src_image=get_value_at_index(preproctwofacesswapgetconds_28, 1),
            src_faces=get_value_at_index(facewarpdetectfacesmethod_25, 0),
        )

        twofacesdetectpts_15 = twofacesdetectpts.detect_2faces_pts(
            ptstype="5",
            model=get_value_at_index(twoface_swap_inited_models['faceswappipebuilder_3'], 0),
            src_image=get_value_at_index(twofaceswarpimgonly_29, 0),
            src_faces=get_value_at_index(facewarpdetectfacesmethod_4, 0),
        )

        twofacesdetectpts_16 = twofacesdetectpts.detect_2faces_pts(
            ptstype="256",
            model=get_value_at_index(twoface_swap_inited_models['faceswappipebuilder_3'], 0),
            src_image=get_value_at_index(twofaceswarpimgonly_29, 0),
            src_faces=get_value_at_index(facewarpdetectfacesmethod_4, 0),
        )

        twofacesfaceswap_14 = twofacesfaceswap.swap_two_faces(
            model=get_value_at_index(twoface_swap_inited_models['faceswappipebuilder_3'], 0),
            src_image1=get_value_at_index(preproctwofacesswapgetconds_28, 0),
            src_image2=get_value_at_index(preproctwofacesswapgetconds_28, 1),
            two_stage_image=get_value_at_index(twofaceswarpimgonly_29, 0),
            source_5pts1=get_value_at_index(faceswapdetectpts_5, 0),
            source_5pts2=get_value_at_index(faceswapdetectpts_17, 0),
            target_5pts1=get_value_at_index(twofacesdetectpts_15, 0),
            target_5pts2=get_value_at_index(twofacesdetectpts_15, 1),
            target_256pts1=get_value_at_index(twofacesdetectpts_16, 0),
            target_256pts2=get_value_at_index(twofacesdetectpts_16, 1),
        )

        outimg = get_value_at_index(twofacesfaceswap_14, 0)
        
        return outimg

import numpy as np
from PIL import Image
import cv2
if __name__ == "__main__":
    img_path_left = '/data/projs/WebServer/distributed-server-node/submodules/VisualForge/ComfyUI/input/5.jpg'
    img_path_right = '/data/projs/WebServer/distributed-server-node/submodules/VisualForge/ComfyUI/input/zhuyu_selfie.jpg'
    image_left = np.array(Image.open(img_path_left))[..., :3]
    image_right = np.array(Image.open(img_path_right))[..., :3]
    twoface_swap_inited_models = init_twoface_swap_models()
    for i in range(10):
        final_img = twoface_swap_process(twoface_swap_inited_models, image_left, image_right, "couples/couple1", is_ndarray=True)
    
    cv2.imwrite('fuckaigc.png', final_img)
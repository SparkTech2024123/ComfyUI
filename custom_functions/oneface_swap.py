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

def init_oneface_swap_models():
    import_custom_nodes()
    with torch.inference_mode():
        facewarppipebuilder = NODE_CLASS_MAPPINGS["FaceWarpPipeBuilder"]()
        facewarppipebuilder_11 = facewarppipebuilder.load_models(
            detect_model_path="facedetect/scrfd_10g_bnkps_shape640x640.onnx",
            deca_dir="deca",
            gpu_choose="cuda:0",
        )

        faceswappipebuilder = NODE_CLASS_MAPPINGS["FaceSwapPipeBuilder"]()
        faceswappipebuilder_17 = faceswappipebuilder.load_models(
            swap_own_model="faceswap/swapper_own.pth",
            arcface_model="faceswap/arcface_checkpoint.tar",
            facealign_config_dir="face_align",
            phase1_model="facealign/p1.pt",
            phase2_model="facealign/p2.pt",
            device="cuda:0",
        )

        # srblendbuildpipe = NODE_CLASS_MAPPINGS["SrBlendBuildPipe"]()
        # srblendbuildpipe_23 = srblendbuildpipe.load_models(
        #     lut_path="lut", gpu_choose="cuda:0", sr_type="RealESRGAN_X2", half=True
        # )

        # gpenpbuildipeline = NODE_CLASS_MAPPINGS["GPENPBuildipeline"]()
        # gpenpbuildipeline_24 = gpenpbuildipeline.load_model(
        #     model="GPEN-BFR-1024.pth",
        #     in_size=1024,
        #     channel_multiplier=2,
        #     narrow=1,
        #     alpha=1,
        #     device="cuda:0",
        # )

        preprocswapbuildpipe = NODE_CLASS_MAPPINGS["PreprocSwapBuildPipe"]()
        preprocswapbuildpipe_34 = preprocswapbuildpipe.load_models()

        return {
            'facewarppipebuilder_11': facewarppipebuilder_11,
            'faceswappipebuilder_17': faceswappipebuilder_17,
            # 'srblendbuildpipe_23': srblendbuildpipe_23,
            # 'gpenpbuildpipeline_24': gpenpbuildipeline_24,
            'preprocswapbuildpipe_34': preprocswapbuildpipe_34,
        }

def oneface_swap_process(oneface_swap_inited_models, image_path, template_id, abs_path=True, is_ndarray=False):
    # import_custom_nodes()
    with torch.inference_mode():
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_10 = loadimage.load_image(image=image_path, abs_path=abs_path, is_ndarray=is_ndarray)

        facewarpdetectfacesimginput = NODE_CLASS_MAPPINGS[
            "FaceWarpDetectFacesImgInput"
        ]()
        preprocswapgetconds = NODE_CLASS_MAPPINGS["PreprocSwapGetConds"]()
        gpenprocess = NODE_CLASS_MAPPINGS["GPENProcess"]()
        facewarpdetectfacesmethod = NODE_CLASS_MAPPINGS["FaceWarpDetectFacesMethod"]()
        facewarpgetfaces3dinfomethod = NODE_CLASS_MAPPINGS[
            "FaceWarpGetFaces3DinfoMethod"
        ]()
        facewarpwarp3dfaceimgmethod = NODE_CLASS_MAPPINGS[
            "FaceWarpWarp3DfaceImgMethod"
        ]()
        faceswapdetectpts = NODE_CLASS_MAPPINGS["FaceSwapDetectPts"]()
        faceswapmethod = NODE_CLASS_MAPPINGS["FaceSwapMethod"]()
        srblendprocess = NODE_CLASS_MAPPINGS["SrBlendProcess"]()
        # previewnumpy = NODE_CLASS_MAPPINGS["PreviewNumpy"]()

        # for q in range(1):
        facewarpdetectfacesimginput_37 = facewarpdetectfacesimginput.detect_faces(
            model=get_value_at_index(oneface_swap_inited_models['facewarppipebuilder_11'], 0),
            image=get_value_at_index(loadimage_10, 0),
        )

        preprocswapgetconds_35 = preprocswapgetconds.prepare(
            template_id="christmas/female-10",
            model=get_value_at_index(oneface_swap_inited_models['preprocswapbuildpipe_34'], 0),
            src_img=get_value_at_index(loadimage_10, 0),
            faces=get_value_at_index(facewarpdetectfacesimginput_37, 0),
        )

        # gpenprocess_39 = gpenprocess.enhance_face(
        #     aligned=False,
        #     model=get_value_at_index(oneface_swap_inited_models['gpenpbuildpipeline_24'], 0),
        #     image=get_value_at_index(preprocswapgetconds_35, 0),
        # )

        facewarpdetectfacesmethod_13 = facewarpdetectfacesmethod.detect_faces(
            model=get_value_at_index(oneface_swap_inited_models['facewarppipebuilder_11'], 0),
            image=get_value_at_index(preprocswapgetconds_35, 0),
        )

        facewarpgetfaces3dinfomethod_12 = (
            facewarpgetfaces3dinfomethod.get_faces_3dinfo(
                model=get_value_at_index(oneface_swap_inited_models['facewarppipebuilder_11'], 0),
                image=get_value_at_index(preprocswapgetconds_35, 0),
                faces=get_value_at_index(facewarpdetectfacesmethod_13, 0),
            )
        )

        facewarpwarp3dfaceimgmethod_42 = facewarpwarp3dfaceimgmethod.warp_3d_face(
            model=get_value_at_index(oneface_swap_inited_models['facewarppipebuilder_11'], 0),
            user_dict=get_value_at_index(facewarpgetfaces3dinfomethod_12, 0),
            template_image=get_value_at_index(preprocswapgetconds_35, 1),
        )

        facewarpwarp3dfaceimgmethod_36 = facewarpwarp3dfaceimgmethod.warp_3d_face(
            model=get_value_at_index(oneface_swap_inited_models['facewarppipebuilder_11'], 0),
            user_dict=get_value_at_index(facewarpgetfaces3dinfomethod_12, 0),
            template_image=get_value_at_index(facewarpwarp3dfaceimgmethod_42, 0),
        )

        facewarpdetectfacesmethod_21 = facewarpdetectfacesmethod.detect_faces(
            model=get_value_at_index(oneface_swap_inited_models['facewarppipebuilder_11'], 0),
            image=get_value_at_index(facewarpwarp3dfaceimgmethod_36, 0),
        )

        faceswapdetectpts_18 = faceswapdetectpts.detect_face_pts(
            ptstype="256",
            model=get_value_at_index(oneface_swap_inited_models['faceswappipebuilder_17'], 0),
            src_image=get_value_at_index(facewarpwarp3dfaceimgmethod_36, 0),
            src_faces=get_value_at_index(facewarpdetectfacesmethod_21, 0),
        )

        faceswapdetectpts_22 = faceswapdetectpts.detect_face_pts(
            ptstype="5",
            model=get_value_at_index(oneface_swap_inited_models['faceswappipebuilder_17'], 0),
            src_image=get_value_at_index(preprocswapgetconds_35, 0),
            src_faces=get_value_at_index(facewarpdetectfacesmethod_13, 0),
        )

        faceswapdetectpts_20 = faceswapdetectpts.detect_face_pts(
            ptstype="5",
            model=get_value_at_index(oneface_swap_inited_models['faceswappipebuilder_17'], 0),
            src_image=get_value_at_index(facewarpwarp3dfaceimgmethod_36, 0),
            src_faces=get_value_at_index(facewarpdetectfacesmethod_21, 0),
        )

        faceswapmethod_19 = faceswapmethod.swap_face(
            model=get_value_at_index(oneface_swap_inited_models['faceswappipebuilder_17'], 0),
            src_image=get_value_at_index(preprocswapgetconds_35, 0),
            two_stage_image=get_value_at_index(facewarpwarp3dfaceimgmethod_36, 0),
            source_5pts=get_value_at_index(faceswapdetectpts_22, 0),
            target_5pts=get_value_at_index(faceswapdetectpts_20, 0),
            target_256pts=get_value_at_index(faceswapdetectpts_18, 0),
        )

        # gpenprocess_25 = gpenprocess.enhance_face(
        #     aligned=False,
        #     model=get_value_at_index(oneface_swap_inited_models['gpenpbuildpipeline_24'], 0),
        #     image=get_value_at_index(faceswapmethod_19, 0),
        # )

        # _, final_image = srblendprocess.enhance_process(
        #     is_front="False",
        #     model=get_value_at_index(oneface_swap_inited_models['srblendbuildpipe_23'], 0),
        #     src_img=get_value_at_index(gpenprocess_25, 0),
        # )
        return get_value_at_index(faceswapmethod_19, 0)
        # outimg = get_value_at_index(faceswapmethod_19, 0) * 255.
        # return outimg.cpu().numpy().astype(np.uint8)

import numpy as np
from PIL import Image
import cv2
if __name__ == "__main__":
    img_path = '/data/projs/WebServer/distributed-server-node/submodules/VisualForge/ComfyUI/input/5.jpg'
    image = np.array(Image.open(img_path))[..., :3]
    oneface_swap_inited_models = init_oneface_swap_models()
    for i in range(10):
        final_img = oneface_swap_process(oneface_swap_inited_models, image, "christmas/female-10", is_ndarray=True)
    
    cv2.imwrite('fuckaigc.png', final_img)

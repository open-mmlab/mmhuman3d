import json
import os

from mmhuman3d.core.cameras.camera_parameters import CameraParameter
from mmhuman3d.utils.path_utils import check_path_suffix
from .renderer.vedo_render import VedoRenderer


def visualize_chessboard_kinects_rgb(chessboard_path: str,
                                     interactive: bool = True,
                                     show: bool = True):
    """Visualize all the RGB cameras in a chessboard file.

    Args:
        chessboard_path (str):
            Path to the chessboard file.
        interactive (bool, optional):
            Pause and interact with window (True) or
            continue execution (False).
            Defaults to True.
        show (bool, optional):
            Whether to show in a window.
            Defaults to True.
    """
    # Load camera parameter from a json file
    camera_para_json_dict = json.load(open(chessboard_path))
    camera_para_dict = {}
    for camera_id in camera_para_json_dict.keys():
        try:
            camera_id_int = int(camera_id)
            # if camera_id is an instance of int
            # and it can be divided by 2, it's an rgb camera
            if camera_id_int % 2 == 0:
                pass
            else:
                continue
        except ValueError:
            continue
        temp_camera_parameter = CameraParameter(name=camera_id)
        temp_camera_parameter.load_from_chessboard(
            camera_para_json_dict[camera_id], camera_id)
        camera_para_dict[camera_id] = temp_camera_parameter
    camera_vedo_renderer = VedoRenderer()
    camera_vedo_renderer.set_y_reverse()
    for camera_id in camera_para_dict.keys():
        camera_vedo_renderer.add_camera(camera_para_dict[camera_id])
    if show:
        camera_vedo_renderer.show(with_axis=False, interactive=interactive)


def visualize_dumped_camera_parameter(dumped_dir: str,
                                      interactive: bool = True,
                                      show: bool = True):
    """Visualize all cameras dumped in a directory.

    Args:
        dumped_dir (str):
            Path to the directory.
        interactive (bool, optional):
            Pause and interact with window (True) or
            continue execution (False).
            Defaults to True.
        show (bool, optional):
            Whether to show in a window.
            Defaults to True.
    """
    file_list = os.listdir(dumped_dir)
    camera_para_list = []
    for file_name in file_list:
        file_path = os.path.join(dumped_dir, file_name)
        if not check_path_suffix(file_path, ['.json']):
            continue
        else:
            cam_para = CameraParameter()
            cam_para.load(file_path)
            camera_para_list.append(cam_para)
    camera_vedo_renderer = VedoRenderer()
    camera_vedo_renderer.set_y_reverse()
    for camera_para in camera_para_list:
        camera_vedo_renderer.add_camera(camera_para)
    if show:
        camera_vedo_renderer.show(with_axis=False, interactive=interactive)

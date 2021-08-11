import json

from mmhuman3d.core.visualization.renderer.vedo_render import VedoRenderer
from mmhuman3d.data.data_structures.smc_reader import SMCReader
from mmhuman3d.utils.camera import CameraParameter


def visualize_chessboard_kinects_rgb(chessboard_path, interactive=True):
    # Load camera parameter from a json file
    camera_para_json_dict = json.load(open(chessboard_path))
    camera_para_dict = {}
    for camera_id in camera_para_json_dict.keys():
        try:
            camera_id_int = int(camera_id)
            # if camera_id can be parsed into int
            # it is an rgb camera
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
    camera_vedo_renderer.show(with_axis=False, interactive=interactive)


def visualize_smc_kinects_rgb(smc_path, interactive=True):
    sense_mocap = SMCReader(smc_path)
    camera_number = sense_mocap.get_num_kinect()
    camera_para_dict = {}
    for kinect_id in range(camera_number):
        temp_camera_parameter = CameraParameter(name=kinect_id)
        temp_camera_parameter.load_from_smc(sense_mocap, kinect_id)
        camera_para_dict[kinect_id] = temp_camera_parameter
    camera_vedo_renderer = VedoRenderer()
    camera_vedo_renderer.set_y_reverse()
    for kinect_id in camera_para_dict.keys():
        camera_vedo_renderer.add_camera(camera_para_dict[kinect_id])
    camera_vedo_renderer.show(with_axis=False, interactive=interactive)

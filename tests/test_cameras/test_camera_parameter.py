import json
import os

import numpy as np
import pytest

from mmhuman3d.core.cameras.camera_parameter import CameraParameter
from mmhuman3d.core.cameras.cameras import WeakPerspectiveCameras
from mmhuman3d.utils.path_utils import Existence, check_path_existence

chessboard_path = 'tests/data/camera/' +\
    'calibration_chessboard_05_28_18_05_19.json'
dump_path = 'tests/data/camera/' +\
    'camera_parameter_dump.json'


def test_set_from_mat():
    empty_param = CameraParameter(name='test_set')
    mat_3x3 = np.eye(3)
    mat_4x4 = np.eye(4)
    vec_3 = np.zeros(shape=[3])
    empty_param.set_K_R_T(K_mat=mat_3x3, R_mat=mat_3x3, T_vec=vec_3)
    empty_param.set_K_R_T(
        K_mat=mat_3x3, R_mat=mat_3x3, T_vec=vec_3, inverse_extrinsic=True)
    with pytest.raises(AssertionError):
        empty_param.set_K_R_T(K_mat=mat_4x4, R_mat=mat_3x3, T_vec=vec_3)
    with pytest.raises(AssertionError):
        empty_param.set_K_R_T(K_mat=mat_3x3, R_mat=mat_4x4, T_vec=vec_3)
    assert len(empty_param.get_value('translation')) == 3


def test_load_chessboard():
    empty_param = CameraParameter(name='test_kinect')
    chessboard_dict = json.load(open(chessboard_path))
    empty_param.load_from_chessboard(chessboard_dict[str(0)], 'chessboard_0')
    assert len(empty_param.get_value('translation')) == 3


def test_dump_json():
    cam_param = CameraParameter(name='test_kinect')
    chessboard_dict = json.load(open(chessboard_path))
    cam_param.load_from_chessboard(chessboard_dict[str(0)], 'test_kinect')
    if check_path_existence(dump_path, 'file') == Existence.FileExist:
        os.remove(dump_path)
    cam_param.dump(dump_path)
    assert check_path_existence(dump_path, 'file') == Existence.FileExist


def test_load_json():
    cam_param = CameraParameter(name='test_kinect')
    cam_param.load(dump_path)
    assert len(cam_param.get_value('translation')) == 3


def test_type():
    cam_param = CameraParameter(name='src_cam')
    # wrong distortion value type
    with pytest.raises(TypeError):
        cam_param.set_value('k1', '1.0')
    with pytest.raises(TypeError):
        cam_param.set_value('k1', np.ones(shape=(3))[0])
    cam_param.set_value('k1', 1.0)
    cam_param.reset_distort()

    mat_3x3 = np.eye(3)
    # wrong mat type
    with pytest.raises(TypeError):
        cam_param.set_mat_list('in_mat', mat_3x3)
    with pytest.raises(TypeError):
        cam_param.set_value('in_mat', mat_3x3)
    cam_param.set_mat_np('in_mat', mat_3x3)
    mat_3x3_list = mat_3x3.tolist()
    with pytest.raises(TypeError):
        cam_param.set_mat_np('rotation_mat', mat_3x3_list)
    cam_param.set_mat_list('rotation_mat', mat_3x3_list)
    dumped_str = cam_param.to_string()
    assert isinstance(dumped_str, str)
    assert len(dumped_str) > 0


def test_vibe_camera():
    cam_param = CameraParameter(name='src_cam')
    dumped_dict = json.load(open(dump_path))
    cam_param.load_from_dict(dumped_dict)
    vibe_cam_arg = cam_param.get_vibe_dict()
    assert len(vibe_cam_arg) > 0
    vibe_cam = WeakPerspectiveCameras(**vibe_cam_arg)
    empty_param = CameraParameter(name='dst_cam')
    empty_param.load_from_vibe(vibe_cam, name='dst_cam', batch_index=0)
    src_mat = cam_param.get_mat_np('in_mat')
    dst_mat = empty_param.get_mat_np('in_mat')
    k_diff = (src_mat - dst_mat).sum()
    assert k_diff == 0
    src_mat = cam_param.get_mat_np('rotation_mat')
    dst_mat = empty_param.get_mat_np('rotation_mat')
    r_diff = (src_mat - dst_mat).sum()
    assert r_diff == 0
    src_tran = np.asarray(cam_param.get_value('translation'))
    dst_tran = np.asarray(empty_param.get_value('translation'))
    t_diff = (src_tran - dst_tran).sum()
    assert t_diff == 0


def test_misc():
    empty_param = CameraParameter(name='test_misc')
    mat_3x3 = np.eye(3)
    vec_3 = np.zeros(shape=[3])
    empty_param.set_K_R_T(K_mat=mat_3x3, R_mat=mat_3x3, T_vec=vec_3)
    empty_param.set_value('k1', 1.0)
    empty_param.reset_distort()
    distort_vec = empty_param.get_opencv_distort_mat()
    assert float(np.sum(np.abs(distort_vec))) == 0
    # set wrong key
    with pytest.raises(KeyError):
        empty_param.set_mat_np('intrinsic_mat', mat_3x3)
    with pytest.raises(KeyError):
        empty_param.set_mat_list('intrinsic_mat', mat_3x3.tolist())
    with pytest.raises(KeyError):
        empty_param.set_value('distortion_k1', 1.0)
    # correct key
    empty_param.set_mat_list('in_mat', mat_3x3.tolist())
    empty_param.set_value('k1', 0.0)
    # get wrong key
    with pytest.raises(KeyError):
        empty_param.get_mat_np('intrinsic_mat')
    with pytest.raises(KeyError):
        empty_param.get_value('distortion_k1')
    dumped_str = empty_param.to_string()
    assert isinstance(dumped_str, str)

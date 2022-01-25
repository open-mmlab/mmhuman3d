import json
import os

import numpy as np
import pytest
import torch

from mmhuman3d.core.cameras.camera_parameters import CameraParameter
from mmhuman3d.data.data_structures.smc_reader import SMCReader
from mmhuman3d.utils.path_utils import Existence, check_path_existence

chessboard_path = 'tests/data/camera/' +\
    'calibration_chessboard_05_28_18_05_19.json'
dump_path = 'tests/data/camera/' +\
    'camera_parameter_dump.json'

smc_path = 'tests/data/dataset_sample/humman/p000003_a000014_tiny.smc'


def test_set_from_mat():
    empty_param = CameraParameter(name='test_set')
    mat_3x3 = np.eye(3)
    mat_4x4 = np.eye(4)
    vec_3 = np.zeros(shape=[3])
    empty_param.set_KRT(K_mat=mat_3x3, R_mat=mat_3x3, T_vec=vec_3)
    empty_param.set_KRT(
        K_mat=mat_3x3, R_mat=mat_3x3, T_vec=vec_3, inverse_extrinsic=True)
    with pytest.raises(AssertionError):
        empty_param.set_KRT(K_mat=mat_4x4, R_mat=mat_3x3, T_vec=vec_3)
    with pytest.raises(AssertionError):
        empty_param.set_KRT(K_mat=mat_3x3, R_mat=mat_4x4, T_vec=vec_3)
    assert len(empty_param.get_value('translation')) == 3


def test_load_chessboard():
    empty_param = CameraParameter(name='test_kinect')
    chessboard_dict = json.load(open(chessboard_path))
    empty_param.load_from_chessboard(chessboard_dict[str(0)], 'chessboard_0')
    assert len(empty_param.get_value('translation')) == 3


def test_load_from_smc():
    smc = SMCReader(smc_path)
    kinect_param = CameraParameter(name='test_kinect')
    kinect_param.load_kinect_from_smc(smc_reader=smc, kinect_id=0)
    assert len(kinect_param.get_value('translation')) == 3
    iphone_param = CameraParameter(name='test_iphone')
    iphone_param.load_iphone_from_smc(smc_reader=smc, iphone_id=0)
    assert len(kinect_param.get_value('translation')) == 3


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
    cam_param = CameraParameter(name='src_cam', H=720, W=1280)
    # wrong distortion value type (expect float)
    with pytest.raises(TypeError):
        cam_param.set_value('k1', '1.0')
    with pytest.raises(TypeError):
        cam_param.set_value('k1', np.ones(shape=(3, ), dtype=np.int8)[0])
    cam_param.set_value('k1', 1.0)
    cam_param.set_value('k1', np.ones(shape=(3, ), dtype=np.float16)[0])
    cam_param.set_value('k1', torch.ones(size=(3, ), dtype=torch.float16)[0])
    cam_param.reset_distort()

    # wrong height value type (expect float)
    with pytest.raises(TypeError):
        cam_param.set_value('H', '1080')
    with pytest.raises(TypeError):
        cam_param.set_value('H', 1080.0)
    with pytest.raises(TypeError):
        cam_param.set_value('H', np.ones(shape=(3, ), dtype=np.float32)[0])
    with pytest.raises(TypeError):
        cam_param.set_value('H', np.ones(shape=(3, ), dtype=np.int64)[0:1])
    with pytest.raises(TypeError):
        cam_param.set_value('H',
                            torch.ones(size=(3, ), dtype=torch.float32)[0])
    with pytest.raises(TypeError):
        cam_param.set_value('H',
                            torch.ones(size=(3, ), dtype=torch.int32)[0:1])
    cam_param.set_value('H', np.ones(shape=(3, ), dtype=np.uint8)[0])
    cam_param.set_value('H', torch.ones(size=(3, ), dtype=torch.int64)[0])
    cam_param.set_value('H', 720)

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


def test_misc():
    empty_param = CameraParameter(name='test_misc')
    mat_3x3 = np.eye(3)
    vec_3 = np.zeros(shape=[3])
    empty_param.set_KRT(K_mat=mat_3x3, R_mat=mat_3x3, T_vec=vec_3)
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
    # get K R T
    KRT_list = empty_param.get_KRT()
    assert len(KRT_list) == 3
    assert KRT_list[0].shape == (3, 3)
    KRT_list = empty_param.get_KRT(k_dim=4)
    assert KRT_list[0].shape == (4, 4)
    with pytest.raises(ValueError):
        KRT_list = empty_param.get_KRT(k_dim=5)

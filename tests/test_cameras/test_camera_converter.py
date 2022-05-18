import random

import numpy as np
import pytest
import torch

from mmhuman3d.core.cameras.cameras import FoVPerspectiveCameras
from mmhuman3d.core.conventions.cameras.convert_convention import (
    CAMERA_CONVENTIONS,
    convert_camera_matrix,
    convert_K_3x3_to_4x4,
    convert_K_4x4_to_3x3,
    convert_ndc_to_screen,
    convert_screen_to_ndc,
    convert_world_view,
)
from mmhuman3d.core.conventions.cameras.convert_projection import (
    convert_perspective_to_weakperspective,
    convert_weakperspective_to_perspective,
)
from mmhuman3d.utils.camera_utils import convert_smpl_from_opencv_calibration
from mmhuman3d.utils.transforms import ee_to_rotmat

model_path = 'data/body_models'


def check_isclose(K,
                  R,
                  T,
                  src,
                  dst,
                  is_perspective=True,
                  in_ndc_src=True,
                  in_ndc_dst=True,
                  resolution_src=None,
                  resolution_dst=None,
                  use_numpy=True,
                  eps=1e-2):
    K1, R1, T1 = convert_camera_matrix(
        K=K,
        R=R,
        T=T,
        is_perspective=is_perspective,
        convention_src=src,
        convention_dst=dst,
        in_ndc_src=in_ndc_src,
        in_ndc_dst=in_ndc_dst,
        resolution_src=resolution_src,
        resolution_dst=resolution_dst)
    K2, R2, T2 = convert_camera_matrix(
        K=K1,
        R=R1,
        T=T1,
        is_perspective=is_perspective,
        convention_src=dst,
        convention_dst=src,
        in_ndc_src=in_ndc_dst,
        in_ndc_dst=in_ndc_src,
        resolution_src=resolution_dst,
        resolution_dst=resolution_src)
    if use_numpy:
        assert np.isclose(K.all(), K2.all(), rtol=0, atol=eps)
        assert np.isclose(R.all(), R2.all(), rtol=0, atol=eps)
        assert np.isclose(T.all(), T2.all(), rtol=0, atol=eps)
    else:
        assert torch.isclose(K, K2, rtol=0, atol=eps).all()
        assert torch.isclose(R, R2, rtol=0, atol=eps).all()
        assert torch.isclose(T, T2, rtol=0, atol=eps).all()


def test_convert_cameras():
    for src in CAMERA_CONVENTIONS:
        for dst in CAMERA_CONVENTIONS:
            R = ee_to_rotmat(
                np.random.uniform(low=-np.pi, high=np.pi,
                                  size=(1, 3))).reshape(1, 3, 3)
            T = np.random.uniform(low=-10, high=10, size=(1, 3))
            fx = random.uniform(1 / 4, 4)
            fy = random.uniform(1 / 4, 4)
            px = random.uniform(-1, 1)
            py = random.uniform(-1, 1)
            K_ = np.eye(3, 3)[None]
            K_[:, 0, 0] = fx
            K_[:, 1, 1] = fy
            K_[:, 0, 2] = px
            K_[:, 1, 2] = py

            for is_perspective in [True, False]:
                K = convert_K_3x3_to_4x4(K_, is_perspective=is_perspective)

                check_isclose(
                    K=torch.Tensor(K),
                    R=torch.Tensor(R),
                    T=torch.Tensor(T),
                    src=src,
                    dst=dst,
                    use_numpy=False)

                with pytest.raises(TypeError):
                    check_isclose(
                        K=torch.Tensor(K),
                        R=R,
                        T=torch.Tensor(T),
                        src=src,
                        dst=dst,
                        use_numpy=False)

                check_isclose(
                    K=K,
                    R=R,
                    T=T,
                    src=src,
                    dst=dst,
                    in_ndc_src=True,
                    in_ndc_dst=True,
                )
                check_isclose(
                    K=K,
                    R=R,
                    T=T,
                    src=src,
                    dst=dst,
                    in_ndc_src=True,
                    in_ndc_dst=False,
                    resolution_dst=(1080, 1920),
                )

                check_isclose(
                    K=K,
                    R=R,
                    T=T,
                    src=src,
                    dst=dst,
                    in_ndc_src=False,
                    in_ndc_dst=True,
                    resolution_src=(1080, 1920),
                    eps=1e-1)

                check_isclose(
                    K=K,
                    R=R,
                    T=T,
                    src=src,
                    dst=dst,
                    in_ndc_src=False,
                    in_ndc_dst=False,
                    resolution_src=(1080, 1920),
                    resolution_dst=(1080, 1920),
                    eps=1e-1)


def test_convert_K():
    opencv_K = np.array([[1.7, 0., 1.0], [0., 1.7, 2.0], [0., 0., 1.0]])
    assert np.isclose(
        opencv_K.all(),
        convert_K_4x4_to_3x3(convert_K_3x3_to_4x4(opencv_K)).all(),
        rtol=0,
        atol=1e-2)
    assert np.isclose(
        opencv_K.all(),
        convert_K_4x4_to_3x3(
            convert_K_3x3_to_4x4(opencv_K, 'orthograophics'),
            'orthograophics').all(),
        rtol=0,
        atol=1e-2)

    opencv_K = torch.Tensor(opencv_K)
    assert torch.isclose(
        opencv_K,
        convert_K_4x4_to_3x3(convert_K_3x3_to_4x4(opencv_K)),
        rtol=0,
        atol=1e-2).all()
    assert torch.isclose(
        opencv_K,
        convert_K_4x4_to_3x3(
            convert_K_3x3_to_4x4(opencv_K, 'orthograophics'),
            'orthograophics'),
        rtol=0,
        atol=1e-2).all()


def test_convert_ndc_screen():
    eps = 1e-2
    fx = random.uniform(1 / 4, 4)
    fy = random.uniform(1 / 4, 4)
    px = random.uniform(-1, 1)
    py = random.uniform(-1, 1)
    K_ = np.eye(3, 3)[None]
    K_[:, 0, 0] = fx
    K_[:, 1, 1] = fy
    K_[:, 0, 2] = px
    K_[:, 1, 2] = py
    sign = [-1, 1, -1]
    for is_perspective in [True, False]:
        K = convert_K_3x3_to_4x4(K_, is_perspective=is_perspective)

        K1 = convert_ndc_to_screen(
            K,
            is_perspective=is_perspective,
            sign=sign,
            resolution=(1080, 1920))
        K2 = convert_screen_to_ndc(
            K1,
            is_perspective=is_perspective,
            sign=sign,
            resolution=(1080, 1920))
        assert np.isclose(K.all(), K2.all(), rtol=0, atol=eps)

        K1 = convert_ndc_to_screen(
            torch.Tensor(K),
            is_perspective=is_perspective,
            sign=sign,
            resolution=(1080, 1920))
        K2 = convert_screen_to_ndc(
            K1,
            is_perspective=is_perspective,
            sign=sign,
            resolution=(1080, 1920))
        assert torch.isclose(torch.Tensor(K), K2, rtol=0, atol=eps).all()


def test_convert_world_view():
    eps = 1e-2
    R = ee_to_rotmat(np.random.uniform(low=-np.pi, high=np.pi,
                                       size=(1, 3))).reshape(1, 3, 3)
    T = np.random.uniform(low=-10, high=10, size=(1, 3))
    R1, T1 = convert_world_view(R, T)
    R2, T2 = convert_world_view(R1, T1)
    assert np.isclose(R.all(), R2.all(), rtol=0, atol=eps)
    assert np.isclose(T.all(), T2.all(), rtol=0, atol=eps)

    R1, T1 = convert_world_view(torch.Tensor(R), torch.Tensor(T))
    R2, T2 = convert_world_view(torch.Tensor(R1), torch.Tensor(T1))
    assert torch.isclose(torch.Tensor(R), R2, rtol=0, atol=eps).all()
    assert torch.isclose(torch.Tensor(T), T2, rtol=0, atol=eps).all()

    with pytest.raises(TypeError):
        R1, T1 = convert_world_view(torch.Tensor(R), T)


def test_camera_utils():
    poses = torch.zeros(10, 72)
    R = torch.eye(3, 3)[None]
    T = torch.zeros(1, 3)
    resolution = (1080, 1920)
    transl = torch.zeros(10, 3)
    _, _ = convert_smpl_from_opencv_calibration(
        R=R,
        T=T,
        transl=transl,
        poses=poses,
        resolution=resolution,
        model_path=model_path)

    _, _ = convert_smpl_from_opencv_calibration(
        K=FoVPerspectiveCameras.get_default_projection_matrix(),
        R=R,
        T=T,
        transl=transl,
        poses=poses,
        resolution=resolution,
        model_path=model_path)


def test_convert_projection():
    K = torch.eye(4, 4)[None]
    K1 = convert_weakperspective_to_perspective(
        K=K, zmean=10, resolution=(1024, 1024), in_ndc=True)
    K2 = convert_perspective_to_weakperspective(
        K=K1, zmean=10, resolution=(1024, 1024), in_ndc=True)
    assert (K == K2).all()

import math

import torch

from mmhuman3d.core.cameras import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
    WeakPerspectiveCameras,
    build_cameras,
    compute_direction_cameras,
)
from mmhuman3d.core.cameras.camera_parameters import CameraParameter
from mmhuman3d.core.conventions.cameras.convert_convention import \
    convert_camera_matrix  # prevent yapf isort conflict


def check_camera_close(cam1, cam2, points=None, eps=1e-3):

    points = torch.rand(100, 3) if points is None else points
    assert cam1.device == cam2.device
    assert cam1.in_ndc() is cam2.in_ndc()
    assert cam1.is_perspective() is cam2.is_perspective()
    assert torch.isclose(
        cam1.transform_points_screen(points)[..., :2],
        cam2.transform_points_screen(points)[..., :2],
        rtol=0,
        atol=2e-1).all()

    assert torch.isclose(
        cam1.compute_depth_of_points(points)[..., :1],
        cam2.compute_depth_of_points(points)[..., :1],
        rtol=0,
        atol=eps).all()


def check_camera_slice(cam1):
    cam2 = cam1[0]
    cam2 = cam2.extend(len(cam1))
    check_camera_close(cam1, cam2)


def check_camera_concat(cam):
    cam0 = cam[0]
    cam1 = cam[1]
    cam2 = cam0.concat(cam1)
    check_camera_close(cam[:2], cam2)


def test_cameras_parameter():

    cam = PerspectiveCameras(
        T=torch.zeros(10, 3),
        image_size=((1080, 1920)),
        convention='opencv',
        focal_length=(500, 500),
        principal_point=(540, 960),
        R=torch.eye(3, 3)[None],
        in_ndc=False)
    cam_param = CameraParameter.load_from_perspective_cameras(cam, name='1')
    cam1 = cam_param.export_to_perspective_cameras()
    check_camera_close(cam[0], cam1)


def test_cameras():
    cameras_list = [
        FoVOrthographicCameras, FoVPerspectiveCameras, OrthographicCameras,
        PerspectiveCameras, WeakPerspectiveCameras
    ]
    K_list = [
        None,
        torch.eye(4, 4)[None],
        torch.eye(4, 4)[None].repeat(10, 1, 1)
    ]
    R_list = [
        None,
        torch.eye(3, 3)[None],
        torch.eye(3, 3)[None].repeat(10, 1, 1)
    ]
    T_list = [None, torch.zeros(1, 3), torch.zeros(10, 3)]
    for camera_type in cameras_list:
        image_size = (1080, 1920)
        # default K, R, T
        for idx in range(3):
            cam1 = build_cameras(
                dict(
                    type=camera_type.__name__,
                    K=K_list[idx],
                    R=R_list[idx],
                    T=T_list[idx],
                    image_size=image_size))
            cam2 = camera_type(
                K=K_list[idx],
                R=R_list[idx],
                T=T_list[idx],
                image_size=image_size)

            check_camera_close(cam1, cam2)
            check_camera_slice(cam1)
            if len(cam1) > 1:
                check_camera_concat(cam1)

            cam3 = build_cameras(
                dict(
                    type=camera_type.__name__,
                    K=K_list[idx],
                    R=R_list[idx],
                    T=T_list[idx],
                    convention='opencv',
                    image_size=image_size))
            K, R, T = convert_camera_matrix(
                K=K_list[idx],
                R=R_list[idx],
                T=T_list[idx],
                convention_src='opencv',
                convention_dst='pytorch3d')
            cam4 = build_cameras(
                dict(
                    type=camera_type.__name__,
                    K=K,
                    R=R,
                    T=T,
                    convention='pytorch3d',
                    image_size=image_size))
            check_camera_close(cam3, cam4)

    for camera_type in [PerspectiveCameras, OrthographicCameras]:
        cam1 = camera_type(
            principal_point=((1000, 1000.0)),
            focal_length=1000,
            image_size=(1080, 1920),
            in_ndc=False)
        cam2 = build_cameras(
            dict(
                type=camera_type.__name__,
                principal_point=((1000, 1000.0)),
                focal_length=(1000),
                image_size=(1080, 1920),
                in_ndc=False))
        assert not cam1.in_ndc()
        assert not cam2.in_ndc()
        assert cam1.is_perspective() is cam2.is_perspective()
        check_camera_close(cam1, cam2)

        cam1 = camera_type(
            principal_point=((1.73, 1.73)),
            focal_length=1.0,
            batch_size=1,
            image_size=(1080, 1920),
            in_ndc=True)
        cam1.extend_(100)
        cam2 = build_cameras(
            dict(
                type=camera_type,
                principal_point=((1.73, 1.73)),
                focal_length=1.0,
                batch_size=100,
                image_size=(1080, 1920),
                in_ndc=True))
        assert cam1.in_ndc()
        assert cam2.in_ndc()
        assert cam1.is_perspective() is cam2.is_perspective()
        assert len(cam1) == len(cam2) == 100
        check_camera_close(cam1, cam2)
        check_camera_slice(cam1)
        check_camera_slice(cam2)


def test_weakperspective():
    cam = build_cameras(dict(type='weakperspective', resolution=(1024, 1024)))
    orig_cam = cam.convert_K_to_orig_cam(torch.eye(4, 4)[None])
    K = cam.compute_projection_matrix(orig_cam[0, 0], orig_cam[0, 1],
                                      orig_cam[0, 2], orig_cam[0, 3], 1)
    cam.unproject_points(torch.zeros(1, 3))
    assert (K == torch.eye(4, 4)[None]).all()


def test_perspective():
    cam = build_cameras(dict(type='perspective', resolution=(1024, 1024)))
    cam1 = cam.to_screen()
    cam2 = cam1.to_ndc()
    check_camera_close(cam, cam2)

    cam2.to_screen_()
    cam2.to_ndc_()
    check_camera_close(cam, cam2)


def test_perspective_projection():

    def perspective_projection(points, rotation, translation, focal_length,
                               camera_center):
        batch_size = points.shape[0]
        K = torch.zeros([batch_size, 3, 3], device=points.device)
        K[:, 0, 0] = focal_length
        K[:, 1, 1] = focal_length
        K[:, 2, 2] = 1.
        K[:, :-1, -1] = camera_center

        # Transform points
        points = torch.einsum('bij,bkj->bki', rotation, points)
        points = points + translation.unsqueeze(1)

        # Apply perspective distortion
        projected_points = points / points[:, :, -1].unsqueeze(-1)

        # Apply camera intrinsics
        projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

        return projected_points[:, :, :-1]

    focal_length = 5000
    image_size = (224, 224)
    principal_point = (112, 112)

    camera = build_cameras(
        dict(
            type='PerspectiveCameras',
            convention='opencv',
            in_ndc=False,
            focal_length=focal_length,
            image_size=image_size,
            principal_point=principal_point))

    test_keypoints = torch.rand(10, 100, 3)

    projected_keypoints_xyd = camera.transform_points_screen(test_keypoints)
    projected_keypoints = projected_keypoints_xyd[..., :2]

    # alternative implementation
    projected_keypoints_alt = perspective_projection(
        test_keypoints,
        rotation=torch.eye(3).view(1, 3, 3).expand(10, -1, -1),
        translation=torch.zeros(10, 3),
        focal_length=focal_length,
        camera_center=torch.Tensor((112, 112)))

    assert torch.allclose(projected_keypoints, projected_keypoints_alt)


def test_direction_cameras():
    # Four ways to get direction cameras
    K1, R1, T1 = compute_direction_cameras(
        eye=(0, 0, 0), dist=math.sqrt(2), z_vec=(1, 0, 1))
    K2, R2, T2 = compute_direction_cameras(
        at=(1, 0, 1), dist=math.sqrt(2), z_vec=(1, 0, 1))
    K3, R3, T3 = compute_direction_cameras(
        at=(1, 0, 1), dist=math.sqrt(2), plane=((0, 1, 0), (1, 0, -1)))
    K4, R4, T4 = compute_direction_cameras(at=(1, 0, 1), eye=(0, 0, 0))

    cam1 = build_cameras(
        dict(
            type='perspective', in_ndc=True, image_size=256, K=K1, R=R1, T=T1))
    cam2 = build_cameras(
        dict(
            type='perspective', in_ndc=True, image_size=256, K=K2, R=R2, T=T2))
    cam3 = build_cameras(
        dict(
            type='perspective', in_ndc=True, image_size=256, K=K3, R=R3, T=T3))
    cam4 = build_cameras(
        dict(
            type='perspective', in_ndc=True, image_size=256, K=K4, R=R4, T=T4))

    points = torch.Tensor([[0, 0, 1]])
    check_camera_close(cam1, cam2, points=points)
    check_camera_close(cam1, cam3, points=points)
    check_camera_close(cam1, cam4, points=points)

import torch

from mmhuman3d.core.cameras import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
    WeakPerspectiveCameras,
    build_cameras,
)
from mmhuman3d.core.conventions.cameras import convert_cameras


def check_camera_close(cam1, cam2, eps=1e-3):
    N = 100
    points = torch.rand(N, 3)
    assert cam1.device == cam2.device
    assert cam1.in_ndc() is cam2.in_ndc()
    assert cam1.is_perspective() is cam2.is_perspective()

    assert torch.isclose(
        cam1.transform_points_screen(points)[..., :2],
        cam2.transform_points_screen(points)[..., :2],
        rtol=0,
        atol=eps).all()

    assert torch.isclose(
        cam1.transform_points_screen(points)[..., :2],
        cam2.transform_points_screen(points)[..., :2],
        rtol=0,
        atol=eps).all()
    assert torch.isclose(
        cam1.compute_depth_of_points(points)[..., :2],
        cam2.compute_depth_of_points(points)[..., :2],
        rtol=0,
        atol=eps).all()


def check_camera_slice(cam1):
    cam2 = cam1[0]
    cam2 = cam2.extend(len(cam1))
    check_camera_close(cam1, cam2)
    print(cam1.__repr__)


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

            cam3 = build_cameras(
                dict(
                    type=camera_type.__name__,
                    K=K_list[idx],
                    R=R_list[idx],
                    T=T_list[idx],
                    convention='opencv',
                    image_size=image_size))
            K, R, T = convert_cameras(
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

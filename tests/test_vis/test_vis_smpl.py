import subprocess

import numpy as np
import pytest
import torch

from mmhuman3d.core.visualization import (
    visualize_smpl_calibration,
    visualize_smpl_hmr,
    visualize_smpl_pose,
    visualize_smpl_vibe,
    visualize_T_pose,
)
from mmhuman3d.utils.ffmpeg_utils import (
    array_to_images,
    array_to_video,
    video_to_array,
)

model_path = 'data/body_models'


def test_visualize_smpl_pose():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    # wrong input shape
    with pytest.raises(ValueError):
        visualize_smpl_pose(
            poses=torch.zeros(2, 71),
            model_type='smpl',
            model_path=model_path,
            output_path='/tmp/1.mp4',
            render_choice='hq',
            resolution=(128, 128),
            overwrite=True,
            device=device_name)
    with pytest.raises(ValueError):
        visualize_smpl_pose(
            poses=torch.zeros(2, 164),
            model_type='smplx',
            model_path=model_path,
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            render_choice='hq',
            overwrite=True,
            device=device_name)
    with pytest.raises(RuntimeError):
        pose_dict = {
            'body_pose': torch.zeros(2, 68),
            'global_orient': torch.zeros(2, 3)
        }
        visualize_smpl_pose(
            poses=pose_dict,
            model_type='smpl',
            model_path=model_path,
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            render_choice='hq',
            overwrite=True,
            device=device_name)
    with pytest.raises(RuntimeError):
        pose_dict = {
            'body_pose': torch.zeros(2, 64),
            'global_orient': torch.zeros(2, 3),
            'left_hand_pose': torch.zeros(2, 45),
            'right_hand_pose': torch.zeros(2, 45),
            'jaw_pose': torch.zeros(2, 3),
            'leye_pose': torch.zeros(2, 3),
            'reye_pose': torch.zeros(2, 3),
        }
        visualize_smpl_pose(
            poses=pose_dict,
            model_type='smplx',
            model_path=model_path,
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            render_choice='hq',
            overwrite=True,
            device=device_name)
    # wrong input keys
    with pytest.raises(KeyError):
        pose_dict = {
            'wrong_smpl_name': torch.zeros(2, 69),
            'global_orient': torch.zeros(2, 3)
        }
        visualize_smpl_pose(
            poses=pose_dict,
            model_type='smpl',
            model_path=model_path,
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            render_choice='hq',
            overwrite=True,
            device=device_name)

    with pytest.raises(KeyError):
        pose_dict = {
            'wrong_smplx_name': torch.zeros(2, 63),
            'global_orient': torch.zeros(2, 3),
            'left_hand_pose': torch.zeros(2, 45),
            'right_hand_pose': torch.zeros(2, 45),
            'jaw_pose': torch.zeros(2, 3),
            'leye_pose': torch.zeros(2, 3),
            'reye_pose': torch.zeros(2, 3),
        }
        visualize_smpl_pose(
            poses=pose_dict,
            model_type='smplx',
            model_path=model_path,
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            render_choice='hq',
            overwrite=True,
            device=device_name)
    # wrong output path
    with pytest.raises(FileExistsError):
        v = np.zeros((3, 512, 512, 3))
        array_to_video(v, output_path='/tmp/1.mp4')
        visualize_smpl_pose(
            poses=torch.zeros(2, 72),
            model_type='smpl',
            output_path='/tmp/1.mp4',
            model_path=model_path,
            resolution=(128, 128),
            render_choice='hq',
            overwrite=False,
            device=device_name)

    # wrong body model weight path
    with pytest.raises(FileNotFoundError):
        command = ['touch', '/tmp/1.mp4']
        subprocess.call(command)
        visualize_smpl_pose(
            poses=torch.zeros(2, 72),
            model_type='smpl',
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            model_path='/312',
            render_choice='hq',
            overwrite=True,
            device=device_name)

    with pytest.raises(AssertionError):
        command = ['touch', '/tmp/1.mp4']
        subprocess.call(command)
        visualize_smpl_pose(
            poses=torch.zeros(2, 72),
            model_type='smpl',
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            model_path='/tmp',
            render_choice='hq',
            overwrite=True,
            device=device_name)

    with pytest.raises(FileNotFoundError):
        command = ['touch', '/tmp/1.mp4']
        subprocess.call(command)
        visualize_smpl_pose(
            poses=torch.zeros(2, 72),
            model_type='smpl',
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            model_path='/123',
            render_choice='hq',
            overwrite=True,
            device=device_name)

    visualize_smpl_pose(
        poses=torch.zeros(1, 72),
        model_type='smpl',
        model_path=model_path,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (1, 48, 48, 3)

    visualize_smpl_pose(
        poses=torch.zeros(1, 2, 72),
        model_type='smpl',
        betas=torch.zeros(1, 10),
        model_path=model_path,
        output_path='/tmp/1.mp4',
        resolution=(128, 128),
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (1, 128, 128, 3)

    visualize_smpl_pose(
        poses=torch.zeros(1, 2, 72),
        betas=torch.zeros(1, 2, 10),
        model_type='smpl',
        model_path=model_path,
        output_path='/tmp/1.mp4',
        resolution=(128, 128),
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (1, 128, 128, 3)

    with pytest.raises(ValueError):
        visualize_smpl_pose(
            poses=torch.zeros(1, 3, 72),
            betas=torch.zeros(1, 2, 10),
            model_type='smpl',
            model_path=model_path,
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            overwrite=True,
            device=device_name)
    with pytest.raises(ValueError):
        visualize_smpl_pose(
            poses=torch.zeros(1, 3, 72),
            transl=torch.zeros(1, 2, 3),
            model_type='smpl',
            model_path=model_path,
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            overwrite=True,
            device=device_name)

    visualize_smpl_pose(
        poses=torch.zeros(1, 2, 72),
        betas=torch.zeros(1, 3, 10),
        transl=torch.zeros(1, 3, 3),
        model_type='smpl',
        model_path=model_path,
        output_path='/tmp/1.mp4',
        resolution=(128, 128),
        overwrite=True,
        device=device_name)
    visualize_smpl_pose(
        poses=torch.zeros(10, 72),
        betas=torch.zeros(1, 10),
        model_type='smpl',
        model_path=model_path,
        output_path='/tmp/1.mp4',
        resolution=(128, 128),
        overwrite=True,
        device=device_name)

    visualize_smpl_pose(
        poses=torch.zeros(1, 165),
        model_type='smplx',
        model_path=model_path,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (1, 48, 48, 3)

    visualize_smpl_pose(
        poses=torch.zeros(1, 72),
        model_type='smpl',
        model_path=model_path,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (1, 48, 48, 3)

    pose_dict = {
        'body_pose': torch.zeros(2, 69),
        'global_orient': torch.zeros(2, 3)
    }
    visualize_smpl_pose(
        poses=pose_dict,
        model_type='smpl',
        model_path=model_path,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (2, 48, 48, 3)

    pose_dict = {
        'body_pose': torch.zeros(2, 63),
        'global_orient': torch.zeros(2, 3),
        'left_hand_pose': torch.zeros(2, 45),
        'right_hand_pose': torch.zeros(2, 45),
        'jaw_pose': torch.zeros(2, 3),
        'leye_pose': torch.zeros(2, 3),
        'reye_pose': torch.zeros(2, 3),
    }
    visualize_smpl_pose(
        poses=pose_dict,
        model_type='smplx',
        model_path=model_path,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (2, 48, 48, 3)

    pred_cam = torch.ones(10, 4)
    bbox = torch.tensor([0, 0, 100, 100]).view(1, 4).repeat(10, 1)
    visualize_smpl_vibe(
        poses=pose_dict,
        model_type='smplx',
        pred_cam=pred_cam,
        bbox=bbox,
        model_path=model_path,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (2, 48, 48, 3)

    visualize_T_pose(
        num_frames=2,
        model_type='smplx',
        orbit_speed=(1.0, 0.5),
        model_path=model_path,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (2, 48, 48, 3)

    visualize_T_pose(
        num_frames=2,
        model_type='smpl',
        orbit_speed=(1.0, 0.5),
        model_path=model_path,
        output_path='/tmp/1.mp4',
        palette='segmentation',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (2, 48, 48, 3)

    visualize_T_pose(
        num_frames=2,
        model_type='smpl',
        orbit_speed=(1.0, 0.5),
        model_path=model_path,
        render_choice='normal',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (2, 48, 48, 3)

    visualize_T_pose(
        num_frames=2,
        model_type='smpl',
        orbit_speed=(1.0, 0.5),
        model_path=model_path,
        render_choice='depth',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (2, 48, 48, 3)

    visualize_T_pose(
        num_frames=2,
        model_type='smpl',
        orbit_speed=(1.0, 0.5),
        model_path=model_path,
        render_choice='pointcloud',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (2, 48, 48, 3)

    visualize_T_pose(
        num_frames=2,
        model_type='smpl',
        orbit_speed=(1.0, 0.5),
        model_path=model_path,
        render_choice='silhouette',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (2, 48, 48, 3)

    visualize_T_pose(
        num_frames=2,
        model_type='smpl',
        orbit_speed=(1.0, 0.5),
        model_path=model_path,
        render_choice='part_silhouette',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (2, 48, 48, 3)

    visualize_T_pose(
        num_frames=2,
        model_type='smpl',
        orbit_speed=(1.0, 0.5),
        model_path=model_path,
        render_choice='mq',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (2, 48, 48, 3)

    visualize_T_pose(
        num_frames=2,
        model_type='smpl',
        orbit_speed=(1.0, 0.5),
        model_path=model_path,
        render_choice='lq',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (2, 48, 48, 3)

    visualize_T_pose(
        num_frames=2,
        model_type='smpl',
        orbit_speed=(1.0, 0.5),
        model_path=model_path,
        render_choice='hq',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        device=device_name)
    assert video_to_array('/tmp/1.mp4').shape == (2, 48, 48, 3)

    K = torch.zeros(1, 4, 4)
    K[:, 0, 0] = 1
    K[:, 1, 1] = 1
    K[:, 0, 2] = 1
    K[:, 1, 2] = 1
    R = torch.eye(3, 3)[None]
    T = torch.zeros(1, 3)
    visualize_smpl_calibration(
        poses=pose_dict,
        model_type='smplx',
        model_path=model_path,
        K=K,
        R=R,
        T=T,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)

    K = torch.zeros(1, 4, 4)
    K[:, 0, 0] = 1
    K[:, 1, 1] = 1
    K[:, 0, 2] = 1
    K[:, 1, 2] = 1
    R = torch.eye(3, 3)[None]
    T = torch.zeros(1, 3)
    betas = torch.zeros(2, 10)
    transl = torch.zeros(2, 3)
    visualize_smpl_calibration(
        poses=pose_dict,
        model_type='smplx',
        model_path=model_path,
        betas=betas,
        transl=transl,
        K=K,
        R=R,
        T=T,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)

    bbox = np.zeros((3, 1, 4))
    cam_transl = torch.zeros(3, 1, 3)
    visualize_smpl_hmr(
        poses=torch.zeros(3, 165),
        model_type='smplx',
        model_path=model_path,
        bbox=bbox,
        cam_transl=cam_transl,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)

    bbox = np.zeros((3, 1, 4))
    cam_transl = torch.zeros(3, 1, 3)
    visualize_smpl_hmr(
        verts=torch.zeros(3, 6890, 3),
        model_type='smpl',
        model_path=model_path,
        bbox=bbox,
        cam_transl=cam_transl,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)

    bbox = np.zeros((3, 1, 4))
    T = torch.zeros(3, 1, 3)
    visualize_smpl_hmr(
        poses=torch.zeros(3, 165),
        model_type='smplx',
        model_path=model_path,
        bbox=bbox,
        cam_transl=cam_transl,
        origin_frames='/tmp/1.mp4',
        output_path='/tmp/2.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)

    image_array = np.random.randint(
        low=0, high=255, size=(3, 128, 128, 3), dtype=np.uint8)
    array_to_images(image_array, '/tmp/temp_images', img_format='%06d.png')

    bbox = np.zeros((3, 1, 4))
    cam_transl = torch.zeros(3, 1, 3)
    visualize_smpl_hmr(
        poses=torch.zeros(3, 165),
        model_type='smplx',
        model_path=model_path,
        bbox=bbox,
        cam_transl=cam_transl,
        output_path='/tmp/1.mp4',
        frame_list=['/tmp/temp_images/%06d.png' % 0] * 3,
        resolution=(48, 48),
        overwrite=True,
        device=device_name)

    bbox = np.zeros((3, 1, 4))
    cam_transl = torch.zeros(3, 1, 3)
    visualize_smpl_hmr(
        poses=torch.zeros(3, 165),
        model_type='smplx',
        model_path=model_path,
        bbox=bbox,
        cam_transl=cam_transl,
        output_path='/tmp/1.mp4',
        img_format='%06d.png',
        origin_frames='/tmp/temp_images',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)

    visualize_smpl_hmr(
        poses=torch.zeros(3, 165),
        model_type='smplx',
        model_path=model_path,
        bbox=bbox,
        cam_transl=cam_transl,
        output_path='/tmp/1.mp4',
        origin_frames='/tmp/temp_images',
        img_format='%06d.png',
        resolution=(48, 48),
        overwrite=True,
        palette=np.ones((1, 3)),
        device=device_name)

    visualize_smpl_hmr(
        poses=torch.zeros(3, 3, 165),
        model_type='smplx',
        model_path=model_path,
        bbox=np.zeros((3, 3, 4)),
        cam_transl=torch.zeros(3, 3, 3),
        output_path='/tmp/1.mp4',
        origin_frames='/tmp/temp_images',
        img_format='%06d.png',
        resolution=(128, 128),
        overwrite=True,
        palette=np.ones((1, 3)),
        device=device_name)

    visualize_smpl_hmr(
        poses=torch.zeros(3, 3, 165),
        model_type='smplx',
        model_path=model_path,
        bbox=np.zeros((3, 3, 4)),
        cam_transl=torch.zeros(3, 3, 3),
        output_path='/tmp/1.mp4',
        origin_frames='/tmp/temp_images',
        img_format='%06d.png',
        resolution=(128, 128),
        overwrite=True,
        mesh_file_path='/tmp',
        palette=np.ones((1, 3)),
        device=device_name)

    visualize_smpl_hmr(
        poses=torch.zeros(3, 3, 165),
        model_type='smplx',
        model_path=model_path,
        bbox=np.zeros((3, 3, 4)),
        cam_transl=torch.zeros(3, 3, 3),
        output_path='/tmp/1.mp4',
        origin_frames='/tmp/temp_images',
        img_format='%06d.png',
        resolution=(128, 128),
        overwrite=True,
        mesh_file_path='/tmp',
        palette='random',
        device=device_name)

    with pytest.raises(ValueError):
        visualize_smpl_hmr(
            poses=torch.zeros(3, 3, 165),
            model_type='smplx',
            model_path=model_path,
            bbox=np.zeros((3, 3, 4)),
            cam_transl=torch.zeros(3, 3, 3),
            output_path='/tmp/1.mp4',
            origin_frames='/tmp/temp_images',
            img_format='%06d.png',
            resolution=(128, 128),
            overwrite=True,
            mesh_file_path='/tmp',
            palette='wrong_palette',
            device=device_name)

    with pytest.raises(ValueError):
        visualize_smpl_hmr(
            poses=torch.zeros(3, 3, 165),
            model_type='smplx',
            model_path=model_path,
            bbox=np.zeros((3, 3, 4)),
            cam_transl=torch.zeros(3, 3, 3),
            output_path='/tmp/1.mp4',
            origin_frames='/tmp/temp_images',
            img_format='%06d.png',
            resolution=(128, 128),
            overwrite=True,
            mesh_file_path='/tmp',
            palette=None,
            device=device_name)

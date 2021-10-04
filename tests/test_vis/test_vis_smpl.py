import subprocess

import numpy as np
import pytest
import torch

from mmhuman3d.core.visualization.visualize_smpl import (
    neural_render_part_silhouette,
    neural_render_silhouette,
    visualize_part_silhouette,
    visualize_smpl_opencv,
    visualize_smpl_pose,
    visualize_smpl_pred,
)
from mmhuman3d.utils.ffmpeg_utils import array_to_video, video_to_array

body_model_dir = 'body_models'


def test_visualize_smpl_pose():
    # wrong render_choice, visualize smpl_pose should only chose quality!
    with pytest.raises(ValueError):
        visualize_smpl_pose(
            torch.zeros(10, 72),
            model_type='smpl',
            body_model_dir=body_model_dir,
            output_path='/tmp/1.mp4',
            render_choice='silhouette',
            overwrite=True)
    # wrong input shape
    with pytest.raises(ValueError):
        visualize_smpl_pose(
            poses=torch.zeros(10, 71),
            model_type='smpl',
            body_model_dir=body_model_dir,
            output_path='/tmp/1.mp4',
            render_choice='hq',
            overwrite=True)
    with pytest.raises(ValueError):
        visualize_smpl_pose(
            poses=torch.zeros(10, 164),
            model_type='smplx',
            body_model_dir=body_model_dir,
            output_path='/tmp/1.mp4',
            render_choice='hq',
            overwrite=True)
    with pytest.raises(RuntimeError):
        pose_dict = {
            'body_pose': torch.zeros(10, 68),
            'global_orient': torch.zeros(10, 3)
        }
        visualize_smpl_pose(
            poses=pose_dict,
            model_type='smpl',
            body_model_dir=body_model_dir,
            output_path='/tmp/1.mp4',
            render_choice='hq',
            overwrite=True)
    with pytest.raises(RuntimeError):
        pose_dict = {
            'body_pose': torch.zeros(10, 64),
            'global_orient': torch.zeros(10, 3),
            'left_hand_pose': torch.zeros(10, 45),
            'right_hand_pose': torch.zeros(10, 45),
            'jaw_pose': torch.zeros(10, 3),
            'leye_pose': torch.zeros(10, 3),
            'reye_pose': torch.zeros(10, 3),
        }
        visualize_smpl_pose(
            poses=pose_dict,
            model_type='smplx',
            body_model_dir=body_model_dir,
            output_path='/tmp/1.mp4',
            render_choice='hq',
            overwrite=True)
    # wrong input keys
    with pytest.raises(KeyError):
        pose_dict = {
            'wrong_smpl_name': torch.zeros(10, 69),
            'global_orient': torch.zeros(10, 3)
        }
        visualize_smpl_pose(
            poses=pose_dict,
            model_type='smpl',
            body_model_dir=body_model_dir,
            output_path='/tmp/1.mp4',
            render_choice='hq',
            overwrite=True)

    with pytest.raises(KeyError):
        pose_dict = {
            'wrong_smplx_name': torch.zeros(10, 63),
            'global_orient': torch.zeros(10, 3),
            'left_hand_pose': torch.zeros(10, 45),
            'right_hand_pose': torch.zeros(10, 45),
            'jaw_pose': torch.zeros(10, 3),
            'leye_pose': torch.zeros(10, 3),
            'reye_pose': torch.zeros(10, 3),
        }
        visualize_smpl_pose(
            poses=pose_dict,
            model_type='smplx',
            body_model_dir=body_model_dir,
            output_path='/tmp/1.mp4',
            render_choice='hq',
            overwrite=True)
    # wrong output path
    with pytest.raises(FileExistsError):
        v = np.zeros((3, 512, 512, 3))
        array_to_video(v, output_path='/tmp/1.mp4')
        visualize_smpl_pose(
            poses=torch.zeros(10, 72),
            model_type='smpl',
            output_path='/tmp/1.mp4',
            body_model_dir=body_model_dir,
            render_choice='hq',
            overwrite=False)

    # wrong body model weight path
    with pytest.raises(FileNotFoundError):
        command = ['touch', '/tmp/1.mp4']
        subprocess.call(command)
        visualize_smpl_pose(
            poses=torch.zeros(10, 72),
            model_type='smpl',
            output_path='/tmp/1.mp4',
            body_model_dir='/312',
            render_choice='hq',
            overwrite=True)

    with pytest.raises(AssertionError):
        command = ['touch', '/tmp/1.mp4']
        subprocess.call(command)
        visualize_smpl_pose(
            poses=torch.zeros(10, 72),
            model_type='smpl',
            output_path='/tmp/1.mp4',
            body_model_dir='/tmp',
            render_choice='hq',
            overwrite=True)
    visualize_smpl_pose(
        poses=torch.zeros(1, 72),
        model_type='smpl',
        body_model_dir=body_model_dir,
        output_path='/tmp/1.mp4',
        resolution=(1024, 1024),
        overwrite=True)
    assert video_to_array('/tmp/1.mp4').shape == (1, 1024, 1024, 3)

    visualize_smpl_pose(
        poses=torch.zeros(1, 165),
        model_type='smplx',
        body_model_dir=body_model_dir,
        output_path='/tmp/1.mp4',
        resolution=(1024, 1024),
        overwrite=True)
    assert video_to_array('/tmp/1.mp4').shape == (1, 1024, 1024, 3)

    visualize_smpl_pose(
        poses=torch.zeros(1, 72),
        model_type='smpl',
        body_model_dir=body_model_dir,
        output_path='/tmp/1.mp4',
        resolution=(1024, 1024),
        overwrite=True)
    assert video_to_array('/tmp/1.mp4').shape == (1, 1024, 1024, 3)

    pose_dict = {
        'body_pose': torch.zeros(10, 69),
        'global_orient': torch.zeros(10, 3)
    }
    visualize_smpl_pose(
        poses=pose_dict,
        model_type='smpl',
        body_model_dir=body_model_dir,
        output_path='/tmp/1.mp4',
        resolution=(1024, 1024),
        overwrite=True)
    assert video_to_array('/tmp/1.mp4').shape == (10, 1024, 1024, 3)

    pose_dict = {
        'body_pose': torch.zeros(10, 63),
        'global_orient': torch.zeros(10, 3),
        'left_hand_pose': torch.zeros(10, 45),
        'right_hand_pose': torch.zeros(10, 45),
        'jaw_pose': torch.zeros(10, 3),
        'leye_pose': torch.zeros(10, 3),
        'reye_pose': torch.zeros(10, 3),
    }
    visualize_smpl_pose(
        poses=pose_dict,
        model_type='smplx',
        body_model_dir=body_model_dir,
        output_path='/tmp/1.mp4',
        resolution=(1024, 1024),
        overwrite=True)
    assert video_to_array('/tmp/1.mp4').shape == (10, 1024, 1024, 3)

    pred_cam = torch.ones(10, 4)
    visualize_part_silhouette(
        poses=pose_dict,
        model_type='smplx',
        pred_cam=pred_cam,
        body_model_dir=body_model_dir,
        output_path='/tmp/1.mp4',
        resolution=(1024, 1024),
        overwrite=True)
    assert video_to_array('/tmp/1.mp4').shape == (10, 1024, 1024, 3)

    pred_cam = torch.ones(10, 4)
    visualize_smpl_pred(
        poses=pose_dict,
        model_type='smplx',
        pred_cam=pred_cam,
        body_model_dir=body_model_dir,
        output_path='/tmp/1.mp4',
        resolution=(1024, 1024),
        overwrite=True)
    assert video_to_array('/tmp/1.mp4').shape == (10, 1024, 1024, 3)

    K = torch.zeros(1, 4, 4)
    K[:, 0, 0] = 1
    K[:, 1, 1] = 1
    K[:, 0, 2] = 1
    K[:, 1, 2] = 1
    R = torch.eye(3, 3)[None]
    T = torch.zeros(1, 3)
    visualize_smpl_opencv(
        poses=pose_dict,
        model_type='smplx',
        body_model_dir=body_model_dir,
        K=K,
        R=R,
        T=T,
        output_path='/tmp/1.mp4',
        resolution=(1024, 1024),
        overwrite=True)


def test_neural_render():
    poses = torch.zeros(3, 72)
    poses.requires_grad = True
    pred_cam = torch.ones(3, 4)
    res_tensor = neural_render_silhouette(
        poses=torch.zeros(3, 72),
        model_type='smpl',
        pred_cam=pred_cam,
        body_model_dir=body_model_dir,
        resolution=(1024, 1024))
    assert res_tensor.shape == (3, 1024, 1024)
    assert res_tensor.requires_grad

    res_tensor = neural_render_part_silhouette(
        poses=torch.zeros(3, 72),
        model_type='smpl',
        pred_cam=pred_cam,
        body_model_dir=body_model_dir,
        resolution=(1024, 1024))
    assert res_tensor.shape == (3, 1024, 1024, 24)
    assert res_tensor.requires_grad

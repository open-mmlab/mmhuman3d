import subprocess

import numpy as np
import pytest
import torch

from mmhuman3d.core.visualization.visualize_smpl import (
    visualize_smpl_calibration,
    visualize_smpl_hmr,
    visualize_smpl_pose,
    visualize_smpl_vibe,
    visualize_T_pose,
)
from mmhuman3d.utils.ffmpeg_utils import array_to_images, array_to_video

body_model_config = {
    'use_pca': False,
    'use_face_contour': True,
    'model_path': 'data/body_models'
}


def test_visualize_smpl_pose():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    # wrong input tensor shape for smpl
    with pytest.raises(ValueError):
        body_model_config.update(type='smpl')
        visualize_smpl_pose(
            poses=torch.zeros(2, 71),
            body_model_config=body_model_config,
            output_path='/tmp/1.mp4',
            render_choice='hq',
            resolution=(128, 128),
            overwrite=True,
            device=device_name)
    # wrong input tensor shape for smplx
    with pytest.raises(ValueError):
        visualize_smpl_pose(
            poses=torch.zeros(2, 164),
            body_model_config=body_model_config,
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            render_choice='hq',
            overwrite=True,
            return_tensor=True,
            device=device_name)
    # wrong input tensor shape for smpl dict
    with pytest.raises(RuntimeError):
        pose_dict = {
            'body_pose': torch.zeros(2, 68),
            'global_orient': torch.zeros(2, 3)
        }
        body_model_config.update(type='smpl')
        visualize_smpl_pose(
            poses=pose_dict,
            body_model_config=body_model_config,
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            render_choice='hq',
            overwrite=True,
            vis_kp_index=True,
            device=device_name)
    # wrong input tensor shape for smplx dict
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
        body_model_config.update(type='smplx')
        visualize_smpl_pose(
            poses=pose_dict,
            body_model_config=body_model_config,
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            render_choice='hq',
            overwrite=True,
            device=device_name)
    # wrong input dict keys for smpl dict
    with pytest.raises(KeyError):
        pose_dict = {
            'wrong_smpl_name': torch.zeros(2, 69),
            'global_orient': torch.zeros(2, 3)
        }
        body_model_config.update(type='smpl')
        visualize_smpl_pose(
            poses=pose_dict,
            body_model_config=body_model_config,
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            render_choice='hq',
            overwrite=True,
            device=device_name)
    # wrong input dict keys for smplx dict
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
        body_model_config.update(type='smplx')
        visualize_smpl_pose(
            poses=pose_dict,
            body_model_config=body_model_config,
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            render_choice='hq',
            overwrite=True,
            device=device_name)
    # wrong output path, write to existing file path without overwrite=True
    body_model_config.update(type='smpl')
    with pytest.raises(FileExistsError):
        v = np.zeros((3, 512, 512, 3))
        array_to_video(v, output_path='/tmp/1.mp4')
        visualize_smpl_pose(
            poses=torch.zeros(2, 72),
            output_path='/tmp/1.mp4',
            body_model_config=body_model_config,
            resolution=(128, 128),
            render_choice='hq',
            overwrite=False,
            device=device_name)

    # wrong body model weight path, folder does not exist
    body_model_config.update(type='smpl')
    with pytest.raises(FileNotFoundError):
        body_model_config_ = body_model_config.copy()
        body_model_config_.update(model_path='/321')
        command = ['touch', '/tmp/1.mp4']
        subprocess.call(command)
        visualize_smpl_pose(
            poses=torch.zeros(2, 72),
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            body_model_config=body_model_config_,
            render_choice='hq',
            overwrite=True,
            device=device_name)

    # wrong body model weight path, folder exist without body model file
    body_model_config.update(type='smpl')
    with pytest.raises(AssertionError):
        command = ['touch', '/tmp/1.mp4']
        subprocess.call(command)
        body_model_config_ = body_model_config.copy()
        body_model_config_.update(model_path='/tmp')
        visualize_smpl_pose(
            poses=torch.zeros(2, 72),
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            body_model_config=body_model_config_,
            render_choice='hq',
            overwrite=True,
            device=device_name)

    # render single frame single person of smpl mesh
    body_model_config.update(type='smpl')
    tensor = visualize_smpl_pose(
        poses=torch.zeros(1, 72),
        body_model_config=body_model_config,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        return_tensor=True,
        plot_kps=True,
        device=device_name)
    assert tensor.shape == (1, 48, 48, 4)

    # render single frame multiple person of same betas of smpl mesh
    body_model_config.update(type='smpl')
    tensor = visualize_smpl_pose(
        poses=torch.zeros(1, 2, 72),
        betas=torch.zeros(1, 10),
        body_model_config=body_model_config,
        output_path='/tmp/1.mp4',
        resolution=(128, 128),
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (1, 128, 128, 4)

    # render single frame multiple person of different betas of smpl mesh
    body_model_config.update(type='smpl')
    tensor = visualize_smpl_pose(
        poses=torch.zeros(1, 2, 72),
        betas=torch.zeros(1, 2, 10),
        body_model_config=body_model_config,
        output_path='/tmp/1.mp4',
        resolution=(128, 128),
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (1, 128, 128, 4)

    # when render multiple person, betas number should be one or same as person
    # number
    body_model_config.update(type='smpl')
    with pytest.raises(ValueError):
        visualize_smpl_pose(
            poses=torch.zeros(1, 3, 72),
            betas=torch.zeros(1, 2, 10),
            body_model_config=body_model_config,
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            overwrite=True,
            device=device_name)

    # when render multiple person, transl number should be one or same as
    # person number
    body_model_config.update(type='smpl')
    with pytest.raises(ValueError):
        visualize_smpl_pose(
            poses=torch.zeros(1, 3, 72),
            transl=torch.zeros(1, 2, 3),
            body_model_config=body_model_config,
            output_path='/tmp/1.mp4',
            resolution=(128, 128),
            overwrite=True,
            device=device_name)

    # render multiple person, betas and transl will be sliced according to
    # the person number indicated by poses
    body_model_config.update(type='smpl')
    visualize_smpl_pose(
        poses=torch.zeros(1, 2, 72),
        betas=torch.zeros(1, 3, 10),
        transl=torch.zeros(1, 3, 3),
        body_model_config=body_model_config,
        output_path='/tmp/1.mp4',
        resolution=(128, 128),
        overwrite=True,
        device=device_name)

    # render 10 frames of single smpl mesh using single betas.
    body_model_config.update(type='smpl')
    visualize_smpl_pose(
        poses=torch.zeros(10, 72),
        betas=torch.zeros(1, 10),
        body_model_config=body_model_config,
        output_path='/tmp/1.mp4',
        resolution=(128, 128),
        overwrite=True,
        device=device_name)

    # render 1 frame of single smplx mesh using default betas.
    body_model_config.update(type='smplx')
    tensor = visualize_smpl_pose(
        poses=torch.zeros(1, 165),
        body_model_config=body_model_config,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (1, 48, 48, 4)

    # render 1 frame of single smpl mesh using default betas.
    body_model_config.update(type='smpl')
    tensor = visualize_smpl_pose(
        poses=torch.zeros(1, 72),
        body_model_config=body_model_config,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (1, 48, 48, 4)

    # use pose_dict to render single smpl mesh using default betas.
    pose_dict = {
        'body_pose': torch.zeros(2, 69),
        'global_orient': torch.zeros(2, 3)
    }
    body_model_config.update(type='smpl')
    tensor = visualize_smpl_pose(
        poses=pose_dict,
        body_model_config=body_model_config,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (2, 48, 48, 4)

    # use pose_dict to render single smplx mesh using default betas.
    pose_dict = {
        'body_pose': torch.zeros(2, 63),
        'global_orient': torch.zeros(2, 3),
        'left_hand_pose': torch.zeros(2, 45),
        'right_hand_pose': torch.zeros(2, 45),
        'jaw_pose': torch.zeros(2, 3),
        'leye_pose': torch.zeros(2, 3),
        'reye_pose': torch.zeros(2, 3),
    }
    body_model_config.update(type='smplx')
    tensor = visualize_smpl_pose(
        poses=pose_dict,
        body_model_config=body_model_config,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (2, 48, 48, 4)

    # use vibe camera outputs to render single smplx mesh using
    # function visualize_smpl_vibe.
    body_model_config.update(type='smplx')
    pred_cam = torch.ones(10, 4)
    bbox = np.array([0, 0, 100, 100]).reshape(1, 4).repeat(10, 1)
    tensor = visualize_smpl_vibe(
        poses=pose_dict,
        pred_cam=pred_cam,
        bbox=bbox,
        body_model_config=body_model_config,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (2, 48, 48, 4)

    # test function visualize_T_pose to render smplx mesh
    body_model_config.update(type='smplx')
    tensor = visualize_T_pose(
        num_frames=2,
        orbit_speed=(1.0, 0.5),
        body_model_config=body_model_config,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (2, 48, 48, 4)

    # render colorful smpl mesh
    body_model_config.update(type='smpl')
    tensor = visualize_T_pose(
        num_frames=2,
        orbit_speed=(1.0, 0.5),
        body_model_config=body_model_config,
        output_path='/tmp/1.mp4',
        palette='segmentation',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (2, 48, 48, 4)

    # render normal map of smpl mesh
    body_model_config.update(type='smpl')
    tensor = visualize_T_pose(
        num_frames=2,
        orbit_speed=(1.0, 0.5),
        body_model_config=body_model_config,
        render_choice='normal',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (2, 48, 48, 3)

    # render depth map of smpl mesh
    body_model_config.update(type='smpl')
    tensor = visualize_T_pose(
        num_frames=2,
        orbit_speed=(1.0, 0.5),
        body_model_config=body_model_config,
        render_choice='depth',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (2, 48, 48, 1)

    # render pointcloud of smpl mesh
    body_model_config.update(type='smpl')
    tensor = visualize_T_pose(
        num_frames=2,
        orbit_speed=(1.0, 0.5),
        body_model_config=body_model_config,
        render_choice='pointcloud',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (2, 48, 48, 4)

    # render silhouette mask of smpl mesh
    body_model_config.update(type='smpl')
    tensor = visualize_T_pose(
        num_frames=2,
        orbit_speed=(1.0, 0.5),
        body_model_config=body_model_config,
        render_choice='silhouette',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (2, 48, 48, 4)

    # render body part silhouette of smpl mesh
    body_model_config.update(type='smpl')
    tensor = visualize_T_pose(
        num_frames=2,
        orbit_speed=(1.0, 0.5),
        body_model_config=body_model_config,
        render_choice='part_silhouette',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (2, 48, 48, 1)

    # render smpl mesh in medium quaility
    body_model_config.update(type='smpl')
    tensor = visualize_T_pose(
        num_frames=2,
        orbit_speed=(1.0, 0.5),
        body_model_config=body_model_config,
        render_choice='mq',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (2, 48, 48, 4)

    # render smpl mesh in low quaility
    body_model_config.update(type='smpl')
    tensor = visualize_T_pose(
        num_frames=2,
        orbit_speed=(1.0, 0.5),
        body_model_config=body_model_config,
        render_choice='lq',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (2, 48, 48, 4)

    # render smpl mesh in high quaility
    body_model_config.update(type='smpl')
    tensor = visualize_T_pose(
        num_frames=2,
        orbit_speed=(1.0, 0.5),
        body_model_config=body_model_config,
        render_choice='hq',
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        batch_size=5,
        overwrite=True,
        return_tensor=True,
        device=device_name)
    assert tensor.shape == (2, 48, 48, 4)

    # test function visualize_smpl_calibration
    K = torch.zeros(1, 4, 4)
    K[:, 0, 0] = 1
    K[:, 1, 1] = 1
    K[:, 0, 2] = 1
    K[:, 1, 2] = 1
    R = torch.eye(3, 3)[None]
    T = torch.zeros(1, 3)
    body_model_config.update(type='smplx')
    visualize_smpl_calibration(
        poses=pose_dict,
        body_model_config=body_model_config,
        K=K,
        R=R,
        T=T,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)

    # test function visualize_smpl_calibration with betas and transl
    K = torch.zeros(1, 4, 4)
    K[:, 0, 0] = 1
    K[:, 1, 1] = 1
    K[:, 0, 2] = 1
    K[:, 1, 2] = 1
    R = torch.eye(3, 3)[None]
    T = torch.zeros(1, 3)
    betas = torch.zeros(2, 10)
    transl = torch.zeros(2, 3)
    body_model_config.update(type='smplx')
    visualize_smpl_calibration(
        poses=pose_dict,
        body_model_config=body_model_config,
        betas=betas,
        transl=transl,
        K=K,
        R=R,
        T=T,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)

    # use function visualize_smpl_hmr to render smplx mesh
    bbox = np.zeros((3, 1, 4))
    cam_transl = torch.zeros(3, 1, 3)
    body_model_config.update(type='smplx')
    visualize_smpl_hmr(
        poses=torch.zeros(3, 165),
        body_model_config=body_model_config,
        bbox=bbox,
        cam_transl=cam_transl,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)

    # render smpl mesh by passing verts
    bbox = np.zeros((3, 1, 4))
    cam_transl = torch.zeros(3, 1, 3)
    body_model_config.update(type='smpl')
    visualize_smpl_hmr(
        verts=torch.zeros(3, 6890, 3),
        body_model_config=body_model_config,
        bbox=bbox,
        cam_transl=cam_transl,
        output_path='/tmp/1.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)

    # render smplx mesh with background images from video
    bbox = np.zeros((3, 1, 4))
    T = torch.zeros(3, 1, 3)
    body_model_config.update(type='smplx')
    visualize_smpl_hmr(
        poses=torch.zeros(3, 165),
        body_model_config=body_model_config,
        bbox=bbox,
        end=2,
        cam_transl=cam_transl,
        origin_frames='/tmp/1.mp4',
        read_frames_batch=True,
        output_path='/tmp/2.mp4',
        resolution=(48, 48),
        overwrite=True,
        device=device_name)

    # render smplx mesh with background images from image folder
    image_array = np.random.randint(
        low=0, high=255, size=(3, 128, 128, 3), dtype=np.uint8)
    array_to_images(image_array, '/tmp/temp_images', img_format='%06d.png')

    bbox = np.zeros((3, 1, 4))
    cam_transl = torch.zeros(3, 1, 3)
    body_model_config.update(type='smplx')
    visualize_smpl_hmr(
        poses=torch.zeros(3, 165),
        body_model_config=body_model_config,
        bbox=bbox,
        cam_transl=cam_transl,
        output_path='/tmp/1.mp4',
        frame_list=['/tmp/temp_images/%06d.png' % 0] * 3,
        resolution=(48, 48),
        overwrite=True,
        device=device_name)

    # render smplx mesh with background images from video
    bbox = np.zeros((3, 1, 4))
    cam_transl = torch.zeros(3, 1, 3)
    body_model_config.update(type='smplx')
    visualize_smpl_hmr(
        poses=torch.zeros(3, 165),
        body_model_config=body_model_config,
        bbox=bbox,
        cam_transl=cam_transl,
        output_path='/tmp/1.mp4',
        img_format='%06d.png',
        image_array=image_array,
        resolution=(48, 48),
        overwrite=True,
        device=device_name)

    # render smplx mesh with specified palette from a numpy array
    body_model_config.update(type='smplx')
    visualize_smpl_hmr(
        poses=torch.zeros(3, 165),
        body_model_config=body_model_config,
        bbox=bbox,
        cam_transl=cam_transl,
        output_path='/tmp/1.mp4',
        img_format='%06d.png',
        origin_frames='/tmp/temp_images',
        resolution=(48, 48),
        overwrite=True,
        palette=np.ones((1, 3)),
        device=device_name)

    # render multi-person smplx mesh with specified palette from a numpy array
    body_model_config.update(type='smplx')
    visualize_smpl_hmr(
        poses=torch.zeros(3, 3, 165),
        body_model_config=body_model_config,
        bbox=np.zeros((3, 3, 4)),
        cam_transl=torch.zeros(3, 3, 3),
        output_path='/tmp/1.mp4',
        origin_frames='/tmp/temp_images',
        img_format='%06d.png',
        resolution=(128, 128),
        overwrite=True,
        palette=np.ones((1, 3)),
        device=device_name)

    # export the smplx mesh file as ply into the `mesh_file_path` folder
    body_model_config.update(type='smplx')
    visualize_smpl_hmr(
        poses=torch.zeros(3, 3, 165),
        body_model_config=body_model_config,
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

    # render multi-person smplx mesh with random palette from colormap
    body_model_config.update(type='smplx')
    visualize_smpl_hmr(
        poses=torch.zeros(3, 3, 165),
        body_model_config=body_model_config,
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

    # wrong palette, should be numpy or tensor of shape (N, 3) or a string
    # in pre-defined range
    body_model_config.update(type='smplx')
    with pytest.raises(ValueError):
        visualize_smpl_hmr(
            poses=torch.zeros(3, 3, 165),
            body_model_config=body_model_config,
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

    # wrong palette, should be numpy or tensor of shape (N, 3) or a string
    # in pre-defined range
    body_model_config.update(type='smplx')
    with pytest.raises(ValueError):
        visualize_smpl_hmr(
            poses=torch.zeros(3, 3, 165),
            body_model_config=body_model_config,
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

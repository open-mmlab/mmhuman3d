import mmcv
import numpy as np
import torch

from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.models.registrants.builder import build_registrant

body_model_load_dir = 'data/body_models'
batch_size = 2


def test_smplify():
    """Test adaptive batch size."""

    smplify_config = dict(mmcv.Config.fromfile('configs/smplify/smplify.py'))

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    smplify_config['body_model'] = dict(
        type='SMPL',
        gender='neutral',
        num_betas=10,
        keypoint_src='smpl_45',
        keypoint_dst='smpl_45',
        model_path='data/body_models/smpl',
        batch_size=1  # need not to know batch size at init
    )
    smplify_config['num_epochs'] = 1
    smplify_config['use_one_betas_per_video'] = True

    smplify = build_registrant(smplify_config)

    # Generate keypoints
    smpl = build_body_model(
        dict(
            type='SMPL',
            gender='neutral',
            num_betas=10,
            keypoint_src='smpl_45',
            keypoint_dst='smpl_45',
            model_path='data/body_models/smpl',
            batch_size=batch_size)  # keypoints shape: (2, 45, 3)
    )
    keypoints3d = smpl()['joints'].detach().to(device=device)
    keypoints3d_conf = torch.ones(*keypoints3d.shape[:2], device=device)

    # Run SMPLify
    smplify_output = smplify(
        keypoints3d=keypoints3d, keypoints3d_conf=keypoints3d_conf)

    for k, v in smplify_output.items():
        if isinstance(v, torch.Tensor):
            assert not np.any(np.isnan(
                v.detach().cpu().numpy())), f'{k} fails.'

    # Run SMPLify with init parameters
    smplify_output = smplify(
        keypoints3d=keypoints3d,
        keypoints3d_conf=keypoints3d_conf,
        init_global_orient=torch.rand([1, 3]).to(device),
        init_body_pose=torch.rand([1, 69]).to(device),
        init_betas=torch.rand([1, 10]).to(device),
        init_transl=torch.rand([1, 3]).to(device),
    )

    for k, v in smplify_output.items():
        if isinstance(v, torch.Tensor):
            assert not np.any(np.isnan(
                v.detach().cpu().numpy())), f'{k} fails.'


def test_smplifyx():
    smplifyx_config = dict(mmcv.Config.fromfile('configs/smplify/smplifyx.py'))

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    smplifyx_config['body_model'] = dict(
        type='SMPLX',
        gender='neutral',
        num_betas=10,
        use_face_contour=True,
        keypoint_src='smplx',
        keypoint_dst='smplx',
        model_path='data/body_models/smplx',
        batch_size=1  # need not to know batch size at init
    )
    smplifyx_config['num_epochs'] = 1
    smplifyx_config['use_one_betas_per_video'] = True

    smplifyx = build_registrant(smplifyx_config)

    smplx = build_body_model(
        dict(
            type='SMPLX',
            gender='neutral',
            num_betas=10,
            use_face_contour=True,
            keypoint_src='smplx',
            keypoint_dst='smplx',
            model_path='data/body_models/smplx',
            batch_size=batch_size)  # keypoints shape: (2, 144, 3)
    )
    keypoints3d = smplx()['joints'].detach().to(device=device)
    keypoints3d_conf = torch.ones(*keypoints3d.shape[:2], device=device)

    # Run SMPLify-X
    smplifyx_output = smplifyx(
        keypoints3d=keypoints3d, keypoints3d_conf=keypoints3d_conf)

    for k, v in smplifyx_output.items():
        if isinstance(v, torch.Tensor):
            assert not np.any(np.isnan(
                v.detach().cpu().numpy())), f'{k} fails.'

    # Run SMPLify-X with init parameters
    smplifyx_output = smplifyx(
        keypoints3d=keypoints3d,
        keypoints3d_conf=keypoints3d_conf,
        init_global_orient=torch.rand([1, 3]).to(device),
        init_transl=torch.rand([1, 3]).to(device),
        init_body_pose=torch.rand([1, 63]).to(device),
        init_betas=torch.rand([1, 10]).to(device),
        init_left_hand_pose=torch.rand([1, 6]).to(device),
        init_right_hand_pose=torch.rand([1, 6]).to(device),
        init_expression=torch.rand([1, 10]).to(device),
        init_jaw_pose=torch.rand([1, 3]).to(device),
        init_leye_pose=torch.rand([1, 3]).to(device),
        init_reye_pose=torch.rand([1, 3]).to(device))

    for k, v in smplifyx_output.items():
        if isinstance(v, torch.Tensor):
            assert not np.any(np.isnan(
                v.detach().cpu().numpy())), f'{k} fails.'

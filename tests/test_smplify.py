import mmcv
import numpy as np
import torch

from mmhuman3d.core.parametric_model.builder import build_registrant
from mmhuman3d.models.builder import build_body_model

body_model_load_dir = 'data/body_models'
batch_size = 2


def test_smpl():
    smplify_config = dict(mmcv.Config.fromfile('configs/smplify/smplify.py'))

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    body_model_config = dict(
        type='SMPL',
        gender='neutral',
        num_betas=10,
        keypoint_src='smpl_45',
        keypoint_dst='smpl_45',
        model_path='data/body_models/smpl',
        batch_size=batch_size)

    smplify_config['body_model'] = body_model_config
    smplify_config['num_epochs'] = 1
    smplify_config['use_one_betas_per_video'] = True
    smplify_config['num_videos'] = 1

    smplify = build_registrant(smplify_config)

    # Generate keypoints
    smpl = build_body_model(body_model_config)
    keypoints3d = smpl()['joints'].detach().to(device=device)
    keypoints3d_conf = torch.ones(*keypoints3d.shape[:2], device=device)

    # Run SMPLify
    smplify_output = smplify(
        keypoints3d=keypoints3d, keypoints3d_conf=keypoints3d_conf)

    for k, v in smplify_output.items():
        if isinstance(v, torch.Tensor):
            assert not np.any(np.isnan(
                v.detach().cpu().numpy())), f'{k} fails.'


def test_smplx():
    smplifyx_config = dict(mmcv.Config.fromfile('configs/smplify/smplifyx.py'))

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    body_model_config = dict(
        type='SMPLX',
        gender='neutral',
        num_betas=10,
        use_face_contour=True,
        keypoint_src='smplx',
        keypoint_dst='smplx',
        model_path='data/body_models/smplx',
        batch_size=batch_size)

    smplifyx_config['body_model'] = body_model_config
    smplifyx_config['num_epochs'] = 1
    smplifyx_config['use_one_betas_per_video'] = True
    smplifyx_config['num_videos'] = 1

    smplifyx = build_registrant(smplifyx_config)

    smplx = build_body_model(body_model_config)
    keypoints3d = smplx()['joints'].detach().to(device=device)
    keypoints3d_conf = torch.ones(*keypoints3d.shape[:2], device=device)

    # Run SMPLify-X
    smplifyx_output = smplifyx(
        keypoints3d=keypoints3d, keypoints3d_conf=keypoints3d_conf)

    for k, v in smplifyx_output.items():
        if isinstance(v, torch.Tensor):
            assert not np.any(np.isnan(
                v.detach().cpu().numpy())), f'{k} fails.'

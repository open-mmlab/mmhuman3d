import mmcv
import numpy as np
import smplx
import torch

from mmhuman3d.core.parametric_model.builder import build_registrant

body_model_load_dir = 'data/body_models'


def test_smpl():
    smplify_config = dict(mmcv.Config.fromfile('configs/smplify/smplify.py'))

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    smplify_config['num_epochs'] = 1
    smplify_config['use_one_betas_per_video'] = True

    smplify = build_registrant(smplify_config)

    # Generate keypoints
    body_model = smplx.create(
        body_model_load_dir, model_type='smpl', gender='neutral', num_betas=10)
    output = body_model()

    keypoints3d = output.joints.detach().to(device=device)
    keypoints3d_conf = torch.ones((keypoints3d.shape[1]), device=device)

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

    smplifyx_config['num_epochs'] = 1
    smplifyx_config['use_one_betas_per_video'] = True

    smplifyx = build_registrant(smplifyx_config)

    # Generate keypoints
    body_model = smplx.create(
        body_model_load_dir,
        model_type='smplx',
        gender='neutral',
        use_face_contour=True,
        num_betas=10)
    output = body_model()

    keypoints3d = output.joints.detach().to(device=device)
    keypoints3d_conf = torch.ones((keypoints3d.shape[1]), device=device)

    # Run SMPLify-X
    smplifyx_output = smplifyx(
        keypoints3d=keypoints3d, keypoints3d_conf=keypoints3d_conf)

    for k, v in smplifyx_output.items():
        if isinstance(v, torch.Tensor):
            assert not np.any(np.isnan(
                v.detach().cpu().numpy())), f'{k} fails.'

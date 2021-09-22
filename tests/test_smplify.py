import numpy as np
import smplx
import torch

from mmhuman3d.core.parametric_model.smplify import SMPLify, SMPLifyX

body_model_load_dir = 'body_models'
batch_size = 1


def test_smpl():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    smpl_body_model = smplx.create(
        body_model_load_dir,
        model_type='smpl',
        gender='neutral',
        num_betas=10,
        batch_size=batch_size,
    )

    smplify = SMPLify(
        body_model=smpl_body_model, use_one_betas_per_video=True, num_epochs=1)

    # Generate keypoints
    output = smpl_body_model()

    # TODO: support 24 smpl joints
    model_joints = output.joints[:, :24, :]

    keypoints3d = model_joints.detach().to(device=device)
    keypoints3d_conf = torch.ones((keypoints3d.shape[1]), device=device)

    # Run SMPLify
    smplify_output = smplify(
        keypoints3d=keypoints3d, keypoints3d_conf=keypoints3d_conf)

    for k, v in smplify_output.items():
        if isinstance(v, torch.Tensor):
            assert not np.any(np.isnan(
                v.detach().cpu().numpy())), f'{k} fails.'


def test_smplx():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    smpl_body_model = smplx.create(
        body_model_load_dir,
        model_type='smplx',
        gender='neutral',
        use_face_contour=True,  # 127 -> 144
        num_betas=10,
        batch_size=batch_size,
    )

    smplifyx = SMPLifyX(
        body_model=smpl_body_model, use_one_betas_per_video=True, num_epochs=1)

    # Generate keypoints
    output = smpl_body_model()
    keypoints3d = output.joints.detach().to(device=device)
    keypoints3d_conf = torch.ones((keypoints3d.shape[1]), device=device)

    # Run SMPLify-X
    smplifyx_output = smplifyx(
        keypoints3d=keypoints3d, keypoints3d_conf=keypoints3d_conf)

    for k, v in smplifyx_output.items():
        if isinstance(v, torch.Tensor):
            assert not np.any(np.isnan(
                v.detach().cpu().numpy())), f'{k} fails.'

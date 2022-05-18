import torch

from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.models.body_models.utils import (
    batch_transform_to_camera_frame,
    transform_to_camera_frame,
)
from mmhuman3d.utils.transforms import ee_to_rotmat


def test_transform_to_camera_frame():

    # initialize body model
    body_model = build_body_model(
        dict(
            type='SMPL',
            keypoint_src='smpl_45',
            keypoint_dst='smpl_45',
            model_path='data/body_models/smpl',
        ))

    # generate random values
    random_transl = torch.rand((1, 3))
    random_rotation = torch.rand((1, 3))
    random_rotmat = ee_to_rotmat(random_rotation)

    random_extrinsic = torch.eye(4)
    random_extrinsic[:3, :3] = random_rotmat
    random_extrinsic[:3, 3] = random_transl

    random_global_orient = torch.rand((1, 3))
    random_body_pose = torch.rand((1, 69))
    random_transl = torch.rand((1, 3))
    random_betas = torch.rand((1, 10))

    random_output = body_model(
        global_orient=random_global_orient,
        body_pose=random_body_pose,
        transl=random_transl,
        betas=random_betas)

    random_joints = random_output['joints']
    random_pelvis = random_joints[:, 0, :]

    # transform params
    transformed_global_orient, transformed_transl = \
        transform_to_camera_frame(
            global_orient=random_global_orient.numpy().squeeze(),  # (3, )
            transl=random_transl.numpy().squeeze(),  # (3, )
            pelvis=random_pelvis.numpy().squeeze(),  # (3, )
            extrinsic=random_extrinsic.numpy().squeeze()  # (4, 4)
        )

    transformed_output = body_model(
        global_orient=torch.tensor(transformed_global_orient.reshape(1, 3)),
        transl=torch.tensor(transformed_transl.reshape(1, 3)),
        body_pose=random_body_pose,
        betas=random_betas)

    transformed_joints = transformed_output['joints']

    # check validity
    random_joints = random_joints.squeeze()  # (45, 3)
    random_joints = torch.cat([random_joints, torch.ones(45, 1)],
                              dim=1)  # (45, 4)
    test_joints = torch.einsum('ij,kj->ki', random_extrinsic,
                               random_joints)  # (45, 4)
    test_joints = test_joints[:, :3]  # (45, 3)
    assert torch.allclose(transformed_joints, test_joints, atol=1e-6)


def test_batch_transform_to_camera_frame():
    # batch size
    N = 2

    # initialize body model
    body_model = build_body_model(
        dict(
            type='SMPL',
            keypoint_src='smpl_45',
            keypoint_dst='smpl_45',
            model_path='data/body_models/smpl',
        ))

    # generate random values
    random_transl = torch.rand((1, 3))
    random_rotation = torch.rand((1, 3))
    random_rotmat = ee_to_rotmat(random_rotation)

    random_extrinsic = torch.eye(4)
    random_extrinsic[:3, :3] = random_rotmat
    random_extrinsic[:3, 3] = random_transl

    random_global_orient = torch.rand((N, 3))
    random_body_pose = torch.rand((N, 69))
    random_transl = torch.rand((N, 3))
    random_betas = torch.rand((N, 10))

    random_output = body_model(
        global_orient=random_global_orient,
        body_pose=random_body_pose,
        transl=random_transl,
        betas=random_betas)

    random_joints = random_output['joints']
    random_pelvis = random_joints[:, 0, :]

    # transform params
    transformed_global_orient, transformed_transl = \
        batch_transform_to_camera_frame(
            global_orient=random_global_orient.numpy(),  # (N, 3)
            transl=random_transl.numpy(),  # (N, 3)
            pelvis=random_pelvis.numpy(),  # (N, 3)
            extrinsic=random_extrinsic.numpy()  # (4, 4)
        )

    transformed_output = body_model(
        global_orient=torch.tensor(transformed_global_orient.reshape(N, 3)),
        transl=torch.tensor(transformed_transl.reshape(N, 3)),
        body_pose=random_body_pose,
        betas=random_betas)

    transformed_joints = transformed_output['joints']

    # check validity
    random_joints = random_joints  # (N, 45, 3)
    random_joints = torch.cat(
        [random_joints, torch.ones(N, 45, 1)], dim=2)  # (N, 45, 4)
    test_joints = torch.einsum('ij,bkj->bki', random_extrinsic,
                               random_joints)  # (N, 45, 4)
    test_joints = test_joints[:, :, :3]  # (N, 45, 3)
    assert torch.allclose(transformed_joints, test_joints, atol=1e-6)

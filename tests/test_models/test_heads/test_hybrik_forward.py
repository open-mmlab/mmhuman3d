import os
import os.path as osp
import tempfile

import numpy as np
import pytest
import torch

from mmhuman3d.models.architectures.hybrik import HybrIK_trainer
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.models.heads.hybrik_head import HybrIKHead
from mmhuman3d.models.utils.inverse_kinematics import (
    batch_get_3children_orient_svd,
    batch_get_pelvis_orient,
    batch_get_pelvis_orient_svd,
    batch_inverse_kinematics_transform,
)


def generate_weights(output_dir):
    """Generate a SMPL model weight file to initialize SMPL model, and generate
    a 3D joints regressor file."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    joint_regressor_file = os.path.join(output_dir, 'J_regressor_h36m.npy')
    np.save(joint_regressor_file, np.zeros([17, 6890]))

    smpl_mean_file = os.path.join(output_dir, 'h36m_mean_beta.npy')
    np.save(smpl_mean_file, np.zeros([
        10,
    ]))
    return


def test_HybrIK_head():

    tmpdir = tempfile.TemporaryDirectory()
    # generate weight file for SMPL model.
    generate_weights(tmpdir.name)

    # initialize models
    head = HybrIKHead(
        smpl_mean_params=osp.join(tmpdir.name, 'h36m_mean_beta.npy'))
    smpl = build_body_model(
        dict(
            type='HybrIKSMPL',
            model_path='data/body_models/smpl',
            extra_joints_regressor=osp.join(tmpdir.name,
                                            'J_regressor_h36m.npy')))

    if torch.cuda.is_available():
        head = head.cuda()
        smpl = smpl.cuda()

    with pytest.raises(TypeError):
        _ = HybrIKHead()

    with pytest.raises(TypeError):
        _ = HybrIKHead(
            feature_channel=[512, 8],
            smpl_mean_params='data/body_models/h36m_mean_beta.npy')

    # mock inputs
    batch_size = 4
    input_shape = (batch_size, 512, 8, 8)
    mm_inputs = _demo_head_inputs(input_shape)
    features = mm_inputs.pop('features')
    trans_inv = mm_inputs.pop('trans_inv')
    joint_root = mm_inputs.pop('joint_root')
    depth_factor = mm_inputs.pop('depth_factor')
    intrinsic_param = mm_inputs.pop('intrinsic_param')

    if torch.cuda.is_available():
        predictions = head(features, trans_inv, intrinsic_param, joint_root,
                           depth_factor, smpl)
        pred_keys = [
            'pred_phi', 'pred_delta_shape', 'pred_shape', 'pred_pose',
            'pred_uvd_jts', 'pred_xyz_jts_24', 'pred_xyz_jts_24_struct',
            'pred_xyz_jts_17', 'pred_vertices', 'maxvals'
        ]
        for k in pred_keys:
            assert k in predictions
            assert predictions[k].shape[0] == batch_size

        with pytest.raises(RuntimeError):
            joint_root = torch.zeros((6, 3)).cuda()
            _ = head(features, trans_inv, intrinsic_param, joint_root,
                     depth_factor, smpl)

        with pytest.raises(RuntimeError):
            joint_root = torch.zeros((batch_size, 3))
            _ = head(features, trans_inv, intrinsic_param, joint_root,
                     depth_factor, smpl)

    tmpdir.cleanup()


def test_HybrIK_trainer():

    tmpdir = tempfile.TemporaryDirectory()
    # generate weight file for SMPL model.
    generate_weights(tmpdir.name)

    model_cfg = dict(
        backbone=dict(
            type='ResNet',
            depth=34,
            out_indices=[3],
            norm_eval=False,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet34')),
        head=dict(
            type='HybrIKHead',
            smpl_mean_params=osp.join(tmpdir.name, 'h36m_mean_beta.npy')),
        body_model=dict(
            type='HybrIKSMPL',
            model_path=  # noqa: E251
            'data/body_models/smpl',
            extra_joints_regressor=osp.join(tmpdir.name,
                                            'J_regressor_h36m.npy')),
        loss_beta=dict(type='MSELoss', loss_weight=1),
        loss_theta=dict(type='MSELoss', loss_weight=0.01),
        loss_twist=dict(type='MSELoss', loss_weight=0.01),
        loss_uvd=dict(type='L1Loss', loss_weight=1),
    )

    model = HybrIK_trainer(**model_cfg)
    if torch.cuda.is_available():
        model = model.cuda()
    input_shape = (4, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape)
    img = mm_inputs.pop('img')
    img_metas = mm_inputs.pop('img_metas')
    if torch.cuda.is_available():
        output = model.forward_train(img, img_metas, **mm_inputs)
        assert isinstance(output, dict)
        assert 'loss_beta' in output
        assert output['loss_beta'].dtype == torch.float32

        with torch.no_grad():
            output = model.forward_test(img, img_metas, **mm_inputs)
            assert isinstance(output, dict)
            for k in ['vertices', 'xyz_17', 'uvd_jts', 'xyz_24', 'image_path']:
                assert k in output

    tmpdir.cleanup()


def test_IK_functions():

    N = 4
    mm_inputs = _demo_IK_inputs(N)

    pose_skeleton = mm_inputs['pose_skeleton']
    phis = mm_inputs['phis']
    rest_pose = mm_inputs['rest_pose']
    children = mm_inputs['children']
    parents = mm_inputs['parents']
    rel_pose_skeleton = mm_inputs['rel_pose_skeleton']
    rel_rest_pose = mm_inputs['rel_rest_pose']
    rot_mat_chain_parent = mm_inputs['rot_mat_chain_parent']
    global_orient = None
    dtype = torch.float32

    rot_mat, rot_rest_pose = batch_inverse_kinematics_transform(
        pose_skeleton,
        global_orient,
        phis,
        rest_pose,
        children,
        parents,
        dtype,
        train=False,
        leaf_thetas=None)
    assert rot_mat.shape == (N, 24, 3, 3)
    assert rot_rest_pose.shape == (N, 29, 3)

    rot_mat, rot_rest_pose = batch_inverse_kinematics_transform(
        pose_skeleton,
        global_orient,
        phis,
        rest_pose,
        children,
        parents,
        dtype,
        train=True,
        leaf_thetas=None)
    assert rot_mat.shape == (N, 24, 3, 3)
    assert rot_rest_pose.shape == (N, 29, 3)

    global_orient_mat = batch_get_pelvis_orient(rel_pose_skeleton.clone(),
                                                rel_rest_pose.clone(), parents,
                                                children, dtype)
    assert global_orient_mat.shape == (N, 3, 3)

    global_orient_mat = batch_get_pelvis_orient_svd(rel_pose_skeleton.clone(),
                                                    rel_rest_pose.clone(),
                                                    parents, children, dtype)
    assert global_orient_mat.shape == (N, 3, 3)

    rot_mat = batch_get_3children_orient_svd(rel_pose_skeleton, rel_rest_pose,
                                             rot_mat_chain_parent, children,
                                             dtype)
    assert rot_mat.shape == (N, 3, 3)


def _demo_mm_inputs(input_shape=(1, 3, 256, 256)):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    trans_inv = np.zeros([N, 2, 3])
    intrinsic_param = np.zeros([N, 3, 3])
    joint_root = np.zeros([N, 3])
    depth_factor = np.ones([N, 1])
    target_uvd_29 = np.zeros([N, 87])
    target_xyz_24 = np.zeros([N, 72])
    target_weight_24 = np.ones([N, 72])
    target_weight_29 = np.ones([N, 87])
    target_xyz_17 = np.zeros([N, 51])
    target_weight_17 = np.ones([N, 51])
    target_theta = np.zeros([N, 96])
    target_beta = np.zeros([N, 10])
    target_smpl_weight = np.ones([N, 1])
    target_theta_weight = np.ones([N, 96])
    target_twist = np.zeros([N, 23, 2])
    target_twist_weight = np.ones([N, 23, 2])
    bbox = np.zeros([N, 4])

    img_metas = [{
        'img_shape': (H, W, C),
        'center': np.array([W / 2, H / 2]),
        'scale': np.array([0.5, 0.5]),
        'rotation': 0,
        'image_path': '<demo>.png',
    } for _ in range(N)]

    mm_inputs = {
        'img': torch.FloatTensor(imgs).requires_grad_(True),
        'trans_inv': torch.FloatTensor(trans_inv),
        'intrinsic_param': torch.FloatTensor(intrinsic_param),
        'joint_root': torch.FloatTensor(joint_root),
        'depth_factor': torch.FloatTensor(depth_factor),
        'target_uvd_29': torch.FloatTensor(target_uvd_29),
        'target_xyz_24': torch.FloatTensor(target_xyz_24),
        'target_weight_24': torch.FloatTensor(target_weight_24),
        'target_weight_29': torch.FloatTensor(target_weight_29),
        'target_xyz_17': torch.FloatTensor(target_xyz_17),
        'target_weight_17': torch.FloatTensor(target_weight_17),
        'target_theta': torch.FloatTensor(target_theta),
        'target_beta': torch.FloatTensor(target_beta),
        'target_smpl_weight': torch.FloatTensor(target_smpl_weight),
        'target_theta_weight': torch.FloatTensor(target_theta_weight),
        'target_twist': torch.FloatTensor(target_twist),
        'target_twist_weight': torch.FloatTensor(target_twist_weight),
        'bbox': torch.FloatTensor(bbox),
        'img_metas': img_metas,
        'sample_idx': np.arange(N)
    }

    return mm_inputs


def _demo_head_inputs(input_shape=(1, 512, 8, 8)):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    features = rng.rand(*input_shape)

    trans_inv = np.zeros([N, 2, 3])
    intrinsic_param = np.zeros([N, 3, 3])
    joint_root = np.zeros([N, 3])
    depth_factor = np.ones([N, 1])

    mm_inputs = {
        'features': torch.FloatTensor(features),
        'trans_inv': torch.FloatTensor(trans_inv),
        'intrinsic_param': torch.FloatTensor(intrinsic_param),
        'joint_root': torch.FloatTensor(joint_root),
        'depth_factor': torch.FloatTensor(depth_factor),
    }

    if torch.cuda.is_available():
        for k, _ in mm_inputs.items():
            mm_inputs[k] = mm_inputs[k].cuda()

    return mm_inputs


def _demo_IK_inputs(batch_size=1):
    """Create a superset of inputs for testing inverse kinematics function.

    Args:
        batch_size (int):
            input batch size
    """
    N = batch_size
    pose_skeleton = np.ones([N, 29, 3])
    phis = np.ones([N, 23, 2])
    rest_pose = np.ones([N, 29, 3])
    rel_pose_skeleton = np.ones([N, 29, 3, 1])
    rel_rest_pose = np.ones([N, 29, 3, 1])
    parents = np.array([
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18,
        19, 20, 21, 15, 22, 23, 10, 11
    ])
    children = np.array([
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 27, 28, 15, 16, 17, 24, 18, 19, 20,
        21, 22, 23, 25, 26, -1, -1, -1, -1, -1
    ])
    rot_mat_chain_parent = np.ones([N, 3, 3])

    mm_inputs = {
        'pose_skeleton': torch.FloatTensor(pose_skeleton),
        'phis': torch.FloatTensor(phis),
        'rest_pose': torch.FloatTensor(rest_pose),
        'children': torch.Tensor(children).long(),
        'parents': torch.Tensor(parents).long(),
        'rel_pose_skeleton': torch.FloatTensor(rel_pose_skeleton),
        'rel_rest_pose': torch.FloatTensor(rel_rest_pose),
        'rot_mat_chain_parent': torch.FloatTensor(rot_mat_chain_parent),
    }

    if torch.cuda.is_available():
        for k, _ in mm_inputs.items():
            mm_inputs[k] = mm_inputs[k].cuda()

    return mm_inputs

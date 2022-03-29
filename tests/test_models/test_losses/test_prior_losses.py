import torch

from mmhuman3d.models import build_loss


def test_shape_prior_loss():
    loss_config = dict(type='ShapePriorLoss', reduction='mean')
    loss = build_loss(loss_config)
    betas = torch.rand(2, 10)
    output = loss(betas)
    assert isinstance(output, torch.Tensor) and \
           output.size() == ()


def test_joint_prior_loss():
    loss_config = dict(
        type='JointPriorLoss', use_full_body=False, reduction='mean')
    loss = build_loss(loss_config)
    body_pose = torch.rand(2, 69)
    output = loss(body_pose)
    assert isinstance(output, torch.Tensor) and \
           output.size() == ()


def test_smooth_joint_loss():
    loss_config = dict(type='SmoothJointLoss', reduction='mean')
    loss = build_loss(loss_config)
    body_pose = torch.rand(2, 69)
    output = loss(body_pose)
    assert isinstance(output, torch.Tensor) and \
           output.size() == ()


def test_smooth_pelvis_loss():
    loss_config = dict(type='SmoothPelvisLoss', reduction='mean')
    loss = build_loss(loss_config)
    global_orient = torch.rand(2, 3)
    output = loss(global_orient)
    assert isinstance(output, torch.Tensor) and \
           output.size() == ()


def test_smooth_translation_loss():
    loss_config = dict(type='SmoothTranslationLoss', reduction='mean')
    loss = build_loss(loss_config)
    transl = torch.rand(2, 3)
    output = loss(transl)
    assert isinstance(output, torch.Tensor) and \
           output.size() == ()


def test_max_mixture_prior_loss():
    loss_config = dict(type='MaxMixturePrior', reduction='mean')
    loss = build_loss(loss_config)
    body_pose = torch.rand(2, 69)
    output = loss(body_pose)
    assert isinstance(output, torch.Tensor) and \
           output.size() == ()


def test_limb_length_loss():
    loss_cfg = dict(type='LimbLengthLoss', convention='smpl')
    loss = build_loss(loss_cfg)
    pred = torch.zeros(1, 24, 3)
    target = torch.zeros(1, 24, 3)
    pred_conf = torch.ones(1, 24)
    target_conf = torch.ones(1, 24)
    # test without conf
    loss(pred, target)
    # test with pred_conf
    loss(pred, target, keypoints3d_pred_conf=pred_conf)
    # test with target conf
    loss(pred, target, keypoints3d_target_conf=target_conf)
    # test with pred and target conf
    loss(
        pred,
        target,
        keypoints3d_pred_conf=pred_conf,
        keypoints3d_target_conf=target_conf)


def test_pose_reg_loss():
    loss_cfg = dict(type='PoseRegLoss')
    loss = build_loss(loss_cfg)
    body_pose = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(body_pose), torch.tensor(0.))

    body_pose = torch.ones(1, 3, 2)
    assert torch.allclose(loss(body_pose), torch.tensor(1.))

    # test sum reduction
    loss_cfg = dict(type='PoseRegLoss', reduction='sum')
    loss = build_loss(loss_cfg)
    body_pose = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(body_pose), torch.tensor(0.))

    body_pose = torch.ones(1, 3, 2)
    assert torch.allclose(loss(body_pose), torch.tensor(6.))

    # test None reduction
    loss_cfg = dict(type='PoseRegLoss', reduction=None)
    loss = build_loss(loss_cfg)
    body_pose = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(body_pose), body_pose)

    body_pose = torch.ones(1, 3, 2)
    assert torch.allclose(loss(body_pose), body_pose)

    # test loss weight
    loss_cfg = dict(type='PoseRegLoss', loss_weight=2.)
    loss = build_loss(loss_cfg)
    body_pose = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(body_pose), torch.tensor(0.))

    body_pose = torch.ones(1, 3, 2)
    assert torch.allclose(loss(body_pose), torch.tensor(2.))


test_limb_length_loss()

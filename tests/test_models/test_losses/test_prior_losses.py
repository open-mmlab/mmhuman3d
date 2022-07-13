import torch

from mmhuman3d.models.losses.builder import build_loss


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

    loss_config = dict(
        type='SmoothJointLoss', reduction='mean', loss_func='L2')
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
    K = 24  # the number of keypoints of SMPL
    M = 25  # the number of limbs of SMPL
    loss = build_loss(loss_cfg)
    pred = torch.zeros(1, K, 3)
    target = torch.zeros(1, K, 3)
    pred_conf = torch.ones(1, K)
    target_conf = torch.ones(1, K)
    odd_idxs = torch.arange(1, K, 2)
    pred_conf[:, odd_idxs] = 0
    target_conf[:, ~odd_idxs] = 0
    assert torch.allclose(pred_conf + target_conf, torch.ones(1, K))
    # test without conf
    assert torch.allclose(loss(pred, target), torch.tensor(0.))
    # test with pred_conf
    assert torch.allclose(
        loss(pred, target, pred_conf=pred_conf), torch.tensor(0.))
    # test with target conf
    assert torch.allclose(
        loss(pred, target, target_conf=target_conf), torch.tensor(0.))
    # test with pred and target conf
    assert torch.allclose(
        loss(pred, target, pred_conf=pred_conf, target_conf=target_conf),
        torch.tensor(0.))

    pred = torch.zeros(1, K, 3)
    target = torch.ones(1, K, 3)
    # test without conf
    assert torch.allclose(loss(pred, target), torch.tensor(0.))
    # test with pred_conf
    assert torch.allclose(
        loss(pred, target, pred_conf=pred_conf), torch.tensor(0.))
    # test with target conf
    assert torch.allclose(
        loss(pred, target, target_conf=target_conf), torch.tensor(0.))
    # test with pred and target conf
    assert torch.allclose(
        loss(pred, target, pred_conf=pred_conf, target_conf=target_conf),
        torch.tensor(0.))

    pred = torch.zeros(1, K, 3)
    target = torch.zeros(1, K, 3)
    # fake a keypoints where only pelvis is (1, 1, 1)
    target[:, 0, :] = 1.  # pelvis
    # 0-pelvis connects to 1-left_hip, 2-right_hip, 3-spine1
    # test without conf (3 limbs are valid)
    assert torch.allclose(loss(pred, target), torch.tensor(3. * 3 / M))
    # test with pred_conf (only pelvis-right_hip is valid)
    assert torch.allclose(
        loss(pred, target, pred_conf=pred_conf), torch.tensor(3. * 1 / M))
    # test with target conf (3 limbs are invalid, as conf of pelvis is 0)
    assert torch.allclose(
        loss(pred, target, target_conf=target_conf), torch.tensor(0.))
    # test with pred and target conf
    assert torch.allclose(
        loss(pred, target, pred_conf=pred_conf, target_conf=target_conf),
        torch.tensor(0.))


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


def test_shape_threshold_prior_loss():
    loss_cfg = dict(type='ShapeThresholdPriorLoss', loss_weight=1.)
    loss = build_loss(loss_cfg)
    shape = torch.zeros(1, 10)
    assert torch.allclose(loss(shape), torch.tensor(0.))

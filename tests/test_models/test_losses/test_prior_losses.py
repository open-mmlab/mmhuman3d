import torch

from mmhuman3d.models import build_loss

batch_size = 2


def test_shape_prior_loss():
    loss_config = dict(type='ShapePriorLoss', reduction='mean')
    loss = build_loss(loss_config)
    betas = torch.rand((batch_size, 10))
    output = loss(betas)
    assert isinstance(output, torch.Tensor) and \
           output.size() == ()


def test_joint_prior_loss():
    loss_config = dict(
        type='JointPriorLoss', use_full_body=False, reduction='mean')
    loss = build_loss(loss_config)
    body_pose = torch.rand((batch_size, 69))
    output = loss(body_pose)
    assert isinstance(output, torch.Tensor) and \
           output.size() == ()


def test_smooth_joint_loss():
    loss_config = dict(type='SmoothJointLoss', reduction='mean')
    loss = build_loss(loss_config)
    body_pose = torch.rand((batch_size, 69))
    output = loss(body_pose)
    assert isinstance(output, torch.Tensor) and \
           output.size() == ()


def test_smooth_pelvis_loss():
    loss_config = dict(type='SmoothPelvisLoss', reduction='mean')
    loss = build_loss(loss_config)
    global_orient = torch.rand((batch_size, 3))
    output = loss(global_orient)
    assert isinstance(output, torch.Tensor) and \
           output.size() == ()


def test_smooth_translation_loss():
    loss_config = dict(type='SmoothTranslationLoss', reduction='mean')
    loss = build_loss(loss_config)
    transl = torch.rand((batch_size, 3))
    output = loss(transl)
    assert isinstance(output, torch.Tensor) and \
           output.size() == ()


def test_max_mixture_prior_loss():
    loss_config = dict(type='MaxMixturePrior', reduction='mean')
    loss = build_loss(loss_config)
    body_pose = torch.rand((batch_size, 69))
    output = loss(body_pose)
    assert isinstance(output, torch.Tensor) and \
           output.size() == ()


if __name__ == '__main__':
    test_shape_prior_loss()
    test_joint_prior_loss()
    test_smooth_joint_loss()
    test_smooth_pelvis_loss()
    test_smooth_translation_loss()
    test_max_mixture_prior_loss()

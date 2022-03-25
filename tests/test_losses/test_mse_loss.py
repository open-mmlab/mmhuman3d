import torch

from mmhuman3d.models import build_loss


def test_mse_loss():
    loss_cfg = dict(type='MSELoss')
    loss = build_loss(loss_cfg)
    fake_pred = torch.zeros((1, 3, 3))
    fake_target = torch.zeros((1, 3, 3))
    assert torch.allclose(loss(fake_pred, fake_target), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 3))
    fake_target = torch.zeros((1, 3, 3))
    assert torch.allclose(loss(fake_pred, fake_target), torch.tensor(1.))


def test_keypoint_mse__loss():
    loss_cfg = dict(type='KeypointMSELoss')
    loss = build_loss(loss_cfg)
    fake_pred = torch.zeros((1, 3, 3))
    fake_target = torch.zeros((1, 3, 3))
    assert torch.allclose(loss(fake_pred, fake_target), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 3))
    fake_target = torch.zeros((1, 3, 3))
    assert torch.allclose(loss(fake_pred, fake_target), torch.tensor(.5))


def test_pose_reg_loss():
    loss_cfg = dict(type='PoseRegLoss')
    loss = build_loss(loss_cfg)
    fake_body_pose = torch.zeros((1, 3, 3))
    assert torch.allclose(loss(fake_body_pose), torch.tensor(0.))

    fake_body_pose = torch.ones((1, 3, 3))
    assert torch.allclose(loss(fake_body_pose), torch.tensor(1.))

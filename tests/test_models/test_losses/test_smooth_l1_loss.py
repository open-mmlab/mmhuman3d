import torch

from mmhuman3d.models.losses.builder import build_loss


def test_smooth_l1_loss():
    loss_cfg = dict(type='SmoothL1Loss')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros(1, 3, 2)
    fake_target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(fake_pred, fake_target), torch.tensor(0.))

    fake_pred = torch.ones(1, 3, 2)
    fake_target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(fake_pred, fake_target), torch.tensor(.5))

    # test beta
    loss_cfg = dict(type='SmoothL1Loss', beta=2.)
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros(1, 3, 2)
    fake_target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(fake_pred, fake_target), torch.tensor(0.))

    fake_pred = torch.ones(1, 3, 2)
    fake_target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(fake_pred, fake_target), torch.tensor(.25))

    # test reduction
    loss_cfg = dict(type='SmoothL1Loss', reduction='sum')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros(1, 3, 2)
    fake_target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(fake_pred, fake_target), torch.tensor(0.))

    fake_pred = torch.ones(1, 3, 2)
    fake_target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(fake_pred, fake_target), torch.tensor(3.))

    # test loss weight
    loss_cfg = dict(type='SmoothL1Loss', loss_weight=2.)
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros(1, 3, 2)
    fake_target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(fake_pred, fake_target), torch.tensor(0.))

    fake_pred = torch.ones(1, 3, 2)
    fake_target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(fake_pred, fake_target), torch.tensor(1.))

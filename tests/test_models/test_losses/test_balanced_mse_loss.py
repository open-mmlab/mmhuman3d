import torch

from mmhuman3d.models.losses.builder import build_loss


def test_balanced_mse_loss():
    loss_cfg = dict(type='BMCLossMD')
    loss = build_loss(loss_cfg)
    pred = torch.zeros(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(pred, target), torch.tensor(0.))

    loss_cfg = dict(type='BMCLossMD')
    loss = build_loss(loss_cfg)
    pred = torch.ones(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(pred, target), torch.tensor(0.5))

    loss_cfg = dict(type='BMCLossMD', init_noise_sigma=2.0)
    loss = build_loss(loss_cfg)
    pred = torch.ones(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(pred, target), torch.tensor(0.5))

    loss_cfg = dict(type='BMCLossMD', loss_mse_weight=2.0)
    loss = build_loss(loss_cfg)
    pred = torch.ones(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(pred, target), torch.tensor(1.5))

    loss_cfg = dict(type='BMCLossMD', loss_debias_weight=2.0)
    loss = build_loss(loss_cfg)
    pred = torch.ones(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(pred, target), torch.tensor(0.))

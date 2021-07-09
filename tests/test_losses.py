import pytest
import torch

from mmhuman3d.models import build_loss


def test_cross_entropy_loss():
    with pytest.raises(AssertionError):
        # use_sigmoid and use_soft could not be set simultaneously
        loss_cfg = dict(
            type='CrossEntropyLoss', use_sigmoid=True, use_soft=True)
        loss = build_loss(loss_cfg)

    # test ce_loss
    cls_score = torch.Tensor([[100, -100]])
    label = torch.Tensor([1]).long()
    weight = torch.tensor(0.5)

    loss_cfg = dict(type='CrossEntropyLoss', reduction='mean', loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(200.))
    # test ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(100.))

    # test bce_loss
    cls_score = torch.Tensor([[100, -100], [100, -100]])
    label = torch.Tensor([[1, 0], [0, 1]])
    weight = torch.Tensor([0.5, 0.5])

    loss_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(50.))
    # test ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(25.))

    # test soft_ce_loss
    cls_score = torch.Tensor([[100, -100]])
    label = torch.Tensor([[1, 0], [0, 1]])
    weight = torch.tensor(0.5)

    loss_cfg = dict(
        type='CrossEntropyLoss',
        use_soft=True,
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(100.))
    # test soft_ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(50.))

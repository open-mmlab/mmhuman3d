import torch

from mmhuman3d.models.losses.builder import build_loss


def test_keypoint_mse_loss():
    loss_cfg = dict(type='KeypointMSELoss')
    loss = build_loss(loss_cfg)
    pred = torch.zeros(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(pred, target), torch.tensor(0.))

    pred = torch.ones(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(pred, target), torch.tensor(.5))

    # test sum reduction
    loss_cfg = dict(type='KeypointMSELoss', reduction='sum')
    loss = build_loss(loss_cfg)
    pred = torch.zeros(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(pred, target), torch.tensor(0.))

    pred = torch.ones(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(pred, target), torch.tensor(3.))

    # test None reduction
    loss_cfg = dict(type='KeypointMSELoss', reduction=None)
    loss = build_loss(loss_cfg)
    pred = torch.zeros(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(pred, target), pred)

    pred = torch.ones(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    result = torch.ones(1, 3, 2) * 0.5
    assert torch.allclose(loss(pred, target), result)

    # test None reduction
    loss_cfg = dict(type='KeypointMSELoss', reduction='none')
    loss = build_loss(loss_cfg)
    pred = torch.zeros(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(pred, target), pred)

    pred = torch.ones(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    result = torch.ones(1, 3, 2) * 0.5
    assert torch.allclose(loss(pred, target), result)

    # test loss weight
    loss_cfg = dict(type='KeypointMSELoss', loss_weight=2.)
    loss = build_loss(loss_cfg)
    pred = torch.zeros(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(pred, target), torch.tensor(0.))

    pred = torch.ones(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    assert torch.allclose(loss(pred, target), torch.tensor(1.))

    # test keypoint weight
    loss_cfg = dict(type='KeypointMSELoss', keypoint_weight=[1.0, 0.0])
    loss = build_loss(loss_cfg)
    pred = torch.Tensor([[[1, 1], [2, 2]]])
    target = torch.zeros(1, 2, 2)
    print(loss(pred, target))
    assert torch.allclose(loss(pred, target), torch.tensor(0.25))

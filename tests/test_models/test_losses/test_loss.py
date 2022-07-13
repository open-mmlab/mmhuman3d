import pytest
import torch

from mmhuman3d.models.losses.builder import (
    CrossEntropyLoss,
    L1Loss,
    MSELoss,
    RotationDistance,
)


@pytest.mark.parametrize('loss_class', [MSELoss, L1Loss, CrossEntropyLoss])
def test_loss_with_reduction_override(loss_class):
    pred = torch.rand((10, 3))
    target = torch.rand((10, 3)),
    weight = None

    with pytest.raises(AssertionError):
        # only reduction_override from [None, 'none', 'mean', 'sum']
        # is not allowed
        reduction_override = True
        loss_class()(
            pred, target, weight, reduction_override=reduction_override)


@pytest.mark.parametrize('loss_class', [MSELoss, L1Loss])
def test_regression_losses(loss_class):
    pred = torch.rand((10, 3))
    target = torch.rand((10, 3))
    weight = torch.rand((10, 3))

    # Test loss forward
    loss = loss_class()(pred, target)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with weight
    loss = loss_class()(pred, target, weight)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with reduction_override
    loss = loss_class()(pred, target, reduction_override='mean')
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with avg_factor
    loss = loss_class()(pred, target, avg_factor=10)
    assert isinstance(loss, torch.Tensor)

    with pytest.raises(ValueError):
        # loss can evaluate with avg_factor only if
        # reduction is None, 'none' or 'mean'.
        reduction_override = 'sum'
        loss_class()(
            pred, target, avg_factor=10, reduction_override=reduction_override)

    # Test loss forward with avg_factor and reduction
    for reduction_override in [None, 'none', 'mean']:
        loss_class()(
            pred, target, avg_factor=10, reduction_override=reduction_override)
        assert isinstance(loss, torch.Tensor)


@pytest.mark.parametrize('use_sigmoid', [True, False])
@pytest.mark.parametrize('reduction', ['sum', 'mean', None])
def test_loss_with_ignore_index(use_sigmoid, reduction):
    # Test cross_entropy loss

    loss_class = CrossEntropyLoss(
        use_sigmoid=use_sigmoid,
        use_mask=False,
        ignore_index=255,
    )
    pred = torch.rand((10, 5))
    target = torch.randint(0, 5, (10, ))

    ignored_indices = torch.randint(0, 10, (2, ), dtype=torch.long)
    target[ignored_indices] = 255

    # Test loss forward with default ignore
    loss_with_ignore = loss_class(pred, target, reduction_override=reduction)
    assert isinstance(loss_with_ignore, torch.Tensor)

    # Test loss forward with forward ignore
    target[ignored_indices] = 255
    loss_with_forward_ignore = loss_class(
        pred, target, ignore_index=255, reduction_override=reduction)
    assert isinstance(loss_with_forward_ignore, torch.Tensor)

    # Verify correctness

    loss = loss_class(pred, target, reduction_override=reduction)

    assert torch.allclose(loss, loss_with_ignore)
    assert torch.allclose(loss, loss_with_forward_ignore)

    # test ignore all target
    pred = torch.rand((10, 5))
    target = torch.ones((10, ), dtype=torch.long) * 255
    loss = loss_class(pred, target, reduction_override=reduction)
    assert loss == 0


@pytest.mark.parametrize('loss_class', [RotationDistance])
def test_rotation_distance_losses(loss_class):
    pred = torch.rand((10, 3, 3))
    target = torch.rand((10, 3, 3))
    weight = torch.rand((10, 3, 3))

    # Test loss forward
    loss = loss_class()(pred, target)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with weight
    loss = loss_class()(pred, target, weight)
    assert isinstance(loss, torch.Tensor)

    target = torch.eye(3).reshape(1, 3, 3)
    pred = target.clone()
    assert loss_class()(pred, target) < torch.tensor(0.1)

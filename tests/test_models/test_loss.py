import pytest
import torch

from mmhuman3d.models.losses import L1Loss, MSELoss


@pytest.mark.parametrize('loss_class', [MSELoss, L1Loss])
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

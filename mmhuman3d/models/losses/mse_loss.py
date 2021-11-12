import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


def gmof(x, sigma):
    """Geman-McClure error function."""
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


@weighted_loss
def mse_loss(pred, target):
    """Warpper of mse loss."""
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss_with_gmof(pred, target, sigma):
    """Extended MSE Loss with GMOF."""
    loss = F.mse_loss(pred, target, reduction='none')
    loss = gmof(loss, sigma)
    return loss


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * mse_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss


@LOSSES.register_module()
class KeypointMSELoss(nn.Module):
    """MSELoss for 2D and 3D keypoints.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
        sigma (float, optional): Weighing parameter of Geman-McClure
                error function. Defaults to 1.0 (no effect).
    """

    def __init__(self, reduction='mean', loss_weight=1.0, sigma=1.0):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.sigma = sigma

    def forward(self,
                pred,
                target,
                pred_conf=None,
                target_conf=None,
                keypoint_weight=None,
                avg_factor=None,
                loss_weight_override=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction. Shape should be (N, K, 2/3)
                B: batch size. K: number of keypoints.
            target (torch.Tensor): The learning target of the prediction.
                Shape should be the same as pred.
            pred_conf (optional, torch.Tensor): Confidence of
                predicted keypoints. Shape should be (N, K).
            target_conf (optional, torch.Tensor): Confidence of
                target keypoints. Shape should be the same as pred_conf.
            keypoint_weight (optional, torch.Tensor): keypoint-wise weight.
                shape should be (K,). This weight allow different weights
                to be assigned at different body parts.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            loss_weight_override (float, optional): The overall weight of loss
                used to override the original weight of loss.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)

        B, K, D = pred.shape
        pred_conf = pred_conf.view((B, K, 1)) \
            if pred_conf is not None else 1.0
        target_conf = target_conf.view((B, K, 1)) \
            if target_conf is not None else 1.0
        keypoint_weight = keypoint_weight.view((1, K, 1)) \
            if keypoint_weight is not None else 1.0

        weight = keypoint_weight * pred_conf * target_conf
        assert weight.shape == (B, K, 1)

        # B, J, D = pred.shape[:2]
        # if len(weight.shape) == 1:
        #     # for simplify tools
        #     weight = weight.view(1, -1, 1)
        # else:
        #     # for body model estimator
        #     weight = weight.view(B, J, 1)

        loss = loss_weight * mse_loss_with_gmof(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            sigma=self.sigma)

        return loss

import torch
import torch.nn as nn


def rotation_distance_loss(pred, target, epsilon):
    """Warpper of rotation distance loss."""
    tr = torch.einsum(
        'bij,bij->b',
        [pred.view(-1, 3, 3), target.view(-1, 3, 3)])
    theta = (tr - 1) * 0.5
    loss = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
    return loss


class RotationDistance(nn.Module):
    """Rotation Distance Loss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        epsilon (float, optional): A minimal value to avoid NaN.
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', epsilon=1e-7, loss_weight=1.0):
        super(RotationDistance, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        reduction = 'none' if reduction is None else reduction
        self.reduction = reduction
        self.epsilon = epsilon
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
        loss = self.loss_weight * rotation_distance_loss(
            pred, target, epsilon=self.epsilon)
        if weight is not None:
            loss = loss.view(pred.shape[0], -1) * weight.view(
                pred.shape[0], -1)
            return loss.sum() / (weight.gt(0).sum() + self.epsilon)
        else:
            return loss.sum() / pred.shape[0]

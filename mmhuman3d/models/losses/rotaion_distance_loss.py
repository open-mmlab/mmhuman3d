import torch.nn as nn
import torch
from .utils import weighted_loss

def rotation_distance_loss(pred, target, epsilon):
    """Warpper of rotation distance loss."""
    tr = torch.einsum('bij,bij->b',[pred.view(-1, 3, 3),target.view(-1, 3, 3)])
    theta = (tr - 1) * 0.5
    loss = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
    return loss

class RotationDistance(nn.Module):
    def __init__(self, reduction='mean', epsilon=1e-7, loss_weight = 1.0):
        super(RotationDistance, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        reduction = 'none' if reduction is None else reduction
        self.reduction = reduction
        self.epsilon = epsilon
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight = None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * rotation_distance_loss( pred, target, epsilon=self.epsilon)
        if weight is not None:
            loss = loss.view(pred.shape[0], -1) * weight.view(pred.shape[0], -1)
            return loss.sum() / (weight.gt(0).sum() + self.epsilon)
        else:
            return loss.sum() / pred.shape[0]
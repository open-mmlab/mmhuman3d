from mmhuman3d.core.visualization.renderer.torch3d_renderer.builder import build_renderer
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def mse_loss(pred, target):
    """Warpper of mse loss."""
    return F.mse_loss(pred, target, reduction='none')


@LOSSES.register_module('OpticalFlowMSELoss', 'optical_flow_mse_loss',
                        'flow_mse_loss')
class OpticalFlowMSELoss():

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 renderer=None,
                 renderer_config=None):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        if renderer is not None:
            self.renderer = renderer
        if renderer_config is not None:
            self.renderer = build_renderer(renderer_config)
        else:
            raise ValueError('No renderer.')

    def forward(self,
                meshes_source,
                meshes_target,
                cameras_source,
                cameras_target,
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

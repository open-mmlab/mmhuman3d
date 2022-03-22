import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss
from .mse_loss import mse_loss

def keypoint_3d_loss(
        pred_keypoints_3d,
        gt_keypoints_3d,
        has_pose_3d,
        criterion,
):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return (conf * criterion(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(pred_keypoints_3d.device)


@LOSSES.register_module()
class PareKeypoint3DMSELoss(nn.Module):
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

        pred = pred[:,25:,:]
        
        B, K, D = pred.shape
        
        target_conf = target[:, :, -1].unsqueeze(-1).clone()
        target = target[:, :, :-1].clone()
        
        target_conf = target_conf.view((B, K, 1)) \
            if target_conf is not None else 1.0
        keypoint_weight = keypoint_weight.view((1, K, 1)) \
            if keypoint_weight is not None else 1.0

        weight = keypoint_weight * target_conf
        assert weight.shape == (B, K, 1)

        # B, J, D = pred.shape[:2]
        # if len(weight.shape) == 1:
        #     # for simplify tools
        #     weight = weight.view(1, -1, 1)
        # else:
        #     # for body model estimator
        #     weight = weight.view(B, J, 1)

        target_pelvis = (target[:, 2, :] + target[:, 3, :]) / 2
        target = target - target_pelvis[:, None, :]
        pred_pelvis = (pred[:, 2, :] + pred[:, 3, :]) / 2
        pred = pred - pred_pelvis[:, None, :]
        loss = loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            sigma=self.sigma)

        return loss




@LOSSES.register_module()
class CamLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self,
                pred,
            ):
        
        loss = self.loss_weight * ((torch.exp(-pred[:, 0] * 10)) ** 2).mean()
       
        return loss




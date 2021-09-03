import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmhuman3d.utils.transforms import aa_to_rot6d
from ..builder import LOSSES


@LOSSES.register_module()
class ShapePriorLoss(nn.Module):
    """Prior loss for body shape parameters.

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
                betas,
                loss_weight_override=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            betas (torch.Tensor): The body shape parameters
            loss_weight_override (float, optional): The weight of loss used to
                override the original weight of loss
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)

        shape_prior_loss = loss_weight * betas**2

        if reduction == 'mean':
            shape_prior_loss = shape_prior_loss.mean()
        elif reduction == 'sum':
            shape_prior_loss = shape_prior_loss.sum()

        return shape_prior_loss


@LOSSES.register_module()
class JointPriorLoss(nn.Module):
    """Prior loss for joint angles.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
        use_full_body (bool, optional): Use full set of joint constraints
        spine (bool, optional): Limit rotation of 3 spines joints
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 use_full_body=False,
                 spine=False):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_full_body = use_full_body
        self.spine = spine

        if self.use_full_body:
            # all indices in body_pose

            no_angle_prior_idxs = [
                0, 3
            ]  # no pen: L hip 1st angle, R hip 1st angle
            pos_angle_prior_idxs = [
                46
            ]  # penalize extreme positive: L shoulder 2nd angle
            neg_angle_prior_idxs = [
                6, 49
            ]  # penalize extreme negative: lower spine 1st, R shoulder 2nd
            both_angle_prior_idxs = [
                i for i in range(0, 63) if i not in no_angle_prior_idxs +
                pos_angle_prior_idxs + neg_angle_prior_idxs
            ]

            pos_angle_prior_idxs = torch.tensor(
                np.array(pos_angle_prior_idxs, dtype=np.int64),
                dtype=torch.long)
            neg_angle_prior_idxs = torch.tensor(
                np.array(neg_angle_prior_idxs, dtype=np.int64),
                dtype=torch.long)
            both_angle_prior_idxs = torch.tensor(
                np.array(both_angle_prior_idxs, dtype=np.int64),
                dtype=torch.long)
            no_pen_range = torch.tensor([np.pi * 0.5], dtype=torch.float32)

            self.register_buffer('pos_angle_prior_idxs', pos_angle_prior_idxs)
            self.register_buffer('neg_angle_prior_idxs', neg_angle_prior_idxs)
            self.register_buffer('both_angle_prior_idxs',
                                 both_angle_prior_idxs)
            self.register_buffer('no_pen_range', no_pen_range)  # one side

    def forward(self,
                body_pose,
                loss_weight_override=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            body_pose (torch.Tensor): The body pose parameters
            loss_weight_override (float, optional): The weight of loss used to
                override the original weight of loss
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)

        joint_prior_loss = torch.exp(
            body_pose[:, [55, 58, 12, 15]] *
            torch.tensor([1., -1., -1, -1.], device=body_pose.device))**2

        if self.spine:
            spine_poses = body_pose[:, [9, 10, 11, 18, 19, 20, 27, 28, 29]]
            spine_loss = torch.exp(torch.abs(spine_poses))**2
            joint_prior_loss = torch.cat([joint_prior_loss, spine_loss],
                                         axis=1)

        if self.use_full_body:
            pos_angle_prior_angles = body_pose[:, self.pos_angle_prior_idxs]
            pos_angle_prior_loss = F.relu(pos_angle_prior_angles -
                                          self.no_pen_range)  # positive
            pos_angle_prior_loss = torch.exp(pos_angle_prior_loss).pow(2)

            neg_angle_prior_angles = body_pose[:, self.neg_angle_prior_idxs]
            neg_angle_prior_loss = F.relu(
                -(neg_angle_prior_angles + self.no_pen_range))  # negative
            neg_angle_prior_loss = torch.exp(neg_angle_prior_loss).pow(2)

            both_angle_prior_angles = body_pose[:, self.both_angle_prior_idxs]
            both_angle_prior_loss = (
                F.relu(both_angle_prior_angles - self.no_pen_range)
                +  # positive
                F.relu(-(both_angle_prior_angles + self.no_pen_range))
            )  # negative
            both_angle_prior_loss = torch.exp(both_angle_prior_loss).pow(2)

            joint_prior_loss = torch.cat([
                joint_prior_loss, pos_angle_prior_loss, neg_angle_prior_loss,
                both_angle_prior_loss
            ],
                                         axis=1)

        joint_prior_loss = loss_weight * joint_prior_loss

        if reduction == 'mean':
            joint_prior_loss = joint_prior_loss.mean()
        elif reduction == 'sum':
            joint_prior_loss = joint_prior_loss.sum()

        return joint_prior_loss


@LOSSES.register_module()
class SmoothJointLoss(nn.Module):
    """Smooth loss for joint angles.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
        degree (bool, optional): The flag which represents whether the input
            tensor is in degree or radian.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, degree=False):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.degree = degree

    def forward(self,
                body_pose,
                loss_weight_override=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            body_pose (torch.Tensor): The body pose parameters
            loss_weight_override (float, optional): The weight of loss used to
                override the original weight of loss
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)

        theta = body_pose.reshape(body_pose.shape[0], -1, 3)
        if self.degree:
            theta = torch.deg2rad(theta)
        rot_6d = aa_to_rot6d(theta)
        rot_6d_diff = rot_6d[1:] - rot_6d[:-1]
        smooth_joint_loss = rot_6d_diff.abs().sum(dim=-1)
        smooth_joint_loss = torch.cat(
            [torch.zeros_like(smooth_joint_loss)[:1],
             smooth_joint_loss]).sum(dim=-1)

        smooth_joint_loss = loss_weight * smooth_joint_loss

        if reduction == 'mean':
            smooth_joint_loss = smooth_joint_loss.mean()
        elif reduction == 'sum':
            smooth_joint_loss = smooth_joint_loss.sum()

        return smooth_joint_loss


@LOSSES.register_module()
class SmoothPelvisLoss(nn.Module):
    """Smooth loss for pelvis angles.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
        degree (bool, optional): The flag which represents whether the input
            tensor is in degree or radian.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, degree=False):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.degree = degree

    def forward(self,
                global_orient,
                loss_weight_override=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            global_orient (torch.Tensor): The global orientation parameters
            loss_weight_override (float, optional): The weight of loss used to
                override the original weight of loss
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)

        if self.degree:
            global_orient = torch.deg2rad(global_orient)

        pelvis = global_orient.unsqueeze(1)
        rot_6d = aa_to_rot6d(pelvis)

        rot_6d_diff = rot_6d[1:] - rot_6d[:-1]
        smooth_pelvis_loss = rot_6d_diff.abs().sum(dim=-1)
        smooth_pelvis_loss = torch.cat(
            [torch.zeros_like(smooth_pelvis_loss)[:1],
             smooth_pelvis_loss]).sum(dim=-1)

        smooth_pelvis_loss = loss_weight * smooth_pelvis_loss

        if reduction == 'mean':
            smooth_pelvis_loss = smooth_pelvis_loss.mean()
        elif reduction == 'sum':
            smooth_pelvis_loss = smooth_pelvis_loss.sum()

        return smooth_pelvis_loss


@LOSSES.register_module()
class SmoothTranslationLoss(nn.Module):
    """Smooth loss for translations.

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
                translation,
                loss_weight_override=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            translation (torch.Tensor): The body translation parameters
            loss_weight_override (float, optional): The weight of loss used to
                override the original weight of loss
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)

        translation_diff = translation[1:] - translation[:-1]
        smooth_translation_loss = translation_diff.abs().sum(
            dim=-1, keepdim=True)
        smooth_translation_loss = torch.cat([
            torch.zeros_like(smooth_translation_loss)[:1],
            smooth_translation_loss
        ]).sum(dim=-1)
        smooth_translation_loss *= 1e3

        smooth_translation_loss = loss_weight * \
            smooth_translation_loss

        if reduction == 'mean':
            smooth_translation_loss = smooth_translation_loss.mean()
        elif reduction == 'sum':
            smooth_translation_loss = smooth_translation_loss.sum()

        return smooth_translation_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmhuman3d.core.conventions.joints_mapping.standard_joint_angles import (
    STANDARD_JOINT_ANGLE_LIMITS,
    TRANSFORMATION_AA_TO_SJA,
    TRANSFORMATION_SJA_TO_AA,
)
from mmhuman3d.utils.transforms import aa_to_rot6d, aa_to_sja
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
            (in standard joint angles).
        smooth_spine (bool, optional): Ensuring smooth spine rotations
        smooth_spine_loss_weight (float, optional): An additional weight
            factor multiplied on smooth spine loss
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 use_full_body=False,
                 smooth_spine=False,
                 smooth_spine_loss_weight=1.0):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_full_body = use_full_body
        self.smooth_spine = smooth_spine
        self.smooth_spine_loss_weight = smooth_spine_loss_weight

        if self.use_full_body:
            self.register_buffer('R_t', TRANSFORMATION_AA_TO_SJA)
            self.register_buffer('R_t_inv', TRANSFORMATION_SJA_TO_AA)
            self.register_buffer('sja_limits', STANDARD_JOINT_ANGLE_LIMITS)

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

        if self.use_full_body:
            batch_size = body_pose.shape[0]
            body_pose_reshape = body_pose.reshape(batch_size, -1, 3)
            assert body_pose_reshape.shape[1] in (21, 23)  # smpl-x, smpl
            body_pose_reshape = body_pose_reshape[:, :21, :]

            body_pose_sja = aa_to_sja(body_pose_reshape, self.R_t,
                                      self.R_t_inv)

            lower_limits = self.sja_limits[:, :, 0]  # shape: (21, 3)
            upper_limits = self.sja_limits[:, :, 1]  # shape: (21, 3)

            lower_loss = (torch.exp(F.relu(lower_limits - body_pose_sja)) -
                          1).pow(2)
            upper_loss = (torch.exp(F.relu(body_pose_sja - upper_limits)) -
                          1).pow(2)

            standard_joint_angle_prior_loss = (lower_loss + upper_loss).view(
                body_pose.shape[0], -1)  # shape: (n, 3)

            joint_prior_loss = standard_joint_angle_prior_loss

        else:
            # default joint prior loss applied on elbows and knees
            joint_prior_loss = (torch.exp(
                body_pose[:, [55, 58, 12, 15]] *
                torch.tensor([1., -1., -1, -1.], device=body_pose.device)) -
                                1)**2

        if self.smooth_spine:
            spine1 = body_pose[:, [9, 10, 11]]
            spine2 = body_pose[:, [18, 19, 20]]
            spine3 = body_pose[:, [27, 28, 29]]
            smooth_spine_loss_12 = (torch.exp(F.relu(-spine1 * spine2)) -
                                    1).pow(2) * self.smooth_spine_loss_weight
            smooth_spine_loss_23 = (torch.exp(F.relu(-spine2 * spine3)) -
                                    1).pow(2) * self.smooth_spine_loss_weight

            joint_prior_loss = torch.cat(
                [joint_prior_loss, smooth_spine_loss_12, smooth_spine_loss_23],
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

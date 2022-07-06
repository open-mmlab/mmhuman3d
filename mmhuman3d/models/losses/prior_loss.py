import itertools
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmhuman3d.core.conventions.joints_mapping.standard_joint_angles import (
    STANDARD_JOINT_ANGLE_LIMITS,
    TRANSFORMATION_AA_TO_SJA,
    TRANSFORMATION_SJA_TO_AA,
)
from mmhuman3d.utils.keypoint_utils import search_limbs
from mmhuman3d.utils.transforms import aa_to_rot6d, aa_to_sja


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


class ShapeThresholdPriorLoss(nn.Module):
    """Threshold loss for betas. Soft constraint to prevent parameters for
    leaving feasible set. Implements a penalty constraint that encourages the
    parameters to stay in the feasible set of solutions.

    Args:
        margin (int, optional): The threshold value
        norm (str, optional): The loss method. Options are 'l1', l2'
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, margin=1, norm='l2', epsilon=1e-7, loss_weight=1.0):
        super().__init__()
        self.margin = margin
        assert norm in ['l1', 'l2'], 'Norm variable must me l1 or l2'
        self.norm = norm
        self.epsilon = epsilon
        self.loss_weight = loss_weight

    def forward(self, betas):
        """Forward function of loss.

        Args:
            betas (torch.Tensor): The body shape parameters
        Returns:
            torch.Tensor: The calculated loss
        """
        abs_values = betas.abs()
        mask = abs_values.gt(self.margin)
        invalid_values = torch.masked_select(betas, mask)

        if self.norm == 'l1':
            return self.loss_weight * invalid_values.abs().sum() / (
                mask.to(dtype=betas.dtype).sum() + self.epsilon)
        elif self.norm == 'l2':
            return self.loss_weight * invalid_values.pow(2).sum() / (
                mask.to(dtype=betas.dtype).sum() + self.epsilon)


class PoseRegLoss(nn.Module):
    """Regulizer loss for body pose parameters.

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
                body_pose,
                weight=None,
                avg_factor=None,
                loss_weight_override=None,
                reduction_override=None):
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)

        pose_prior_loss = loss_weight * (body_pose**2)

        if reduction == 'mean':
            pose_prior_loss = pose_prior_loss.mean()
        elif reduction == 'sum':
            pose_prior_loss = pose_prior_loss.sum()

        return pose_prior_loss


class LimbLengthLoss(nn.Module):
    """Limb length loss for body shape parameters. As betas are associated with
    the height of a person, fitting on limb length help determine body shape
    parameters. It penalizes the L2 distance between target limb length and
    pred limb length. Note that it should take keypoints3d as input, as limb
    length computed from keypoints2d varies with camera.

    Args:
        convention (str): Limb convention to search for keypoint connections.
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
        eps (float, optional): epsilon for computing normalized limb vector.
            Defaults to 1e-4.
    """

    def __init__(self,
                 convention,
                 reduction='mean',
                 loss_weight=1.0,
                 eps=1e-4):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        limb_idxs, _ = search_limbs(data_source=convention)
        limb_idxs = sorted(limb_idxs['body'])
        self.limb_idxs = np.array(
            list(x for x, _ in itertools.groupby(limb_idxs)))

    def _compute_limb_length(self, keypoints3d):
        kp_src = keypoints3d[:, self.limb_idxs[:, 0], :3]
        kp_dst = keypoints3d[:, self.limb_idxs[:, 1], :3]
        limb_vec = kp_dst - kp_src
        limb_length = torch.norm(limb_vec, dim=2)
        return limb_length

    def _keypoint_conf_to_limb_conf(self, keypoint_conf):
        limb_conf = torch.min(keypoint_conf[:, self.limb_idxs[:, 1]],
                              keypoint_conf[:, self.limb_idxs[:, 0]])
        return limb_conf

    def forward(self,
                pred,
                target,
                pred_conf=None,
                target_conf=None,
                loss_weight_override=None,
                reduction_override=None):
        """Forward function of LimbLengthLoss.

        Args:
            pred (torch.Tensor): The predicted smpl keypoints3d.
                Shape should be (N, K, 3).
                B: batch size. K: number of keypoints.
            target (torch.Tensor): The ground-truth keypoints3d.
                Shape should be (N, K, 3).
            pred_conf (torch.Tensor, optional): Confidence of
                predicted keypoints. Shape should be (N, K).
            target_conf (torch.Tensor, optional): Confidence of
                target keypoints. Shape should be (N, K).
            loss_weight_override (float, optional): The weight of loss used to
                override the original weight of loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None
        Returns:
            torch.Tensor: The calculated loss
        """
        assert pred.dim() == 3 and pred.shape[-1] == 3
        assert pred.shape == target.shape
        if pred_conf is not None:
            assert pred_conf.dim() == 2
            assert pred_conf.shape == pred.shape[:2]
        if target_conf is not None:
            assert target_conf.dim() == 2
            assert target_conf.shape == target.shape[:2]
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)

        limb_len_target = self._compute_limb_length(target)
        limb_len_pred = self._compute_limb_length(pred)

        if target_conf is None:
            target_conf = torch.ones_like(target[..., 0])
        if pred_conf is None:
            pred_conf = torch.ones_like(pred[..., 0])
        limb_conf_target = self._keypoint_conf_to_limb_conf(target_conf)
        limb_conf_pred = self._keypoint_conf_to_limb_conf(pred_conf)
        limb_conf = limb_conf_target * limb_conf_pred

        diff_len = limb_len_target - limb_len_pred
        loss = diff_len**2 * limb_conf

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        loss *= loss_weight

        return loss


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


class SmoothJointLoss(nn.Module):
    """Smooth loss for joint angles.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
        degree (bool, optional): The flag which represents whether the input
            tensor is in degree or radian.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 degree=False,
                 loss_func='L1'):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        assert loss_func in ('L1', 'L2')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.degree = degree
        self.loss_func = loss_func

    def forward(self,
                body_pose,
                loss_weight_override=None,
                reduction_override=None):
        """Forward function of SmoothJointLoss.

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

        if self.loss_func == 'L2':
            smooth_joint_loss = (rot_6d_diff**2).sum(dim=[1, 2])
        elif self.loss_func == 'L1':
            smooth_joint_loss = rot_6d_diff.abs().sum(dim=[1, 2])
        else:
            raise TypeError(f'{self.func} is not defined')

        # add zero padding to retain original batch_size
        smooth_joint_loss = torch.cat(
            [torch.zeros_like(smooth_joint_loss)[:1], smooth_joint_loss])

        if reduction == 'mean':
            smooth_joint_loss = smooth_joint_loss.mean()
        elif reduction == 'sum':
            smooth_joint_loss = smooth_joint_loss.sum()

        smooth_joint_loss *= loss_weight

        return smooth_joint_loss


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
        """Forward function of SmoothPelvisLoss.

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

        # add zero padding to retain original batch_size
        smooth_pelvis_loss = torch.cat(
            [torch.zeros_like(smooth_pelvis_loss)[:1],
             smooth_pelvis_loss]).sum(dim=-1)

        smooth_pelvis_loss = loss_weight * smooth_pelvis_loss

        if reduction == 'mean':
            smooth_pelvis_loss = smooth_pelvis_loss.mean()
        elif reduction == 'sum':
            smooth_pelvis_loss = smooth_pelvis_loss.sum()

        return smooth_pelvis_loss


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

        # add zero padding to retain original batch_size
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


class CameraPriorLoss(nn.Module):
    """Prior loss for predicted camera.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        scale (float, optional): The scale coefficient for regularizing camera
            parameters. Defaults to 10
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, scale=10, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.scale = scale
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cameras,
                loss_weight_override=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            cameras (torch.Tensor): The predicted camera parameters
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

        camera_prior_loss = torch.exp(-cameras[:, 0] * self.scale)
        camera_prior_loss = torch.pow(camera_prior_loss, 2) * loss_weight

        if reduction == 'mean':
            camera_prior_loss = camera_prior_loss.mean()
        elif reduction == 'sum':
            camera_prior_loss = camera_prior_loss.sum()

        return camera_prior_loss


class MaxMixturePrior(nn.Module):
    """Ref: SMPLify-X
    https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
    """

    def __init__(self,
                 prior_folder='data',
                 num_gaussians=8,
                 dtype=torch.float32,
                 epsilon=1e-16,
                 use_merged=True,
                 reduction=None,
                 loss_weight=1.0):
        super(MaxMixturePrior, self).__init__()

        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

        if dtype == torch.float32:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            print('Unknown float type {}, exiting!'.format(dtype))
            sys.exit(-1)

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)

        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            print('The path to the mixture prior "{}"'.format(full_gmm_fn) +
                  ' does not exist, exiting!')
            sys.exit(-1)

        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            print('Unknown type for the prior: {}, exiting!'.format(type(gmm)))
            sys.exit(-1)

        self.register_buffer('means', torch.tensor(means, dtype=dtype))

        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)

        self.register_buffer('precisions',
                             torch.tensor(precisions, dtype=dtype))

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const *
                                                   (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [
            np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon)
            for cov in covs
        ]
        self.register_buffer('cov_dets', torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        """Returns the mean of the mixture."""
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic - \
            torch.log(self.nll_weights)
        #  curr_loglikelihood = 0.5 * (self.cov_dets.unsqueeze(dim=0) +
        #  self.random_var_dim * self.pi_term +
        #  diff_prec_quadratic
        #  ) - torch.log(self.weights)

        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose):
        """Create graph operation for negative log-likelihood calculation."""
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum(
                'bi,bi->b', [curr_loglikelihood, diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (
                cov_term + self.random_var_dim * self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)

        return weight_component + log_likelihoods[:, min_idx]

    def forward(self,
                body_pose,
                loss_weight_override=None,
                reduction_override=None):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)

        if self.use_merged:
            pose_prior_loss = self.merged_log_likelihood(body_pose)
        else:
            pose_prior_loss = self.log_likelihood(body_pose)

        pose_prior_loss = loss_weight * pose_prior_loss

        if reduction == 'mean':
            pose_prior_loss = pose_prior_loss.mean()
        elif reduction == 'sum':
            pose_prior_loss = pose_prior_loss.sum()

        return pose_prior_loss

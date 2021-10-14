# import config
# import constants
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

from mmhuman3d.models.utils.smpl import JOINT_IDS, SMPL49

max_mixture_prior_folder = 'data'
model_path = 'data/body_models/smpl'

# from .losses import camera_fitting_loss, body_fitting_loss
# from utils.geometry import perspective_projection
# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
# from .prior import MaxMixturePrior


def perspective_projection(points, rotation, translation, focal_length,
                           camera_center):
    """This function computes the perspective projection of a set of points.

    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def gmof(x, sigma):
    """Geman-McClure error function."""
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def angle_prior(pose):
    """Angle prior that penalizes unnatural bending of the knees and elbows."""
    # We subtract 3 because pose does not include the global rotation of
    # the model
    return torch.exp(pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] *
                     torch.tensor([1., -1., -1, -1.], device=pose.device))**2


def body_fitting_loss(body_pose,
                      betas,
                      model_joints,
                      camera_t,
                      camera_center,
                      joints_2d,
                      joints_conf,
                      pose_prior,
                      focal_length=5000,
                      sigma=100,
                      pose_prior_weight=4.78,
                      shape_prior_weight=5,
                      angle_prior_weight=15.2,
                      output='sum'):
    """Loss function for body fitting."""

    batch_size = body_pose.shape[0]
    rotation = torch.eye(
        3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    # Weighted robust reprojection error
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf**2) * reprojection_error.sum(dim=-1)

    # Pose prior loss
    pose_prior_loss = (pose_prior_weight**2) * pose_prior(body_pose, betas)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight**2) * \
        angle_prior(body_pose).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight**2) * (betas**2).sum(dim=-1)

    total_loss = reprojection_loss.sum(
        dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        return reprojection_loss


def camera_fitting_loss(model_joints,
                        camera_t,
                        camera_t_est,
                        camera_center,
                        joints_2d,
                        joints_conf,
                        focal_length=5000,
                        depth_loss_weight=100):
    """Loss function for camera optimization."""

    # Project model joints
    batch_size = model_joints.shape[0]
    rotation = torch.eye(
        3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    # op_joints_ind = [constants.JOINT_IDS[joint] for joint in op_joints]
    op_joints_ind = [JOINT_IDS[joint] for joint in op_joints]
    gt_joints = ['Right Hip', 'Left Hip', 'Right Shoulder', 'Left Shoulder']
    # gt_joints_ind = [constants.JOINT_IDS[joint] for joint in gt_joints]
    gt_joints_ind = [JOINT_IDS[joint] for joint in gt_joints]
    reprojection_error_op = (joints_2d[:, op_joints_ind] -
                             projected_joints[:, op_joints_ind])**2
    reprojection_error_gt = (joints_2d[:, gt_joints_ind] -
                             projected_joints[:, gt_joints_ind])**2

    # Check if for each example in the batch all 4 OpenPose detections are
    # valid, otherwise use the GT detections
    # OpenPose joints are more reliable for this task, so we prefer to use
    # them if possible
    is_valid = (joints_conf[:, op_joints_ind].min(dim=-1)[0][:, None, None] >
                0).float()
    reprojection_loss = (is_valid * reprojection_error_op +
                         (1 - is_valid) * reprojection_error_gt).sum(
                             dim=(1, 2))

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight**2) * \
                 (camera_t[:, 2] - camera_t_est[:, 2])**2

    total_loss = reprojection_loss + depth_loss
    return total_loss.sum()


DEFAULT_DTYPE = torch.float32


class MaxMixturePrior(nn.Module):

    def __init__(self,
                 prior_folder='prior',
                 num_gaussians=6,
                 dtype=DEFAULT_DTYPE,
                 epsilon=1e-16,
                 use_merged=True,
                 **kwargs):
        super(MaxMixturePrior, self).__init__()

        if dtype == DEFAULT_DTYPE:
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

    def merged_log_likelihood(self, pose, betas):
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

    def log_likelihood(self, pose, betas, *args, **kwargs):
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

    def forward(self, pose, betas):
        if self.use_merged:
            return self.merged_log_likelihood(pose, betas)
        else:
            return self.log_likelihood(pose, betas)


class SMPLify():
    """Implementation of single-stage SMPLify."""

    def __init__(self,
                 step_size=1e-2,
                 batch_size=64 * 8,
                 num_iters=50,
                 focal_length=5000,
                 device=torch.device('cuda')):

        # Store options
        self.device = device
        self.focal_length = focal_length
        self.step_size = step_size

        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        # self.ign_joints = [constants.JOINT_IDS[i] for i in ign_joints]
        self.ign_joints = [JOINT_IDS[i] for i in ign_joints]

        ign_joints_24 = ['Right Hip', 'Left Hip']
        self.ign_joints_24 = [JOINT_IDS[i] - 25 for i in ign_joints_24]

        self.num_iters = num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(
            prior_folder=max_mixture_prior_folder,
            num_gaussians=8,
            dtype=torch.float32).to(device)
        # Load SMPL model
        self.smpl = SMPL49(
            model_path=model_path,
            batch_size=batch_size,
            create_transl=False,
            extra_joints_regressor='data/J_regressor_extra.npy').to(device)

    def __call__(self, init_pose, init_betas, init_cam_t, camera_center,
                 keypoints_2d):
        """Perform body fitting.

        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """
        num_keypoints = keypoints_2d.shape[1]
        assert num_keypoints in [24, 49]
        if num_keypoints == 24:
            self.ign_joints = self.ign_joints_24

        # batch_size = init_pose.shape[0]

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]
        camera_optimizer = torch.optim.Adam(
            camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.num_iters):
            smpl_output = self.smpl(
                global_orient=global_orient, body_pose=body_pose, betas=betas)

            model_joints = smpl_output.joints
            loss = camera_fitting_loss(
                model_joints,
                camera_translation,
                init_cam_t,
                camera_center,
                joints_2d,
                joints_conf,
                focal_length=self.focal_length)
            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        # Fix camera translation after optimizing camera
        camera_translation.requires_grad = False

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = False
        body_opt_params = [body_pose, betas, global_orient]

        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        body_optimizer = torch.optim.Adam(
            body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        for i in range(self.num_iters):
            smpl_output = self.smpl(
                global_orient=global_orient, body_pose=body_pose, betas=betas)
            model_joints = smpl_output.joints
            loss = body_fitting_loss(
                body_pose,
                betas,
                model_joints,
                camera_translation,
                camera_center,
                joints_2d,
                joints_conf,
                self.pose_prior,
                focal_length=self.focal_length)
            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                return_full_pose=True)
            model_joints = smpl_output.joints
            reprojection_loss = body_fitting_loss(
                body_pose,
                betas,
                model_joints,
                camera_translation,
                camera_center,
                joints_2d,
                joints_conf,
                self.pose_prior,
                focal_length=self.focal_length,
                output='reprojection')

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return vertices, joints, pose, betas, camera_translation, \
            reprojection_loss

    def get_fitting_loss(self, pose, betas, cam_t, camera_center,
                         keypoints_2d):
        """Given body and camera parameters, compute reprojection loss value.

        Input:
            pose: SMPL pose parameters
            betas: SMPL beta parameters
            cam_t: Camera translation
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            reprojection_loss: Final joint reprojection loss
        """

        # batch_size = pose.shape[0]

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]
        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        # Split SMPL pose to body pose and global orientation
        body_pose = pose[:, 3:]
        global_orient = pose[:, :3]

        with torch.no_grad():
            smpl_output = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                return_full_pose=True)
            model_joints = smpl_output.joints
            reprojection_loss = body_fitting_loss(
                body_pose,
                betas,
                model_joints,
                cam_t,
                camera_center,
                joints_2d,
                joints_conf,
                self.pose_prior,
                focal_length=self.focal_length,
                output='reprojection')

        return reprojection_loss

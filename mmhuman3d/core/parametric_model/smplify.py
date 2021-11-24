""" TODO
1. hyperparameter tuining: sigma for keypoints 3d
2. fix default config (smplify use smplify's), also in tests/
3. do we still need the closure?

optional:
4. optimize how to get batch_size and num_frames
5. check if SMPL layer is better
6. better num_videos handling
7. step by step visualization
8. _optimize_stage() inheritance
9. allow_grad to make .detach().clone() optional at init and ret?
10. provide a way to check keypoint conventions are aligned
"""

import torch
from mmcv.runner import build_optimizer

from mmhuman3d.core.conventions.keypoints_mapping import (
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.models.builder import build_body_model, build_loss
from .builder import REGISTRANTS

# from pytorch3d.renderer import cameras


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


class OptimizableParameters():
    """Collects parameters for optimization."""

    def __init__(self):
        self.opt_params = []

    def set_param(self, fit_param, param):
        """Set requires_grad and collect parameters for optimization.

        :param fit_param: (bool) True if optimizable parameters
        :param param: (torch.Tenosr) parameters
        """
        if fit_param:
            param.requires_grad = True
            self.opt_params.append(param)
        else:
            param.requires_grad = False

    def parameters(self):
        """Returns parameters.

        Compatible with mmcv's build_parameters()
        """
        return self.opt_params


@REGISTRANTS.register_module()
class SMPLify(object):
    """Re-implementation of SMPLify with extended features."""

    def __init__(self,
                 body_model,
                 num_epochs=20,
                 camera=None,
                 img_res=224,
                 stages=None,
                 optimizer=None,
                 keypoints2d_loss=None,
                 keypoints3d_loss=None,
                 shape_prior_loss=None,
                 joint_prior_loss=None,
                 smooth_loss=None,
                 pose_prior_loss=None,
                 use_one_betas_per_video=False,
                 ignore_keypoints=None,
                 device=torch.device(
                     'cuda' if torch.cuda.is_available() else 'cpu'),
                 verbose=False,
                 **kwargs):

        self.use_one_betas_per_video = use_one_betas_per_video
        self.num_epochs = num_epochs
        self.img_res = img_res
        self.device = device
        self.stage_config = stages
        self.optimizer = optimizer
        self.keypoints2d_mse_loss = build_loss(keypoints2d_loss)
        self.keypoints3d_mse_loss = build_loss(keypoints3d_loss)
        self.shape_prior_loss = build_loss(shape_prior_loss)
        self.joint_prior_loss = build_loss(joint_prior_loss)
        self.smooth_loss = build_loss(smooth_loss)
        self.pose_prior_loss = build_loss(pose_prior_loss)

        if self.joint_prior_loss is not None:
            self.joint_prior_loss = self.joint_prior_loss.to(self.device)
        if self.smooth_loss is not None:
            self.smooth_loss = self.smooth_loss.to(self.device)
        if self.pose_prior_loss is not None:
            self.pose_prior_loss = self.pose_prior_loss.to(self.device)

        # initialize body model
        self.body_model = build_body_model(body_model).to(self.device)

        self.ignore_keypoints = ignore_keypoints
        self.verbose = verbose

        self._set_keypoint_idxs()

    def __call__(self,
                 keypoints2d=None,
                 keypoints2d_conf=None,
                 keypoints3d=None,
                 keypoints3d_conf=None,
                 init_global_orient=None,
                 init_transl=None,
                 init_body_pose=None,
                 init_betas=None,
                 batch_size=None,
                 num_videos=None,
                 return_verts=False,
                 return_joints=False,
                 return_full_pose=False,
                 return_losses=False):

        assert keypoints2d is not None or keypoints3d is not None, \
            'Neither of 2D nor 3D keypoints are provided.'
        assert not (keypoints2d is not None and keypoints3d is not None), \
            'Do not provide both 2D and 3D keypoints.'
        if batch_size is None:
            batch_size = keypoints2d.shape[
                0] if keypoints2d is not None else keypoints3d.shape[0]
        if num_videos is None:
            if self.use_one_betas_per_video:
                num_videos = 1
            else:
                num_videos = batch_size
        assert batch_size % num_videos == 0

        global_orient = init_global_orient.detach().clone() \
            if init_global_orient is not None \
            else self.body_model.global_orient.detach().clone()
        transl = init_transl.detach().clone() \
            if init_transl is not None \
            else self.body_model.transl.detach().clone()
        body_pose = init_body_pose.detach().clone() \
            if init_body_pose is not None \
            else self.body_model.body_pose.detach().clone()
        if init_betas is not None:
            betas = init_betas.detach().clone()
        elif self.use_one_betas_per_video:
            betas = torch.zeros(
                num_videos, self.body_model.betas.shape[-1]).to(self.device)
        else:
            betas = self.body_model.betas.detach().clone()

        for i in range(self.num_epochs):
            for stage_idx, stage_config in enumerate(self.stage_config):
                self._optimize_stage(
                    global_orient=global_orient,
                    transl=transl,
                    body_pose=body_pose,
                    betas=betas,
                    keypoints2d=keypoints2d,
                    keypoints2d_conf=keypoints2d_conf,
                    keypoints3d=keypoints3d,
                    keypoints3d_conf=keypoints3d_conf,
                    **stage_config,
                )

        # collate results
        ret = {
            'global_orient': global_orient,
            'transl': transl,
            'body_pose': body_pose,
            'betas': betas
        }

        if return_verts or return_joints or \
                return_full_pose or return_losses:
            eval_ret = self.evaluate(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl,
                keypoints2d=keypoints2d,
                keypoints2d_conf=keypoints2d_conf,
                keypoints3d=keypoints3d,
                keypoints3d_conf=keypoints3d_conf,
                return_verts=return_verts,
                return_full_pose=return_full_pose,
                return_joints=return_joints,
                reduction_override='none'  # sample-wise loss
            )

            if return_verts:
                ret['vertices'] = eval_ret['vertices']
            if return_joints:
                ret['joints'] = eval_ret['joints']
            if return_full_pose:
                ret['full_pose'] = eval_ret['full_pose']
            if return_losses:
                for k in eval_ret.keys():
                    if 'loss' in k:
                        ret[k] = eval_ret[k]

        for k, v in ret.items():
            if isinstance(v, torch.Tensor):
                ret[k] = v.detach().clone()

        return ret

    def _optimize_stage(self,
                        global_orient,
                        transl,
                        body_pose,
                        betas,
                        fit_global_orient=True,
                        fit_transl=True,
                        fit_body_pose=True,
                        fit_betas=True,
                        keypoints2d=None,
                        keypoints2d_conf=None,
                        keypoints2d_weight=None,
                        keypoints3d=None,
                        keypoints3d_conf=None,
                        keypoints3d_weight=None,
                        shape_prior_weight=None,
                        joint_prior_weight=None,
                        smooth_loss_weight=None,
                        pose_prior_weight=None,
                        joint_weights={},
                        num_iter=1):

        parameters = OptimizableParameters()
        parameters.set_param(fit_global_orient, global_orient)
        parameters.set_param(fit_transl, transl)
        parameters.set_param(fit_body_pose, body_pose)
        parameters.set_param(fit_betas, betas)

        optimizer = build_optimizer(parameters, self.optimizer)

        for iter_idx in range(num_iter):

            def closure():
                optimizer.zero_grad()
                betas_ext = self._expand_betas(body_pose, betas)

                loss_dict = self.evaluate(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas_ext,
                    transl=transl,
                    keypoints2d=keypoints2d,
                    keypoints2d_conf=keypoints2d_conf,
                    keypoints2d_weight=keypoints2d_weight,
                    keypoints3d=keypoints3d,
                    keypoints3d_conf=keypoints3d_conf,
                    keypoints3d_weight=keypoints3d_weight,
                    joint_prior_weight=joint_prior_weight,
                    shape_prior_weight=shape_prior_weight,
                    smooth_loss_weight=smooth_loss_weight,
                    pose_prior_weight=pose_prior_weight,
                    joint_weights=joint_weights)

                loss = loss_dict['total_loss']
                loss.backward()
                return loss

            optimizer.step(closure)

    def evaluate(
        self,
        global_orient=None,
        body_pose=None,
        betas=None,
        transl=None,
        keypoints2d=None,
        keypoints2d_conf=None,
        keypoints2d_weight=None,
        keypoints3d=None,
        keypoints3d_conf=None,
        keypoints3d_weight=None,
        shape_prior_weight=None,
        joint_prior_weight=None,
        smooth_loss_weight=None,
        pose_prior_weight=None,
        joint_weights={},
        return_verts=False,
        return_full_pose=False,
        return_joints=False,
        reduction_override=None,
    ):
        """Evaluate fitted parameters through loss computation."""

        ret = {}

        body_model_output = self.body_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            return_verts=return_verts,
            return_full_pose=return_full_pose)

        model_joints = body_model_output['joints']

        loss_dict = self._compute_loss(
            model_joints,
            keypoints2d=keypoints2d,
            keypoints2d_conf=keypoints2d_conf,
            keypoints2d_weight=keypoints2d_weight,
            keypoints3d=keypoints3d,
            keypoints3d_conf=keypoints3d_conf,
            keypoints3d_weight=keypoints3d_weight,
            joint_prior_weight=joint_prior_weight,
            shape_prior_weight=shape_prior_weight,
            smooth_loss_weight=smooth_loss_weight,
            pose_prior_weight=pose_prior_weight,
            joint_weights=joint_weights,
            reduction_override=reduction_override,
            body_pose=body_pose,
            betas=betas)
        ret.update(loss_dict)

        if return_verts:
            ret['vertices'] = body_model_output['vertices']
        if return_full_pose:
            ret['full_pose'] = body_model_output['full_pose']
        if return_joints:
            ret['joints'] = model_joints

        return ret

    def _compute_loss(self,
                      model_joints,
                      keypoints2d=None,
                      keypoints2d_conf=None,
                      keypoints3d=None,
                      keypoints3d_conf=None,
                      keypoints2d_weight=None,
                      keypoints3d_weight=None,
                      shape_prior_weight=None,
                      joint_prior_weight=None,
                      smooth_loss_weight=None,
                      pose_prior_weight=None,
                      joint_weights={},
                      reduction_override=None,
                      body_pose=None,
                      betas=None):

        losses = {}

        weight = self._get_weight(**joint_weights)

        # 2D keypoint loss
        if keypoints2d is not None:
            # TODO: temp!
            # projected_joints = \
            # self.camera.transform_points(model_joints)[:, :, :2]
            bs = model_joints.shape[0]
            projected_joints = perspective_projection(
                model_joints,
                torch.eye(3).expand((bs, 3, 3)).to(model_joints.device),
                torch.zeros((bs, 3)).to(model_joints.device), 5000.0,
                torch.Tensor([self.img_res / 2,
                              self.img_res / 2]).to(model_joints.device))

            # normalize keypoints to [-1,1]
            projected_joints = 2 * projected_joints / (self.img_res - 1) - 1
            keypoints2d = 2 * keypoints2d / (self.img_res - 1) - 1

            keypoint2d_loss = self.keypoints2d_mse_loss(
                pred=projected_joints,
                target=keypoints2d,
                target_conf=keypoints2d_conf,
                keypoint_weight=weight,
                loss_weight_override=keypoints2d_weight,
                reduction_override=reduction_override)
            losses['keypoint2d_loss'] = keypoint2d_loss

        # 3D keypoint loss
        if keypoints3d is not None:
            keypoints3d_loss = self.keypoints3d_mse_loss(
                pred=model_joints,
                target=keypoints3d,
                target_conf=keypoints3d_conf,
                keypoint_weight=weight,
                loss_weight_override=keypoints3d_weight,
                reduction_override=reduction_override)
            losses['keypoints3d_loss'] = keypoints3d_loss

        # regularizer to prevent betas from taking large values
        if self.shape_prior_loss is not None:
            shape_prior_loss = self.shape_prior_loss(
                betas=betas,
                loss_weight_override=shape_prior_weight,
                reduction_override=reduction_override)
            losses['shape_prior_loss'] = shape_prior_loss

        # joint prior loss
        if self.joint_prior_loss is not None:
            joint_prior_loss = self.joint_prior_loss(
                body_pose=body_pose,
                loss_weight_override=joint_prior_weight,
                reduction_override=reduction_override)
            losses['joint_prior_loss'] = joint_prior_loss

        # smooth body loss
        if self.smooth_loss is not None:
            smooth_loss = self.smooth_loss(
                body_pose=body_pose,
                loss_weight_override=smooth_loss_weight,
                reduction_override=reduction_override)
            losses['smooth_loss'] = smooth_loss

        # pose prior loss
        if self.pose_prior_loss is not None:
            pose_prior_loss = self.pose_prior_loss(
                body_pose=body_pose,
                loss_weight_override=pose_prior_weight,
                reduction_override=reduction_override)
            losses['pose_prior_loss'] = pose_prior_loss

        if self.verbose:
            msg = ''
            for loss_name, loss in losses.items():
                msg += f'{loss_name}={loss.mean().item():.6f}'
            print(msg)

        total_loss = 0
        for loss_name, loss in losses.items():
            if loss.ndim == 3:
                total_loss = total_loss + loss.sum(dim=(2, 1))
            elif loss.ndim == 2:
                total_loss = total_loss + loss.sum(dim=-1)
            else:
                total_loss = total_loss + loss
        losses['total_loss'] = total_loss

        return losses

    def _set_keypoint_idxs(self):
        convention = self.body_model.keypoint_dst

        # obtain ignore keypoint indices
        if self.ignore_keypoints is not None:
            self.ignore_keypoint_idxs = []
            for keypoint_name in self.ignore_keypoints:
                keypoint_idx = get_keypoint_idx(
                    keypoint_name, convention=convention)
                if keypoint_idx != -1:
                    self.ignore_keypoint_idxs.append(keypoint_idx)

        # obtain body part keypoint indices
        shoulder_keypoint_idxs = get_keypoint_idxs_by_part(
            'shoulder', convention=convention)
        hip_keypoint_idxs = get_keypoint_idxs_by_part(
            'hip', convention=convention)
        self.shoulder_hip_keypoint_idxs = [
            *shoulder_keypoint_idxs, *hip_keypoint_idxs
        ]

    def _get_weight(self, use_shoulder_hip_only=False, body_weight=1.0):
        num_keypoint = self.body_model.num_joints

        if use_shoulder_hip_only:
            weight = torch.zeros([num_keypoint]).to(self.device)
            weight[self.shoulder_hip_keypoint_idxs] = 1.0
            weight = weight * body_weight
        else:
            weight = torch.ones([num_keypoint]).to(self.device)
            weight = weight * body_weight

        if hasattr(self, 'ignore_keypoint_idxs'):
            weight[self.ignore_keypoint_idxs] = 0.0

        return weight

    def _set_param(self, fit_param, param, opt_param):
        if fit_param:
            param.requires_grad = True
            opt_param.append(param)
        else:
            param.requires_grad = False

    def _expand_betas(self, pose, betas):
        batch_size = pose.shape[0]
        num_video = betas.shape[0]

        if batch_size == num_video:
            return betas

        feat_dim = betas.shape[-1]
        video_size = batch_size // num_video
        betas_ext = betas.\
            view(num_video, 1, feat_dim).\
            expand(num_video, video_size, feat_dim).\
            view(batch_size, feat_dim)

        return betas_ext


@REGISTRANTS.register_module()
class SMPLifyX(SMPLify):
    """Re-implementation of SMPLify-X with extended features."""

    def __call__(self,
                 keypoints2d=None,
                 keypoints2d_conf=1.0,
                 keypoints3d=None,
                 keypoints3d_conf=1.0,
                 init_global_orient=None,
                 init_transl=None,
                 init_body_pose=None,
                 init_betas=None,
                 init_left_hand_pose=None,
                 init_right_hand_pose=None,
                 init_expression=None,
                 init_jaw_pose=None,
                 init_leye_pose=None,
                 init_reye_pose=None,
                 batch_size=None,
                 num_videos=None):

        assert keypoints2d is not None or keypoints3d is not None, \
            'Neither of 2D nor 3D keypoints are provided.'
        assert not (keypoints2d is not None and keypoints3d is not None), \
            'Do not provide both 2D and 3D keypoints.'
        if batch_size is None:
            batch_size = keypoints2d.shape[
                0] if keypoints2d is not None else keypoints3d.shape[0]
        if num_videos is None:
            num_videos = batch_size
        assert batch_size % num_videos == 0

        global_orient = init_global_orient if init_global_orient is not None \
            else self.body_model.global_orient
        transl = init_transl if init_transl is not None \
            else self.body_model.transl
        body_pose = init_body_pose if init_body_pose is not None \
            else self.body_model.body_pose

        left_hand_pose = init_left_hand_pose \
            if init_left_hand_pose is not None \
            else self.body_model.left_hand_pose
        right_hand_pose = init_right_hand_pose \
            if init_right_hand_pose is not None \
            else self.body_model.right_hand_pose
        expression = init_expression \
            if init_expression is not None \
            else self.body_model.expression
        jaw_pose = init_jaw_pose \
            if init_jaw_pose is not None \
            else self.body_model.jaw_pose
        leye_pose = init_leye_pose \
            if init_leye_pose is not None \
            else self.body_model.leye_pose
        reye_pose = init_reye_pose \
            if init_reye_pose is not None \
            else self.body_model.reye_pose

        if init_betas is not None:
            betas = init_betas
        elif self.use_one_betas_per_video:
            betas = torch.zeros(
                num_videos, self.body_model.betas.shape[-1]).to(self.device)
        else:
            betas = self.body_model.betas

        for i in range(self.num_epochs):
            for stage_idx, stage_config in enumerate(self.stage_config):
                # print(stage_name)
                self._optimize_stage(
                    global_orient=global_orient,
                    transl=transl,
                    body_pose=body_pose,
                    betas=betas,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    expression=expression,
                    jaw_pose=jaw_pose,
                    leye_pose=leye_pose,
                    reye_pose=reye_pose,
                    keypoints2d=keypoints2d,
                    keypoints2d_conf=keypoints2d_conf,
                    keypoints3d=keypoints3d,
                    keypoints3d_conf=keypoints3d_conf,
                    **stage_config,
                )

        return {
            'global_orient': global_orient,
            'transl': transl,
            'body_pose': body_pose,
            'betas': betas,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'expression': expression,
            'jaw_pose': jaw_pose,
            'leye_pose': leye_pose,
            'reye_pose': reye_pose
        }

    def _optimize_stage(self,
                        global_orient,
                        transl,
                        body_pose,
                        betas,
                        left_hand_pose,
                        right_hand_pose,
                        expression,
                        jaw_pose,
                        leye_pose,
                        reye_pose,
                        fit_global_orient=True,
                        fit_transl=True,
                        fit_body_pose=True,
                        fit_betas=True,
                        fit_left_hand_pose=True,
                        fit_right_hand_pose=True,
                        fit_expression=True,
                        fit_jaw_pose=True,
                        fit_leye_pose=True,
                        fit_reye_pose=True,
                        keypoints2d=None,
                        keypoints2d_conf=None,
                        keypoints2d_weight=None,
                        keypoints3d=None,
                        keypoints3d_conf=None,
                        keypoints3d_weight=None,
                        joint_prior_weight=None,
                        shape_prior_weight=None,
                        smooth_loss_weight=None,
                        pose_prior_weight=None,
                        joint_weights={},
                        num_iter=1):

        parameters = OptimizableParameters()
        parameters.set_param(fit_global_orient, global_orient)
        parameters.set_param(fit_transl, transl)
        parameters.set_param(fit_body_pose, body_pose)
        parameters.set_param(fit_betas, betas)
        parameters.set_param(fit_left_hand_pose, left_hand_pose)
        parameters.set_param(fit_right_hand_pose, right_hand_pose)
        parameters.set_param(fit_expression, expression)
        parameters.set_param(fit_jaw_pose, jaw_pose)
        parameters.set_param(fit_leye_pose, leye_pose)
        parameters.set_param(fit_reye_pose, reye_pose)

        optimizer = build_optimizer(parameters, self.optimizer)

        for iter_idx in range(num_iter):

            def closure():
                # body_pose_fixed = use_reference_spine(body_pose,
                # init_body_pose)

                optimizer.zero_grad()
                betas_ext = self._expand_betas(body_pose, betas)

                loss_dict = self.evaluate(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas_ext,
                    transl=transl,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    expression=expression,
                    jaw_pose=jaw_pose,
                    leye_pose=leye_pose,
                    reye_pose=reye_pose,
                    keypoints2d=keypoints2d,
                    keypoints2d_conf=keypoints2d_conf,
                    keypoints2d_weight=keypoints2d_weight,
                    keypoints3d=keypoints3d,
                    keypoints3d_conf=keypoints3d_conf,
                    keypoints3d_weight=keypoints3d_weight,
                    joint_prior_weight=joint_prior_weight,
                    shape_prior_weight=shape_prior_weight,
                    smooth_loss_weight=smooth_loss_weight,
                    pose_prior_weight=pose_prior_weight,
                    joint_weights=joint_weights)

                loss = loss_dict['total_loss']
                loss.backward()
                return loss

            optimizer.step(closure)

    def evaluate(
        self,
        global_orient=None,
        body_pose=None,
        betas=None,
        transl=None,
        left_hand_pose=None,
        right_hand_pose=None,
        expression=None,
        jaw_pose=None,
        leye_pose=None,
        reye_pose=None,
        keypoints2d=None,
        keypoints2d_conf=None,
        keypoints2d_weight=None,
        keypoints3d=None,
        keypoints3d_conf=None,
        keypoints3d_weight=None,
        shape_prior_weight=None,
        joint_prior_weight=None,
        smooth_loss_weight=None,
        pose_prior_weight=None,
        joint_weights={},
        return_verts=False,
        return_full_pose=False,
        return_joints=False,
        reduction_override=None,
    ):
        """Evaluate fitted parameters through loss computation."""

        ret = {}

        body_model_output = self.body_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            return_verts=return_verts,
            return_full_pose=return_full_pose)

        model_joints = body_model_output['joints']

        loss_dict = self._compute_loss(
            model_joints,
            keypoints2d=keypoints2d,
            keypoints2d_conf=keypoints2d_conf,
            keypoints2d_weight=keypoints2d_weight,
            keypoints3d=keypoints3d,
            keypoints3d_conf=keypoints3d_conf,
            keypoints3d_weight=keypoints3d_weight,
            joint_prior_weight=joint_prior_weight,
            shape_prior_weight=shape_prior_weight,
            smooth_loss_weight=smooth_loss_weight,
            pose_prior_weight=pose_prior_weight,
            joint_weights=joint_weights,
            reduction_override=reduction_override,
            body_pose=body_pose,
            betas=betas)
        ret.update(loss_dict)

        if return_verts:
            ret['vertices'] = body_model_output['vertices']
        if return_full_pose:
            ret['full_pose'] = body_model_output['full_pose']
        if return_joints:
            ret['joints'] = model_joints

        return ret

    def _get_weight(self,
                    use_shoulder_hip_only=False,
                    body_weight=1.0,
                    hand_weight=1.0,
                    face_weight=1.0):

        # TODO: accept any convention!
        num_keypoint = self.body_model.num_joints

        if use_shoulder_hip_only:
            weight = torch.zeros([num_keypoint]).to(self.device)
            weight[self.shoulder_hip_keypoint_idxs] = 1.0
        else:
            weight = torch.ones([num_keypoint]).to(self.device)

            weight[self.body_keypoint_idxs] = \
                weight[self.body_keypoint_idxs] * body_weight
            weight[self.hand_keypoint_idxs] = \
                weight[self.hand_keypoint_idxs] * hand_weight
            weight[self.face_keypoint_idxs] = \
                weight[self.face_keypoint_idxs] * face_weight

        if hasattr(self, 'ignore_keypoint_idxs'):
            weight[self.ignore_keypoint_idxs] = 0.0

        return weight

    def _set_keypoint_idxs(self):
        convention = self.body_model.keypoint_dst

        # obtain ignore keypoint indices
        if self.ignore_keypoints is not None:
            self.ignore_keypoint_idxs = []
            for keypoint_name in self.ignore_keypoints:
                keypoint_idx = get_keypoint_idx(
                    keypoint_name, convention=convention)
                if keypoint_idx != -1:
                    self.ignore_keypoint_idxs.append(keypoint_idx)

        # obtain body part keypoint indices
        shoulder_keypoint_idxs = get_keypoint_idxs_by_part(
            'shoulder', convention=convention)
        hip_keypoint_idxs = get_keypoint_idxs_by_part(
            'hip', convention=convention)
        self.shoulder_hip_keypoint_idxs = [
            *shoulder_keypoint_idxs, *hip_keypoint_idxs
        ]

        # head keypoints include all facial landmarks
        self.face_keypoint_idxs = get_keypoint_idxs_by_part(
            'head', convention=convention)

        left_hand_keypoint_idxs = get_keypoint_idxs_by_part(
            'left_hand', convention=convention)
        right_hand_keypoint_idxs = get_keypoint_idxs_by_part(
            'right_hand', convention=convention)
        self.hand_keypoint_idxs = [
            *left_hand_keypoint_idxs, *right_hand_keypoint_idxs
        ]

        self.body_keypoint_idxs = get_keypoint_idxs_by_part(
            'body', convention=convention)

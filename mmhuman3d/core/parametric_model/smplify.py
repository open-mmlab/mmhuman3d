""" TODO
1. add camera
2. add 2D loss support
3. add GMM prior (vpose, hand, expression)
4. hyperparameter tuining: sigma for keypoints 3d
5. different keypoint weight for different stages
6. add support for 45 smpl joints (update tests also)

optional:
7. optimize how to get batch_size and num_frames
8. add default body model
9. add model inference for param init
10. check if SMPL layer is better
11. better num_videos handling
12. step by step visualization
13. _optimize_stage() inheritance
"""

import torch
from configs.smplify.smplify import smplify_opt_config, smplify_stages
from configs.smplify.smplifyx import (
    joint_prior_loss_config,
    keypoints_2d_loss_config,
    keypoints_3d_loss_config,
    shape_prior_loss_config,
    smplifyx_opt_config,
    smplifyx_stages,
)
from mmcv.runner import build_optimizer

from mmhuman3d.models.builder import build_loss

# TODO: placeholder
default_camera = {}


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


class SMPLify(object):
    """Re-implementation of SMPLify with extended features."""

    def __init__(self,
                 body_model=None,
                 keypoints_2d_weight=1.0,
                 keypoints_3d_weight=1.0,
                 use_one_betas_per_video=False,
                 num_epochs=20,
                 camera=default_camera,
                 stage_config=smplify_stages,
                 opt_config=smplify_opt_config,
                 keypoints_2d_loss_config=keypoints_2d_loss_config,
                 keypoints_3d_loss_config=keypoints_3d_loss_config,
                 shape_prior_loss_config=shape_prior_loss_config,
                 joint_prior_loss_config=joint_prior_loss_config,
                 device=torch.device('cuda'),
                 verbose=False):

        self.keypoints_2d_weight = keypoints_2d_weight
        self.keypoints_3d_weight = keypoints_3d_weight
        self.use_one_betas_per_video = use_one_betas_per_video
        self.num_epochs = num_epochs
        self.stage_config = stage_config
        self.opt_config = opt_config
        self.camera = camera
        self.device = device
        self.body_model = body_model.to(self.device)
        self.keypoints_2d_mse_loss = build_loss(keypoints_2d_loss_config)
        self.keypoints_3d_mse_loss = build_loss(keypoints_3d_loss_config)
        self.shape_prior_loss = build_loss(shape_prior_loss_config)
        self.joint_prior_loss = build_loss(joint_prior_loss_config).to(device)

    def __call__(self,
                 keypoints_2d=None,
                 keypoints_conf_2d=None,
                 keypoints_3d=None,
                 keypoints_conf_3d=None,
                 init_global_orient=None,
                 init_transl=None,
                 init_body_pose=None,
                 init_betas=None,
                 batch_size=None,
                 num_videos=None):

        assert keypoints_2d is not None or keypoints_3d is not None, \
            'Neither of 2D nor 3D keypoints groud truth is provided.'
        if batch_size is None:
            batch_size = keypoints_2d.shape[
                0] if keypoints_2d is not None else keypoints_3d.shape[0]
        if num_videos is None:
            num_videos = batch_size
        assert batch_size % num_videos == 0

        global_orient = init_global_orient if init_global_orient is not None \
            else self.body_model.global_orient
        transl = init_transl if init_transl is not None \
            else self.body_model.transl
        body_pose = init_body_pose if init_body_pose is not None \
            else self.body_model.body_pose
        if init_betas is not None:
            betas = init_betas
        elif self.use_one_betas_per_video:
            betas = torch.zeros(
                num_videos, self.body_model.betas.shape[-1]).to(self.device)
        else:
            betas = self.body_model.betas

        for i in range(self.num_epochs):
            for stage_name, stage_config in self.stage_config.items():
                self._optimize_stage(
                    global_orient=global_orient,
                    transl=transl,
                    body_pose=body_pose,
                    betas=betas,
                    keypoints_2d=keypoints_2d,
                    keypoints_conf_2d=keypoints_conf_2d,
                    keypoints_3d=keypoints_3d,
                    keypoints_conf_3d=keypoints_conf_3d,
                    **stage_config,
                )

        return {
            'global_orient': global_orient,
            'transl': transl,
            'body_pose': body_pose,
            'betas': betas
        }

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

    def _optimize_stage(self,
                        global_orient,
                        transl,
                        body_pose,
                        betas,
                        fit_global_orient=True,
                        fit_transl=True,
                        fit_body_pose=True,
                        fit_betas=True,
                        keypoints_2d=None,
                        keypoints_conf_2d=None,
                        keypoints_2d_weight=None,
                        keypoints_3d=None,
                        keypoints_conf_3d=None,
                        keypoints_3d_weight=None,
                        shape_prior_weight=None,
                        joint_prior_weight=None,
                        joint_weights={},
                        num_iter=1):

        parameters = OptimizableParameters()
        parameters.set_param(fit_global_orient, global_orient)
        parameters.set_param(fit_transl, transl)
        parameters.set_param(fit_body_pose, body_pose)
        parameters.set_param(fit_betas, betas)

        optimizer = build_optimizer(parameters, self.opt_config)

        for iter_idx in range(num_iter):

            def closure():
                # body_pose_fixed = use_reference_spine(body_pose,
                # init_body_pose)

                optimizer.zero_grad()
                betas_ext = self._expand_betas(body_pose, betas)

                smpl_output = self.body_model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas_ext,
                    transl=transl)

                model_joints = smpl_output.joints

                # TODO: add support for 45 smpl joints
                model_joints = model_joints[:, :24, :]

                loss_dict = self._compute_loss(
                    model_joints,
                    keypoints_2d=keypoints_2d,
                    keypoints_conf_2d=keypoints_conf_2d,
                    keypoints_2d_weight=keypoints_2d_weight,
                    keypoints_3d=keypoints_3d,
                    keypoints_conf_3d=keypoints_conf_3d,
                    keypoints_3d_weight=keypoints_3d_weight,
                    joint_prior_weight=joint_prior_weight,
                    shape_prior_weight=shape_prior_weight,
                    joint_weights=joint_weights,
                    body_pose=body_pose,
                    betas=betas_ext)

                loss = loss_dict['total_loss']
                loss.backward()
                return loss

            optimizer.step(closure)

    def _get_weight(self, body_weight=1.0, use_shoulder_hip_only=False):

        weight = torch.ones([24]).to(self.device)

        if use_shoulder_hip_only:
            weight[[1, 2, 16, 17]] = weight[[1, 2, 16, 17]] + 1.0
            weight = weight - 1.0
            weight[[1, 2, 16, 17]] = weight[[1, 2, 16, 17]] * body_weight
        else:
            weight = weight * body_weight

        return weight

    def _compute_loss(self,
                      model_joints,
                      keypoints_2d=None,
                      keypoints_conf_2d=None,
                      keypoints_3d=None,
                      keypoints_conf_3d=None,
                      keypoints_2d_weight=None,
                      keypoints_3d_weight=None,
                      shape_prior_weight=None,
                      joint_prior_weight=None,
                      joint_weights={},
                      body_pose=None,
                      betas=None):

        total_loss = 0

        weight = self._get_weight(**joint_weights)

        # 2D keypoint loss
        if keypoints_2d is not None:
            projected_joints = self.camera(model_joints)
            # reprojection_error = gmof(
            #     projected_joints - keypoints_2d, sigma=100)
            # joints_weights = torch.ones_like(
            #     keypoints_conf_2d) * keypoints_2d_weight
            # reprojection_weight = (joints_weights * keypoints_conf_2d)**2
            # reprojection_loss = reprojection_weight * reprojection_error.sum(
            #     dim=-1)
            # total_loss = total_loss + reprojection_loss.sum(
            #     dim=-1) * keypoints_2d_weight
            reprojection_loss = self.keypoints_2d_mse_loss(
                pred=projected_joints,
                target=keypoints_2d,
                target_conf=keypoints_conf_2d,
                weight=weight,
                loss_weight_override=keypoints_2d_weight)
            total_loss = total_loss + reprojection_loss

        # 3D keypoint loss
        if keypoints_3d is not None:
            # keypoints_3d_weight = 1.0 if keypoints_3d_weight is None
            # else keypoints_3d_weight
            # joint_diff_3d = gmof(model_joints - keypoints_3d, sigma=100)
            # joints_weights = torch.ones_like(
            #     keypoints_conf_3d) * keypoints_3d_weight
            # joint_loss_3d_weight = (joints_weights * keypoints_conf_3d)**2
            # joint_loss_3d = joint_loss_3d_weight * joint_diff_3d.sum(dim=-1)
            # total_loss = total_loss + joint_loss_3d.sum(
            #     dim=-1) * keypoints_3d_weight
            # keypoints_3d_loss = joint_loss_3d.sum(dim=-1) *
            # keypoints_3d_weight  # just to print
            keypoints_3d_loss = self.keypoints_3d_mse_loss(
                pred=model_joints,
                target=keypoints_3d,
                target_conf=keypoints_conf_3d,
                weight=weight,
                loss_weight_override=keypoints_3d_weight)
            total_loss = total_loss + keypoints_3d_loss

        # Regularizer to prevent betas from taking large values
        shape_prior_loss = self.shape_prior_loss(
            betas=betas, loss_weight_override=shape_prior_weight)
        total_loss = total_loss + shape_prior_loss

        # Angle prior for knees and elbows
        # joint_prior_loss = (joint_prior_weight ** 2) * \
        # joint_prior(body_pose, spine=True).sum(dim=-1)
        # total_loss = total_loss + joint_prior_loss
        joint_prior_loss = self.joint_prior_loss(
            body_pose=body_pose, loss_weight_override=joint_prior_weight)
        total_loss = total_loss + joint_prior_loss

        # Smooth body
        # TODO: temp disable body_pose_loss
        # theta = body_pose.reshape(body_pose.shape[0], -1, 3)
        # rot_6d = matrix_to_rotation_6d(axis_angle_to_matrix(theta))
        # rot_6d_diff = rot_6d[1:] - rot_6d[:-1]
        # smooth_body_loss = rot_6d_diff.abs().sum(dim=-1)
        # smooth_body_loss = torch.cat(
        #     [torch.zeros(1, smooth_body_loss.shape[1],
        #                  device=body_pose.device,
        #                  dtype=smooth_body_loss.dtype),
        #      smooth_body_loss]
        # ).mean(dim=-1)

        # batch_size = keypoints_3d.shape[0]
        # print(f'3D Loss={keypoints_3d_loss.sum().item()/batch_size:.6f};',
        #       f'Shape Loss={shape_prior_loss.item()/batch_size:.6f};',
        #       f'joint_prior_loss={joint_prior_loss.item()/batch_size:.6f};')

        return {
            'total_loss': total_loss.sum(),
        }


class SMPLifyX(SMPLify):
    """Re-implementation of SMPLify-X with extended features."""

    def __init__(self,
                 body_model=None,
                 keypoints_2d_weight=1.0,
                 keypoints_3d_weight=1.0,
                 use_one_betas_per_video=False,
                 num_epochs=20,
                 camera=default_camera,
                 stage_config=smplifyx_stages,
                 opt_config=smplifyx_opt_config,
                 keypoints_2d_loss_config=keypoints_2d_loss_config,
                 keypoints_3d_loss_config=keypoints_3d_loss_config,
                 shape_prior_loss_config=shape_prior_loss_config,
                 joint_prior_loss_config=joint_prior_loss_config,
                 device=torch.device('cuda'),
                 verbose=False):
        super(SMPLifyX, self).__init__(
            body_model=body_model,
            keypoints_2d_weight=keypoints_2d_weight,
            keypoints_3d_weight=keypoints_3d_weight,
            use_one_betas_per_video=use_one_betas_per_video,
            num_epochs=num_epochs,
            camera=camera,
            stage_config=stage_config,
            opt_config=opt_config,
            keypoints_2d_loss_config=keypoints_2d_loss_config,
            keypoints_3d_loss_config=keypoints_3d_loss_config,
            shape_prior_loss_config=shape_prior_loss_config,
            joint_prior_loss_config=joint_prior_loss_config,
            device=device,
            verbose=verbose)

    def __call__(self,
                 keypoints_2d=None,
                 keypoints_conf_2d=1.0,
                 keypoints_3d=None,
                 keypoints_conf_3d=1.0,
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

        assert keypoints_2d is not None or keypoints_3d is not None, \
            'Neither of 2D nor 3D keypoints groud truth is provided.'
        if batch_size is None:
            batch_size = keypoints_2d.shape[
                0] if keypoints_2d is not None else keypoints_3d.shape[0]
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
            for stage_name, stage_config in self.stage_config.items():
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
                    keypoints_2d=keypoints_2d,
                    keypoints_conf_2d=keypoints_conf_2d,
                    keypoints_3d=keypoints_3d,
                    keypoints_conf_3d=keypoints_conf_3d,
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

    def _get_weight(self,
                    body_weight=1.0,
                    use_shoulder_hip_only=False,
                    hand_weight=1.0,
                    face_weight=1.0):

        weight = torch.ones([144]).to(self.device)

        if use_shoulder_hip_only:
            weight[[1, 2, 16, 17]] = weight[[1, 2, 16, 17]] + 1.0
            weight = weight - 1.0
            weight[[1, 2, 16, 17]] = weight[[1, 2, 16, 17]] * body_weight
        else:
            weight[:22] = weight[:22] * body_weight
            weight[60:66] = weight[60:66] * body_weight

        weight[25:54] = weight[25:54] * hand_weight
        weight[66:76] = weight[66:76] * hand_weight

        weight[22:25] = weight[22:25] * face_weight
        weight[55:60] = weight[55:60] * face_weight
        weight[76:] = weight[76:] * face_weight

        return weight

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
                        keypoints_2d=None,
                        keypoints_conf_2d=None,
                        keypoints_2d_weight=None,
                        keypoints_3d=None,
                        keypoints_conf_3d=None,
                        keypoints_3d_weight=None,
                        joint_prior_weight=None,
                        shape_prior_weight=None,
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

        optimizer = build_optimizer(parameters, self.opt_config)

        for iter_idx in range(num_iter):

            def closure():
                # body_pose_fixed = use_reference_spine(body_pose,
                # init_body_pose)

                optimizer.zero_grad()
                betas_ext = self._expand_betas(body_pose, betas)

                output = self.body_model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas_ext,
                    transl=transl,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    expression=expression,
                    jaw_pose=jaw_pose,
                    leye_pose=leye_pose,
                    reye_pose=reye_pose)

                model_joints = output.joints

                loss_dict = self._compute_loss(
                    model_joints,
                    keypoints_2d=keypoints_2d,
                    keypoints_conf_2d=keypoints_conf_2d,
                    keypoints_2d_weight=keypoints_2d_weight,
                    keypoints_3d=keypoints_3d,
                    keypoints_conf_3d=keypoints_conf_3d,
                    keypoints_3d_weight=keypoints_3d_weight,
                    joint_prior_weight=joint_prior_weight,
                    shape_prior_weight=shape_prior_weight,
                    joint_weights=joint_weights,
                    body_pose=body_pose,
                    betas=betas_ext)

                loss = loss_dict['total_loss']
                loss.backward()
                return loss

            optimizer.step(closure)

""" TODO
1. use from losses.py for loss computation
2. add camera
3. add 2D loss support
4. add GMM prior

optional:
1. optimize how to get batch_size and num_frames
2. add default body model
3. add model inference for param init
4. check if SMPL layer is better
"""

import torch
from configs.smplify.smplify import smplify_opt_config, smplify_stages
from configs.smplify.smplifyx import smplifyx_opt_config, smplifyx_stages
from mmcv.runner import build_optimizer

# TODO: placeholder
default_camera = {}


def gmof(x, sigma):
    """Geman-McClure error function."""
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


class OptimizableParameters():
    """Collects parameters for optimization."""

    def __init__(self):
        self.opt_params = []

    def set_param(self, fit_param, param):
        """Set require_grads and collect parameters for optimization.

        :param fit_param: (bool) True if optimizable parameters
        :param param: (torch.Tenosr) parameters
        """
        if fit_param:
            param.require_grads = True
            self.opt_params.append(param)
        else:
            param.require_grads = False

    def parameters(self):
        """Returns parameters.

        Compatible with mmcv's build_parameters()
        """
        return self.opt_params


def angle_prior(pose, spine=False):
    """Angle prior that penalizes unnatural bending of the knees and elbows."""
    # We subtract 3 because pose does not include the global rotation of
    # the model
    angle_loss = torch.exp(
        pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] *
        torch.tensor([1., -1., -1, -1.], device=pose.device))**2
    if spine:  # limit rotation of 3 spines
        spine_poses = pose[:, [
            9 - 3, 10 - 3, 11 - 3, 18 - 3, 19 - 3, 20 - 3, 27 - 3, 28 - 3, 29 -
            3
        ]]
        spine_loss = torch.exp(torch.abs(spine_poses))**2
        angle_loss = torch.cat([angle_loss, spine_loss], axis=1)
    return angle_loss


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
            param.require_grads = True
            opt_param.append(param)
        else:
            param.require_grads = False

    def _expand_betas(self, pose, betas):
        batch_size = pose.shape[0]
        num_video = betas.shape[0]
        if batch_size == num_video:
            return betas

        video_size = batch_size // num_video
        betas_ext = torch.zeros(
            batch_size, betas.shape[-1], device=betas.device)
        for i in range(num_video):
            betas_ext[i * video_size:(i + 1) * video_size] = betas[i]

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
                        keypoints_2d_weight=1.0,
                        keypoints_3d=None,
                        keypoints_conf_3d=None,
                        keypoints_3d_weight=1.0,
                        shape_prior_weight=1.0,
                        angle_prior_weight=1.0,
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
                betas_ext = self._arrange_betas(body_pose, betas)

                smpl_output = self.body_model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas_ext,
                    transl=transl)

                model_joints = smpl_output.joints

                loss_dict = self._compute_loss(
                    model_joints,
                    keypoints_2d=keypoints_2d,
                    keypoints_conf_2d=keypoints_conf_2d,
                    keypoints_2d_weight=keypoints_2d_weight,
                    keypoints_3d=keypoints_3d,
                    keypoints_conf_3d=keypoints_conf_3d,
                    keypoints_3d_weight=keypoints_3d_weight,
                    shape_prior_weight=shape_prior_weight,
                    angle_prior_weight=angle_prior_weight,
                    body_pose=body_pose,
                    betas=betas_ext)

                loss = loss_dict['total_loss']
                loss.backward()
                return loss

            optimizer.step(closure)

    def _compute_loss(self,
                      model_joints,
                      keypoints_2d=None,
                      keypoints_conf_2d=None,
                      keypoints_2d_weight=1.0,
                      keypoints_3d=None,
                      keypoints_conf_3d=None,
                      keypoints_3d_weight=1.0,
                      shape_prior_weight=1.0,
                      angle_prior_weight=1.0,
                      body_pose=None,
                      betas=None):

        total_loss = 0

        # 2D keypoint loss
        if keypoints_2d is not None:
            projected_joints = self.camera(model_joints)
            reprojection_error = gmof(
                projected_joints - keypoints_2d, sigma=100)
            joints_weights = torch.ones_like(
                keypoints_conf_2d) * keypoints_2d_weight
            reprojection_weight = (joints_weights * keypoints_conf_2d)**2
            reprojection_loss = reprojection_weight * reprojection_error.sum(
                dim=-1)
            total_loss = total_loss + reprojection_loss.sum(
                dim=-1) * keypoints_2d_weight

        # 3D keypoint loss
        # TODO: proper sigma for keypoints3d
        if keypoints_3d is not None:
            joint_diff_3d = gmof(model_joints - keypoints_3d, sigma=100)
            joints_weights = torch.ones_like(
                keypoints_conf_3d) * keypoints_3d_weight
            joint_loss_3d_weight = (joints_weights * keypoints_conf_3d)**2
            joint_loss_3d = joint_loss_3d_weight * joint_diff_3d.sum(dim=-1)
            total_loss = total_loss + joint_loss_3d.sum(
                dim=-1) * keypoints_3d_weight

        # Regularizer to prevent betas from taking large values
        if betas is not None:
            shape_prior_loss = (shape_prior_weight**2) * \
                               (betas**2).sum(dim=-1)
            total_loss = total_loss + shape_prior_loss

        # Angle prior for knees and elbows
        # TODO: temp disable angle_prior_loss
        # angle_prior_loss = (angle_prior_weight ** 2) * \
        # angle_prior(body_pose, spine=True).sum(dim=-1)
        # total_loss = total_loss + angle_prior_loss

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

        return {
            'total_loss': total_loss.sum(),
        }

    def _arrange_betas(self, pose, betas):
        batch_size = pose.shape[0]
        num_video = betas.shape[0]

        video_size = batch_size // num_video
        betas_ext = torch.zeros(
            batch_size, betas.shape[-1], device=betas.device)
        for i in range(num_video):
            betas_ext[i * video_size:(i + 1) * video_size] = betas[i]

        return betas_ext


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
                        keypoints_2d_weight=1.0,
                        keypoints_3d=None,
                        keypoints_conf_3d=None,
                        keypoints_3d_weight=1.0,
                        shape_prior_weight=1.0,
                        angle_prior_weight=1.0,
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
                betas_ext = self._arrange_betas(body_pose, betas)

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
                    shape_prior_weight=shape_prior_weight,
                    angle_prior_weight=angle_prior_weight,
                    body_pose=body_pose,
                    betas=betas_ext)

                loss = loss_dict['total_loss']
                loss.backward()
                return loss

            optimizer.step(closure)

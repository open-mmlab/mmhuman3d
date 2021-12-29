from typing import List, Tuple, Union

import torch
from mmcv.runner import build_optimizer

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.models.builder import REGISTRANTS, build_body_model, build_loss


class OptimizableParameters():
    """Collects parameters for optimization."""

    def __init__(self):
        self.opt_params = []

    def set_param(self, fit_param: torch.Tensor, param: torch.Tensor) -> None:
        """Set requires_grad and collect parameters for optimization.

        Args:
            fit_param: whether to optimize this body model parameter
            param: body model parameter

        Returns:
            None
        """
        if fit_param:
            param.requires_grad = True
            self.opt_params.append(param)
        else:
            param.requires_grad = False

    def parameters(self) -> List[torch.Tensor]:
        """Returns parameters. Compatible with mmcv's build_parameters()

        Returns:
            opt_params: a list of body model parameters for optimization
        """
        return self.opt_params


@REGISTRANTS.register_module()
class SMPLify(object):
    """Re-implementation of SMPLify with extended features.

    - video input
    - 3D keypoints
    """

    def __init__(
        self,
        body_model: Union[dict, torch.nn.Module],
        num_epochs: int = 20,
        camera: Union[dict, torch.nn.Module] = None,
        img_res: Union[Tuple[int], int] = 224,
        stages: dict = None,
        optimizer: dict = None,
        keypoints2d_loss: dict = None,
        keypoints3d_loss: dict = None,
        shape_prior_loss: dict = None,
        joint_prior_loss: dict = None,
        smooth_loss: dict = None,
        pose_prior_loss: dict = None,
        use_one_betas_per_video: bool = False,
        ignore_keypoints: List[int] = None,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose: bool = False,
    ) -> None:
        """
        Args:
            body_model: config or an object of body model.
            num_epochs: number of epochs of registration
            camera: config or an object of camera
            img_res: image resolution. If tuple, values are (width, height)
            stages: config of registration stages
            optimizer: config of optimizer
            keypoints2d_loss: config of keypoint 2D loss
            keypoints3d_loss: config of keypoint 3D loss
            shape_prior_loss: config of shape prior loss.
                Used to prevent extreme shapes.
            joint_prior_loss: config of joint prior loss.
                Used to prevent large joint rotations.
            smooth_loss: config of smooth loss.
                Used to prevent jittering by temporal smoothing.
            pose_prior_loss: config of pose prior loss.
                Used to prevent
            use_one_betas_per_video: whether to use the same beta parameters
                for all frames in a single video sequence.
            ignore_keypoints: list of keypoint names to ignore in keypoint
                loss computation
            device: torch device
            verbose: whether to print individual losses during registration

        Returns:
            None
        """

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
        if isinstance(body_model, dict):
            self.body_model = build_body_model(body_model).to(self.device)
        elif isinstance(body_model, torch.nn.Module):
            self.body_model = body_model.to(self.device)
        else:
            raise TypeError(f'body_model should be either dict or '
                            f'torch.nn.Module, but got {type(body_model)}')

        # initialize camera
        if camera is not None:
            if isinstance(camera, dict):
                self.camera = build_cameras(camera).to(self.device)
            elif isinstance(camera, torch.nn.Module):
                self.camera = camera.to(device)
            else:
                raise TypeError(f'camera should be either dict or '
                                f'torch.nn.Module, but got {type(camera)}')

        self.ignore_keypoints = ignore_keypoints
        self.verbose = verbose

        self._set_keypoint_idxs()

    def __call__(self,
                 keypoints2d: torch.Tensor = None,
                 keypoints2d_conf: torch.Tensor = None,
                 keypoints3d: torch.Tensor = None,
                 keypoints3d_conf: torch.Tensor = None,
                 init_global_orient: torch.Tensor = None,
                 init_transl: torch.Tensor = None,
                 init_body_pose: torch.Tensor = None,
                 init_betas: torch.Tensor = None,
                 return_verts: bool = False,
                 return_joints: bool = False,
                 return_full_pose: bool = False,
                 return_losses: bool = False) -> dict:
        """Run registration.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension
            Provide only keypoints2d or keypoints3d, not both.

        Args:
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            init_global_orient: initial global_orient of shape (B, 3)
            init_transl: initial transl of shape (B, 3)
            init_body_pose: initial body_pose of shape (B, 69)
            init_betas: initial betas of shape (B, D)
            return_verts: whether to return vertices
            return_joints: whether to return joints
            return_full_pose: whether to return full pose
            return_losses: whether to return loss dict

        Returns:
            ret: a dictionary that includes body model parameters,
                and optional attributes such as vertices and joints
        """
        assert keypoints2d is not None or keypoints3d is not None, \
            'Neither of 2D nor 3D keypoints are provided.'
        assert not (keypoints2d is not None and keypoints3d is not None), \
            'Do not provide both 2D and 3D keypoints.'
        batch_size = keypoints2d.shape[0] if keypoints2d is not None \
            else keypoints3d.shape[0]

        global_orient = self._match_init_batch_size(
            init_global_orient, self.body_model.global_orient, batch_size)
        transl = self._match_init_batch_size(init_transl,
                                             self.body_model.transl,
                                             batch_size)
        body_pose = self._match_init_batch_size(init_body_pose,
                                                self.body_model.body_pose,
                                                batch_size)
        if init_betas is None and self.use_one_betas_per_video:
            betas = torch.zeros(1, self.body_model.betas.shape[-1]).to(
                self.device)
        else:
            betas = self._match_init_batch_size(init_betas,
                                                self.body_model.betas,
                                                batch_size)

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
                        betas: torch.Tensor,
                        body_pose: torch.Tensor,
                        global_orient: torch.Tensor,
                        transl: torch.Tensor,
                        fit_global_orient: bool = True,
                        fit_transl: bool = True,
                        fit_body_pose: bool = True,
                        fit_betas: bool = True,
                        keypoints2d: torch.Tensor = None,
                        keypoints2d_conf: torch.Tensor = None,
                        keypoints2d_weight: float = None,
                        keypoints3d: torch.Tensor = None,
                        keypoints3d_conf: torch.Tensor = None,
                        keypoints3d_weight: float = None,
                        shape_prior_weight: float = None,
                        joint_prior_weight: float = None,
                        smooth_loss_weight: float = None,
                        pose_prior_weight: float = None,
                        joint_weights: dict = {},
                        num_iter: int = 1) -> None:
        """Optimize a stage of body model parameters according to
        configuration.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Args:
            betas: shape (B, D)
            body_pose: shape (B, 69)
            global_orient: shape (B, 3)
            transl: shape (B, 3)
            fit_global_orient: whether to optimize global_orient
            fit_transl: whether to optimize transl
            fit_body_pose: whether to optimize body_pose
            fit_betas: whether to optimize betas
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints2d_weight: weight of 2D keypoint loss
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            keypoints3d_weight: weight of 3D keypoint loss
            shape_prior_weight: weight of shape prior loss
            joint_prior_weight: weight of joint prior loss
            smooth_loss_weight: weight of smooth loss
            pose_prior_weight: weight of pose prior loss
            joint_weights: per joint weight of shape (K, )
            num_iter: number of iterations

        Returns:
            None
        """

        parameters = OptimizableParameters()
        parameters.set_param(fit_global_orient, global_orient)
        parameters.set_param(fit_transl, transl)
        parameters.set_param(fit_body_pose, body_pose)
        parameters.set_param(fit_betas, betas)

        optimizer = build_optimizer(parameters, self.optimizer)

        for iter_idx in range(num_iter):

            def closure():
                optimizer.zero_grad()
                betas_video = self._expand_betas(body_pose.shape[0], betas)

                loss_dict = self.evaluate(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas_video,
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
        betas: torch.Tensor = None,
        body_pose: torch.Tensor = None,
        global_orient: torch.Tensor = None,
        transl: torch.Tensor = None,
        keypoints2d: torch.Tensor = None,
        keypoints2d_conf: torch.Tensor = None,
        keypoints2d_weight: float = None,
        keypoints3d: torch.Tensor = None,
        keypoints3d_conf: torch.Tensor = None,
        keypoints3d_weight: float = None,
        shape_prior_weight: float = None,
        joint_prior_weight: float = None,
        smooth_loss_weight: float = None,
        pose_prior_weight: float = None,
        joint_weights: dict = {},
        return_verts: bool = False,
        return_full_pose: bool = False,
        return_joints: bool = False,
        reduction_override: str = None,
    ) -> dict:
        """Evaluate fitted parameters through loss computation. This function
        serves two purposes: 1) internally, for loss backpropagation 2)
        externally, for fitting quality evaluation.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Args:
            betas: shape (B, D)
            body_pose: shape (B, 69)
            global_orient: shape (B, 3)
            transl: shape (B, 3)
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints2d_weight: weight of 2D keypoint loss
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            keypoints3d_weight: weight of 3D keypoint loss
            shape_prior_weight: weight of shape prior loss
            joint_prior_weight: weight of joint prior loss
            smooth_loss_weight: weight of smooth loss
            pose_prior_weight: weight of pose prior loss
            joint_weights: per joint weight of shape (K, )
            return_verts: whether to return vertices
            return_joints: whether to return joints
            return_full_pose: whether to return full pose
            reduction_override: reduction method, e.g., 'none', 'sum', 'mean'

        Returns:
            ret: a dictionary that includes body model parameters,
                and optional attributes such as vertices and joints
        """

        ret = {}

        body_model_output = self.body_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            return_verts=return_verts,
            return_full_pose=return_full_pose)

        model_joints = body_model_output['joints']
        model_joint_mask = body_model_output['joint_mask']

        loss_dict = self._compute_loss(
            model_joints,
            model_joint_mask,
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
                      model_joints: torch.Tensor,
                      model_joint_conf: torch.Tensor,
                      keypoints2d: torch.Tensor = None,
                      keypoints2d_conf: torch.Tensor = None,
                      keypoints2d_weight: float = None,
                      keypoints3d: torch.Tensor = None,
                      keypoints3d_conf: torch.Tensor = None,
                      keypoints3d_weight: float = None,
                      shape_prior_weight: float = None,
                      joint_prior_weight: float = None,
                      smooth_loss_weight: float = None,
                      pose_prior_weight: float = None,
                      joint_weights: dict = {},
                      reduction_override: str = None,
                      body_pose: torch.Tensor = None,
                      betas: torch.Tensor = None):
        """Loss computation.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Args:
            model_joints: 3D joints regressed from body model of shape (B, K)
            model_joint_conf: 3D joint confidence of shape (B, K). It is
                normally all 1, except for zero-pads due to convert_kps in
                the SMPL wrapper.
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints2d_weight: weight of 2D keypoint loss
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            keypoints3d_weight: weight of 3D keypoint loss
            shape_prior_weight: weight of shape prior loss
            joint_prior_weight: weight of joint prior loss
            smooth_loss_weight: weight of smooth loss
            pose_prior_weight: weight of pose prior loss
            joint_weights: per joint weight of shape (K, )
            reduction_override: reduction method, e.g., 'none', 'sum', 'mean'
            body_pose: shape (B, 69), for loss computation
            betas: shape (B, D), for loss computation

        Returns:
            losses: a dict that contains all losses
        """
        losses = {}

        weight = self._get_weight(**joint_weights)

        # 2D keypoint loss
        if keypoints2d is not None:
            # bs = model_joints.shape[0]
            # projected_joints = perspective_projection(
            #     model_joints,
            #     torch.eye(3).expand((bs, 3, 3)).to(model_joints.device),
            #     torch.zeros((bs, 3)).to(model_joints.device), 5000.0,
            #     torch.Tensor([self.img_res / 2,
            #                   self.img_res / 2]).to(model_joints.device))
            projected_joints_xyd = self.camera.transform_points_screen(
                model_joints)
            projected_joints = projected_joints_xyd[..., :2]

            # normalize keypoints to [-1,1]
            projected_joints = 2 * projected_joints / (self.img_res - 1) - 1
            keypoints2d = 2 * keypoints2d / (self.img_res - 1) - 1

            keypoint2d_loss = self.keypoints2d_mse_loss(
                pred=projected_joints,
                pred_conf=model_joint_conf,
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
                pred_conf=model_joint_conf,
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

    def _match_init_batch_size(self, init_param: torch.Tensor,
                               init_param_body_model: torch.Tensor,
                               batch_size: int) -> torch.Tensor:
        """A helper function to ensure body model parameters have the same
        batch size as the input keypoints.

        Args:
            init_param: input initial body model parameters, may be None
            init_param_body_model: initial body model parameters from the
                body model
            batch_size: batch size of keypoints

        Returns:
            param: body model parameters with batch size aligned
        """

        # param takes init values
        param = init_param.detach().clone() \
            if init_param is not None \
            else init_param_body_model.detach().clone()

        # expand batch dimension to match batch size
        param_batch_size = param.shape[0]
        if param_batch_size != batch_size:
            if param_batch_size == 1:
                param = param.repeat(batch_size, *[1] * (param.ndim - 1))
            else:
                raise ValueError('Init param does not match the batch size of '
                                 'keypoints, and is not 1.')

        # shape check
        assert param.shape[0] == batch_size
        assert param.shape[1:] == init_param_body_model.shape[1:], \
            f'Shape mismatch: {param.shape} vs {init_param_body_model.shape}'

        return param

    def _set_keypoint_idxs(self) -> None:
        """Set keypoint indices to 1) body parts to be assigned different
        weights 2) be ignored for keypoint loss computation.

        Returns:
            None
        """
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

    def _get_weight(self,
                    use_shoulder_hip_only: bool = False,
                    body_weight: float = 1.0) -> torch.Tensor:
        """Get per keypoint weight.

        Notes:
            K: number of keypoints

        Args:
            use_shoulder_hip_only: whether to use only shoulder and hip
                keypoints for loss computation. This is useful in the
                warming-up stage to find a reasonably good initialization.
            body_weight: weight of body keypoints. Body part segmentation
                definition is included in the HumanData convention.

        Returns:
            weight: per keypoint weight tensor of shape (K)
        """

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

    def _expand_betas(self, batch_size, betas):
        """A helper function to expand the betas's first dim to match batch
        size such that the same beta parameters can be used for all frames in a
        video sequence.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Args:
            batch_size: batch size
            betas: shape (B, D)

        Returns:
            betas_video: expanded betas
        """
        # no expansion needed
        if batch_size == betas.shape[0]:
            return betas

        # first dim is 1
        else:
            feat_dim = betas.shape[-1]
            betas_video = betas.view(1, feat_dim).expand(batch_size, feat_dim)

        return betas_video

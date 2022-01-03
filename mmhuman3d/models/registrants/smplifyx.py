import torch
from mmcv.runner import build_optimizer

from mmhuman3d.core.conventions.keypoints_mapping import (
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.models.builder import REGISTRANTS
from .smplify import OptimizableParameters, SMPLify


@REGISTRANTS.register_module()
class SMPLifyX(SMPLify):
    """Re-implementation of SMPLify-X with extended features.

    - video input
    - 3D keypoints
    """

    def __call__(self,
                 keypoints2d: torch.Tensor = None,
                 keypoints2d_conf: torch.Tensor = None,
                 keypoints3d: torch.Tensor = None,
                 keypoints3d_conf: torch.Tensor = None,
                 init_global_orient: torch.Tensor = None,
                 init_transl: torch.Tensor = None,
                 init_body_pose: torch.Tensor = None,
                 init_betas: torch.Tensor = None,
                 init_left_hand_pose: torch.Tensor = None,
                 init_right_hand_pose: torch.Tensor = None,
                 init_expression: torch.Tensor = None,
                 init_jaw_pose: torch.Tensor = None,
                 init_leye_pose: torch.Tensor = None,
                 init_reye_pose: torch.Tensor = None,
                 return_verts: bool = False,
                 return_joints: bool = False,
                 return_full_pose: bool = False,
                 return_losses: bool = False) -> dict:
        """Run registration.

        Notes:
            B: batch size
            K: number of keypoints
            D: body shape dimension
            D_H: hand pose dimension
            D_E: expression dimension
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
            init_left_hand_pose: initial left hand pose of shape (B, D_H)
            init_right_hand_pose: initial right hand pose of shape (B, D_H)
            init_expression: initial left hand pose of shape (B, D_E)
            init_jaw_pose: initial jaw pose of shape (B, 3)
            init_leye_pose: initial left eye pose of shape (B, 3)
            init_reye_pose: initial right eye pose of shape (B, 3)
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
        left_hand_pose = self._match_init_batch_size(
            init_left_hand_pose, self.body_model.left_hand_pose, batch_size)
        right_hand_pose = self._match_init_batch_size(
            init_right_hand_pose, self.body_model.right_hand_pose, batch_size)
        expression = self._match_init_batch_size(init_expression,
                                                 self.body_model.expression,
                                                 batch_size)
        jaw_pose = self._match_init_batch_size(init_jaw_pose,
                                               self.body_model.jaw_pose,
                                               batch_size)
        leye_pose = self._match_init_batch_size(init_leye_pose,
                                                self.body_model.leye_pose,
                                                batch_size)
        reye_pose = self._match_init_batch_size(init_reye_pose,
                                                self.body_model.reye_pose,
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
                        betas: torch.Tensor,
                        body_pose: torch.Tensor,
                        global_orient: torch.Tensor,
                        transl: torch.Tensor,
                        left_hand_pose: torch.Tensor,
                        right_hand_pose: torch.Tensor,
                        expression: torch.Tensor,
                        jaw_pose: torch.Tensor,
                        leye_pose: torch.Tensor,
                        reye_pose: torch.Tensor,
                        fit_global_orient: bool = True,
                        fit_transl: bool = True,
                        fit_body_pose: bool = True,
                        fit_betas: bool = True,
                        fit_left_hand_pose: bool = True,
                        fit_right_hand_pose: bool = True,
                        fit_expression: bool = True,
                        fit_jaw_pose: bool = True,
                        fit_leye_pose: bool = True,
                        fit_reye_pose: bool = True,
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
            fit_left_hand_pose: whether to optimize left hand pose
            fit_right_hand_pose: whether to optimize right hand pose
            fit_expression: whether to optimize expression
            fit_jaw_pose: whether to optimize jaw pose
            fit_leye_pose: whether to optimize left eye pose
            fit_reye_pose: whether to optimize right eye pose
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
                betas_video = self._expand_betas(body_pose.shape[0], betas)

                loss_dict = self.evaluate(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas_video,
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
        betas=None,
        body_pose=None,
        global_orient=None,
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
        """Evaluate fitted parameters through loss computation. This function
        serves two purposes: 1) internally, for loss backpropagation 2)
        externally, for fitting quality evaluation.

        Notes:
            B: batch size
            K: number of keypoints
            D: body shape dimension
            D_H: hand pose dimension
            D_E: expression dimension

        Args:
            betas: shape (B, D)
            body_pose: shape (B, 69)
            global_orient: shape (B, 3)
            transl: shape (B, 3)
            left_hand_pose: shape (B, D_H)
            right_hand_pose: shape (B, D_H)
            expression: shape (B, D_E)
            jaw_pose: shape (B, 3)
            leye_pose: shape (B, 3)
            reye_pose: shape (B, 3)
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
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
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

    def _set_keypoint_idxs(self):
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

    def _get_weight(self,
                    use_shoulder_hip_only: bool = False,
                    body_weight: float = 1.0,
                    hand_weight: float = 1.0,
                    face_weight: float = 1.0):
        """Get per keypoint weight.

        Notes:
            K: number of keypoints

        Args:
            use_shoulder_hip_only: whether to use only shoulder and hip
                keypoints for loss computation. This is useful in the
                warming-up stage to find a reasonably good initialization.
            body_weight: weight of body keypoints. Body part segmentation
                definition is included in the HumanData convention.
            hand_weight: weight of hand keypoints.
            face_weight: weight of face keypoints.

        Returns:
            weight: per keypoint weight tensor of shape (K)
        """

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

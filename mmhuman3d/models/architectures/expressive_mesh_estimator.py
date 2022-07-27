from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmhuman3d.core.conventions.keypoints_mapping import (
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.utils.geometry import (
    batch_rodrigues,
    weak_perspective_projection,
)
from ..backbones.builder import build_backbone
from ..body_models.builder import build_body_model
from ..heads.builder import build_head
from ..losses.builder import build_loss
from ..necks.builder import build_neck
from ..utils import (
    SMPLXFaceCropFunc,
    SMPLXFaceMergeFunc,
    SMPLXHandCropFunc,
    SMPLXHandMergeFunc,
)
from .base_architecture import BaseArchitecture


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def pose2rotmat(pred_pose):
    """aa2rotmat."""
    if len(pred_pose.shape) == 3:
        num_joints = pred_pose.shape[1]
        pred_pose = batch_rodrigues(pred_pose.view(-1, 3)).view(
            -1, num_joints, 3, 3)
    return pred_pose


class SMPLXBodyModelEstimator(BaseArchitecture, metaclass=ABCMeta):
    """BodyModelEstimator Architecture.

    Args:
        backbone (dict | None, optional): Backbone config dict. Default: None.
        neck (dict | None, optional): Neck config dict. Default: None
        head (dict | None, optional): Regressor config dict. Default: None.
        body_model_train (dict | None, optional): SMPL config dict during
            training. Default: None.
        body_model_test (dict | None, optional): SMPL config dict during
            test. Default: None.
        convention (str, optional): Keypoints convention. Default: "human_data"
        loss_keypoints2d (dict | None, optional): Losses config dict for
            2D keypoints. Default: None.
        loss_keypoints3d (dict | None, optional): Losses config dict for
            3D keypoints. Default: None.
        loss_smplx_global_orient (dict | None, optional): Losses config dict
            for smplx global orient. Default: None
        loss_smplx_body_pose (dict | None, optional): Losses config dict
            for smplx body pose. Default: None
        loss_smplx_hand_pose (dict | None, optional): Losses config dict
            for smplx hand pose. Default: None
        loss_smplx_jaw_pose (dict | None, optional): Losses config dict
            for smplx jaw pose. Default: None
        loss_smplx_expression (dict | None, optional): Losses config dict
            for smplx expression. Default: None
        loss_smplx_betas (dict | None, optional): Losses config dict for smplx
            betas. Default: None
        loss_camera (dict | None, optional): Losses config dict for predicted
            camera parameters. Default: None
        extra_hand_model_cfg (dict | None, optional) : Hand model config for
            refining body model prediction. Default: None
        extra_face_model_cfg (dict | None, optional) : Face model config for
            refining body model prediction. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 backbone: Optional[Union[dict, None]] = None,
                 neck: Optional[Union[dict, None]] = None,
                 head: Optional[Union[dict, None]] = None,
                 body_model_train: Optional[Union[dict, None]] = None,
                 body_model_test: Optional[Union[dict, None]] = None,
                 convention: Optional[str] = 'human_data',
                 loss_keypoints2d: Optional[Union[dict, None]] = None,
                 loss_keypoints3d: Optional[Union[dict, None]] = None,
                 loss_smplx_global_orient: Optional[Union[dict, None]] = None,
                 loss_smplx_body_pose: Optional[Union[dict, None]] = None,
                 loss_smplx_hand_pose: Optional[Union[dict, None]] = None,
                 loss_smplx_jaw_pose: Optional[Union[dict, None]] = None,
                 loss_smplx_expression: Optional[Union[dict, None]] = None,
                 loss_smplx_betas: Optional[Union[dict, None]] = None,
                 loss_smplx_betas_prior: Optional[Union[dict, None]] = None,
                 loss_camera: Optional[Union[dict, None]] = None,
                 extra_hand_model_cfg: Optional[Union[dict, None]] = None,
                 extra_face_model_cfg: Optional[Union[dict, None]] = None,
                 frozen_batchnorm: bool = False,
                 init_cfg: Optional[Union[list, dict, None]] = None):
        super(SMPLXBodyModelEstimator, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

        if frozen_batchnorm:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = False

            self.backbone = FrozenBatchNorm2d.convert_frozen_batchnorm(
                self.backbone)
            self.head = FrozenBatchNorm2d.convert_frozen_batchnorm(self.head)

        self.body_model_train = build_body_model(body_model_train)
        self.body_model_test = build_body_model(body_model_test)
        self.convention = convention

        self.apply_hand_model = False
        self.apply_face_model = False
        if extra_hand_model_cfg is not None:
            self.hand_backbone = build_backbone(
                extra_hand_model_cfg.get('backbone', None))
            self.hand_neck = build_neck(extra_hand_model_cfg.get('neck', None))
            self.hand_head = build_head(extra_hand_model_cfg.get('head', None))
            crop_cfg = extra_hand_model_cfg.get('crop_cfg', None)
            if crop_cfg is not None:
                self.crop_hand_func = SMPLXHandCropFunc(
                    self.hand_head,
                    self.body_model_train,
                    convention=self.convention,
                    **crop_cfg)
                self.hand_merge_func = SMPLXHandMergeFunc(
                    self.body_model_train, self.convention)
            self.hand_crop_loss = build_loss(
                extra_hand_model_cfg.get('loss_hand_crop', None))
            self.apply_hand_model = True
            self.left_hand_idxs = get_keypoint_idxs_by_part(
                'left_hand', self.convention)
            self.left_hand_idxs.append(
                get_keypoint_idx('left_wrist', self.convention))
            self.left_hand_idxs = sorted(self.left_hand_idxs)
            self.right_hand_idxs = get_keypoint_idxs_by_part(
                'right_hand', self.convention)
            self.right_hand_idxs.append(
                get_keypoint_idx('right_wrist', self.convention))
            self.right_hand_idxs = sorted(self.right_hand_idxs)

        if extra_face_model_cfg is not None:
            self.face_backbone = build_backbone(
                extra_face_model_cfg.get('backbone', None))
            self.face_neck = build_neck(extra_face_model_cfg.get('neck', None))
            self.face_head = build_head(extra_face_model_cfg.get('head', None))
            crop_cfg = extra_face_model_cfg.get('crop_cfg', None)
            if crop_cfg is not None:
                self.crop_face_func = SMPLXFaceCropFunc(
                    self.face_head,
                    self.body_model_train,
                    convention=self.convention,
                    **crop_cfg)
                self.face_merge_func = SMPLXFaceMergeFunc(
                    self.body_model_train, self.convention)
            self.face_crop_loss = build_loss(
                extra_face_model_cfg.get('loss_face_crop', None))
            self.apply_face_model = True
            self.face_idxs = get_keypoint_idxs_by_part('head', self.convention)
            self.face_idxs = sorted(self.face_idxs)

        self.loss_keypoints2d = build_loss(loss_keypoints2d)
        self.loss_keypoints3d = build_loss(loss_keypoints3d)

        self.loss_smplx_global_orient = build_loss(loss_smplx_global_orient)
        self.loss_smplx_body_pose = build_loss(loss_smplx_body_pose)
        self.loss_smplx_hand_pose = build_loss(loss_smplx_hand_pose)
        self.loss_smplx_jaw_pose = build_loss(loss_smplx_jaw_pose)
        self.loss_smplx_expression = build_loss(loss_smplx_expression)
        self.loss_smplx_betas = build_loss(loss_smplx_betas)
        self.loss_smplx_betas_piror = build_loss(loss_smplx_betas_prior)
        self.loss_camera = build_loss(loss_camera)
        set_requires_grad(self.body_model_train, False)
        set_requires_grad(self.body_model_test, False)

    def train_step(self, data_batch, optimizer, **kwargs):
        """Train step function.

        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (dict[torch.optim.Optimizer]): Dict with optimizers for
                generator.
        Returns:
            outputs (dict): Dict with loss, information for logger,
            the number of samples.
        """
        if self.backbone is not None:
            img = data_batch['img']
            features = self.backbone(img)
        else:
            features = data_batch['features']

        if self.neck is not None:
            features = self.neck(features)

        predictions = self.head(features)
        if self.apply_hand_model:
            hand_input_img, hand_mean, hand_crop_info = self.crop_hand_func(
                predictions, data_batch['img_metas'])
            hand_features = self.hand_backbone(hand_input_img)
            if self.neck is not None:
                hand_features = self.hand_neck(hand_features)
            hand_predictions = self.hand_head(hand_features, cond=hand_mean)
            predictions = self.hand_merge_func(predictions, hand_predictions)
            predictions['hand_crop_info'] = hand_crop_info
        if self.apply_face_model:
            face_input_img, face_mean, face_crop_info = self.crop_face_func(
                predictions, data_batch['img_metas'])
            face_features = self.face_backbone(face_input_img)
            if self.neck is not None:
                face_features = self.face_neck(face_features)
            face_predictions = self.face_head(face_features, cond=face_mean)
            predictions = self.face_merge_func(predictions, face_predictions)
            predictions['face_crop_info'] = face_crop_info

        targets = self.prepare_targets(data_batch)

        losses = self.compute_losses(predictions, targets)

        loss, log_vars = self._parse_losses(losses)
        if self.backbone is not None:
            optimizer['backbone'].zero_grad()
        if self.neck is not None:
            optimizer['neck'].zero_grad()
        if self.head is not None:
            optimizer['head'].zero_grad()

        if self.apply_hand_model:
            if self.hand_backbone is not None:
                optimizer['hand_backbone'].zero_grad()
            if self.hand_neck is not None:
                optimizer['hand_neck'].zero_grad()
            if self.hand_head is not None:
                optimizer['hand_head'].zero_grad()

        if self.apply_face_model:
            if self.face_backbone is not None:
                optimizer['face_backbone'].zero_grad()
            if self.face_neck is not None:
                optimizer['face_neck'].zero_grad()
            if self.face_head is not None:
                optimizer['face_head'].zero_grad()

        loss.backward()
        if self.backbone is not None:
            optimizer['backbone'].step()
        if self.neck is not None:
            optimizer['neck'].step()
        if self.head is not None:
            optimizer['head'].step()

        if self.apply_hand_model:
            if self.hand_backbone is not None:
                optimizer['hand_backbone'].step()
            if self.hand_neck is not None:
                optimizer['hand_neck'].step()
            if self.hand_head is not None:
                optimizer['hand_head'].step()

        if self.apply_face_model:
            if self.face_backbone is not None:
                optimizer['face_backbone'].step()
            if self.face_neck is not None:
                optimizer['face_neck'].step()
            if self.face_head is not None:
                optimizer['face_head'].step()

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))
        return outputs

    def compute_keypoints3d_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            gt_keypoints3d: torch.Tensor,
            has_keypoints3d: Optional[torch.Tensor] = None):
        """Compute loss for 3d keypoints."""
        keypoints3d_conf = gt_keypoints3d[:, :, 3].float().unsqueeze(-1)
        keypoints3d_conf = keypoints3d_conf.repeat(1, 1, 3)
        pred_keypoints3d = pred_keypoints3d.float()
        gt_keypoints3d = gt_keypoints3d[:, :, :3].float()

        if has_keypoints3d is None:
            has_keypoints3d = torch.ones((keypoints3d_conf.shape[0]))
        if keypoints3d_conf[has_keypoints3d == 1].numel() == 0:
            return torch.Tensor([0]).type_as(gt_keypoints3d)
        # Center the predictions using the pelvis
        target_idxs = has_keypoints3d == 1
        pred_keypoints3d = pred_keypoints3d[target_idxs]
        gt_keypoints3d = gt_keypoints3d[target_idxs]
        pred_pelvis = pred_keypoints3d[:, [1, 2], :].mean(dim=1, keepdim=True)
        pred_keypoints3d = pred_keypoints3d - pred_pelvis
        gt_pelvis = gt_keypoints3d[:, [1, 2], :].mean(dim=1, keepdim=True)
        gt_keypoints3d = gt_keypoints3d - gt_pelvis

        loss = self.loss_keypoints3d(
            pred_keypoints3d,
            gt_keypoints3d,
            weight=keypoints3d_conf[target_idxs])
        loss /= gt_keypoints3d.shape[0]
        return loss

    def compute_keypoints2d_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            pred_cam: torch.Tensor,
            gt_keypoints2d: torch.Tensor,
            img_res: Optional[int] = 224,
            focal_length: Optional[int] = 5000,
            has_keypoints2d: Optional[torch.Tensor] = None):
        """Compute loss for 2d keypoints."""
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()
        if has_keypoints2d is None:
            has_keypoints2d = torch.ones((keypoints2d_conf.shape[0]))
        if keypoints2d_conf[has_keypoints2d == 1].numel() == 0:
            return torch.Tensor([0]).type_as(gt_keypoints2d)

        # Expose use weak_perspective_projection
        pred_keypoints2d = weak_perspective_projection(
            pred_keypoints3d,
            scale=pred_cam[:, 0],
            translation=pred_cam[:, 1:3])
        gt_keypoints2d = 2 * gt_keypoints2d / (img_res - 1) - 1

        target_idxs = has_keypoints2d == 1
        pred_keypoints2d = pred_keypoints2d[target_idxs]
        gt_keypoints2d = gt_keypoints2d[target_idxs]
        loss = self.loss_keypoints2d(
            pred_keypoints2d,
            gt_keypoints2d,
            weight=keypoints2d_conf[target_idxs])
        loss /= gt_keypoints2d.shape[0]
        return loss

    def compute_smplx_body_pose_loss(self, pred_rotmat: torch.Tensor,
                                     gt_pose: torch.Tensor,
                                     has_smplx_body_pose: torch.Tensor):
        """Compute loss for smplx body pose."""
        num_joints = pred_rotmat.shape[1]
        target_idxs = has_smplx_body_pose == 1
        if gt_pose[target_idxs].numel() == 0:
            return torch.Tensor([0]).type_as(gt_pose)

        gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(
            -1, num_joints, 3, 3)

        loss = self.loss_smplx_body_pose(pred_rotmat[target_idxs],
                                         gt_rotmat[target_idxs])
        return loss

    def compute_smplx_global_orient_loss(
            self, pred_rotmat: torch.Tensor, gt_global_orient: torch.Tensor,
            has_smplx_global_orient: torch.Tensor):
        """Compute loss for smplx global orient."""
        target_idxs = has_smplx_global_orient == 1
        if gt_global_orient[target_idxs].numel() == 0:
            return torch.Tensor([0]).type_as(gt_global_orient)

        gt_rotmat = batch_rodrigues(gt_global_orient.view(-1, 3)).view(
            -1, 1, 3, 3)

        loss = self.loss_smplx_global_orient(pred_rotmat[target_idxs],
                                             gt_rotmat[target_idxs])
        return loss

    def compute_smplx_jaw_pose_loss(self, pred_rotmat: torch.Tensor,
                                    gt_jaw_pose: torch.Tensor,
                                    has_smplx_jaw_pose: torch.Tensor,
                                    face_conf: torch.Tensor):
        """Compute loss for smplx jaw pose."""
        target_idxs = has_smplx_jaw_pose == 1
        if gt_jaw_pose[target_idxs].numel() == 0:
            return torch.Tensor([0]).type_as(gt_jaw_pose)

        gt_rotmat = batch_rodrigues(gt_jaw_pose.view(-1, 3)).view(-1, 1, 3, 3)
        conf = face_conf.mean(axis=1).float()
        conf = conf.view(-1, 1, 1, 1)

        loss = self.loss_smplx_jaw_pose(
            pred_rotmat[target_idxs],
            gt_rotmat[target_idxs],
            weight=conf[target_idxs])
        return loss

    def compute_smplx_hand_pose_loss(self, pred_rotmat: torch.Tensor,
                                     gt_hand_pose: torch.Tensor,
                                     has_smplx_hand_pose: torch.Tensor,
                                     hand_conf: torch.Tensor):
        """Compute loss for smplx left/right hand pose."""
        joint_num = pred_rotmat.shape[1]
        target_idxs = has_smplx_hand_pose == 1
        if gt_hand_pose[target_idxs].numel() == 0:
            return torch.Tensor([0]).type_as(gt_hand_pose)
        gt_rotmat = batch_rodrigues(gt_hand_pose.view(-1, 3)).view(
            -1, joint_num, 3, 3)
        conf = hand_conf.mean(
            axis=1, keepdim=True).float().expand(-1, joint_num)
        conf = conf.view(-1, joint_num, 1, 1)

        loss = self.loss_smplx_hand_pose(
            pred_rotmat[target_idxs],
            gt_rotmat[target_idxs],
            weight=conf[target_idxs])
        return loss

    def compute_smplx_betas_loss(self, pred_betas: torch.Tensor,
                                 gt_betas: torch.Tensor,
                                 has_smplx_betas: torch.Tensor):
        """Compute loss for smplx betas."""
        target_idxs = has_smplx_betas == 1
        if gt_betas[target_idxs].numel() == 0:
            return torch.Tensor([0]).type_as(gt_betas)

        loss = self.loss_smplx_betas(pred_betas[target_idxs],
                                     gt_betas[target_idxs])
        loss = loss / gt_betas[target_idxs].shape[0]
        return loss

    def compute_smplx_betas_prior_loss(self, pred_betas: torch.Tensor):
        """Compute prior loss for smplx betas."""
        loss = self.loss_smplx_betas_piror(pred_betas)
        return loss

    def compute_smplx_expression_loss(self, pred_expression: torch.Tensor,
                                      gt_expression: torch.Tensor,
                                      has_smplx_expression: torch.Tensor,
                                      face_conf: torch.Tensor):
        """Compute loss for smplx betas."""
        target_idxs = has_smplx_expression == 1
        if gt_expression[target_idxs].numel() == 0:
            return torch.Tensor([0]).type_as(gt_expression)
        conf = face_conf.mean(axis=1).float()
        conf = conf.view(-1, 1)

        loss = self.loss_smplx_expression(
            pred_expression[target_idxs],
            gt_expression[target_idxs],
            weight=conf[target_idxs])
        loss = loss / gt_expression[target_idxs].shape[0]
        return loss

    def compute_camera_loss(self, cameras: torch.Tensor):
        """Compute loss for predicted camera parameters."""
        loss = self.loss_camera(cameras)
        return loss

    def compute_face_crop_loss(self,
                               pred_keypoints3d: torch.Tensor,
                               pred_cam: torch.Tensor,
                               gt_keypoints2d: torch.Tensor,
                               face_crop_info: dict,
                               img_res: Optional[int] = 224,
                               has_keypoints2d: Optional[torch.Tensor] = None):
        """Compute face crop loss for 2d keypoints."""
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()
        if has_keypoints2d is None:
            has_keypoints2d = torch.ones((keypoints2d_conf.shape[0]))
        if keypoints2d_conf[has_keypoints2d == 1].numel() == 0:
            return torch.Tensor([0]).type_as(gt_keypoints2d)

        # Expose use weak_perspective_projection
        pred_keypoints2d = weak_perspective_projection(
            pred_keypoints3d,
            scale=pred_cam[:, 0],
            translation=pred_cam[:, 1:3])
        target_idxs = has_keypoints2d == 1
        pred_keypoints2d = pred_keypoints2d[target_idxs]
        gt_keypoints2d = gt_keypoints2d[target_idxs]

        pred_keypoints2d = (0.5 * pred_keypoints2d + 0.5) * (img_res - 1)
        face_inv_crop_transforms = face_crop_info['face_inv_crop_transforms']
        pred_keypoints2d_hd = torch.einsum('bij,bkj->bki', [
            face_inv_crop_transforms[:, :2, :2], pred_keypoints2d
        ]) + face_inv_crop_transforms[:, :2, 2].unsqueeze(dim=1)
        gt_keypoints2d_hd = torch.einsum('bij,bkj->bki', [
            face_inv_crop_transforms[:, :2, :2], gt_keypoints2d
        ]) + face_inv_crop_transforms[:, :2, 2].unsqueeze(dim=1)

        pred_face_keypoints_hd = pred_keypoints2d_hd[:, self.face_idxs]
        face_crop_transform = face_crop_info['face_crop_transform']
        inv_face_crop_transf = torch.inverse(face_crop_transform)
        face_img_keypoints = torch.einsum('bij,bkj->bki', [
            inv_face_crop_transf[:, :2, :2], pred_face_keypoints_hd
        ]) + inv_face_crop_transf[:, :2, 2].unsqueeze(dim=1)
        gt_face_keypoints_hd = gt_keypoints2d_hd[:, self.face_idxs]
        gt_face_keypoints = torch.einsum('bij,bkj->bki', [
            inv_face_crop_transf[:, :2, :2], gt_face_keypoints_hd
        ]) + inv_face_crop_transf[:, :2, 2].unsqueeze(dim=1)

        loss = self.face_crop_loss(
            face_img_keypoints,
            gt_face_keypoints,
            weight=keypoints2d_conf[target_idxs][:, self.face_idxs])
        loss /= gt_face_keypoints.shape[0]
        return loss

    def compute_hand_crop_loss(self,
                               pred_keypoints3d: torch.Tensor,
                               pred_cam: torch.Tensor,
                               gt_keypoints2d: torch.Tensor,
                               hand_crop_info: dict,
                               img_res: Optional[int] = 224,
                               has_keypoints2d: Optional[torch.Tensor] = None):
        """Compute hand crop loss for 2d keypoints."""
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()
        if has_keypoints2d is None:
            has_keypoints2d = torch.ones((keypoints2d_conf.shape[0]))
        if keypoints2d_conf[has_keypoints2d == 1].numel() == 0:
            return torch.Tensor([0]).type_as(gt_keypoints2d)

        # Expose use weak_perspective_projection
        pred_keypoints2d = weak_perspective_projection(
            pred_keypoints3d,
            scale=pred_cam[:, 0],
            translation=pred_cam[:, 1:3])
        target_idxs = has_keypoints2d == 1
        pred_keypoints2d = pred_keypoints2d[target_idxs]
        gt_keypoints2d = gt_keypoints2d[target_idxs]

        pred_keypoints2d = (0.5 * pred_keypoints2d + 0.5) * (img_res - 1)
        hand_inv_crop_transforms = hand_crop_info['hand_inv_crop_transforms']
        pred_keypoints2d_hd = torch.einsum('bij,bkj->bki', [
            hand_inv_crop_transforms[:, :2, :2], pred_keypoints2d
        ]) + hand_inv_crop_transforms[:, :2, 2].unsqueeze(dim=1)
        gt_keypoints2d_hd = torch.einsum('bij,bkj->bki', [
            hand_inv_crop_transforms[:, :2, :2], gt_keypoints2d
        ]) + hand_inv_crop_transforms[:, :2, 2].unsqueeze(dim=1)

        pred_left_hand_keypoints_hd = pred_keypoints2d_hd[:,
                                                          self.left_hand_idxs]
        left_hand_crop_transform = hand_crop_info['left_hand_crop_transform']
        inv_left_hand_crop_transf = torch.inverse(left_hand_crop_transform)
        left_hand_img_keypoints = torch.einsum('bij,bkj->bki', [
            inv_left_hand_crop_transf[:, :2, :2], pred_left_hand_keypoints_hd
        ]) + inv_left_hand_crop_transf[:, :2, 2].unsqueeze(dim=1)
        gt_left_hand_keypoints_hd = gt_keypoints2d_hd[:, self.left_hand_idxs]
        gt_left_hand_keypoints = torch.einsum('bij,bkj->bki', [
            inv_left_hand_crop_transf[:, :2, :2], gt_left_hand_keypoints_hd
        ]) + inv_left_hand_crop_transf[:, :2, 2].unsqueeze(dim=1)

        pred_right_hand_keypoints_hd = pred_keypoints2d_hd[:, self.
                                                           right_hand_idxs]
        right_hand_crop_transform = hand_crop_info['right_hand_crop_transform']
        inv_right_hand_crop_transf = torch.inverse(right_hand_crop_transform)
        right_hand_img_keypoints = torch.einsum('bij,bkj->bki', [
            inv_right_hand_crop_transf[:, :2, :2], pred_right_hand_keypoints_hd
        ]) + inv_right_hand_crop_transf[:, :2, 2].unsqueeze(dim=1)
        gt_right_hand_keypoints_hd = gt_keypoints2d_hd[:, self.right_hand_idxs]
        gt_right_hand_keypoints = torch.einsum('bij,bkj->bki', [
            inv_right_hand_crop_transf[:, :2, :2], gt_right_hand_keypoints_hd
        ]) + inv_right_hand_crop_transf[:, :2, 2].unsqueeze(dim=1)

        left_loss = self.hand_crop_loss(
            left_hand_img_keypoints,
            gt_left_hand_keypoints,
            weight=keypoints2d_conf[target_idxs][:, self.left_hand_idxs])
        left_loss /= gt_left_hand_keypoints.shape[0]

        right_loss = self.hand_crop_loss(
            right_hand_img_keypoints,
            gt_right_hand_keypoints,
            weight=keypoints2d_conf[target_idxs][:, self.right_hand_idxs])
        right_loss /= gt_right_hand_keypoints.shape[0]

        return left_loss + right_loss

    def compute_losses(self, predictions: dict, targets: dict):
        """Compute losses."""
        pred_param = predictions['pred_param']
        pred_cam = predictions['pred_cam']
        gt_keypoints3d = targets['keypoints3d']
        gt_keypoints2d = targets['keypoints2d']

        if self.body_model_train is not None:
            pred_output = self.body_model_train(**pred_param)
            pred_keypoints3d = pred_output['joints']
        if 'has_keypoints3d' in targets:
            has_keypoints3d = targets['has_keypoints3d'].squeeze(-1)
        else:
            has_keypoints3d = None
        if 'has_keypoints2d' in targets:
            has_keypoints2d = targets['has_keypoints2d'].squeeze(-1)
        else:
            has_keypoints2d = None

        losses = {}
        if self.loss_keypoints3d is not None:
            losses['keypoints3d_loss'] = self.compute_keypoints3d_loss(
                pred_keypoints3d,
                gt_keypoints3d,
                has_keypoints3d=has_keypoints3d)
        if self.loss_keypoints2d is not None:
            losses['keypoints2d_loss'] = self.compute_keypoints2d_loss(
                pred_keypoints3d,
                pred_cam,
                gt_keypoints2d,
                img_res=targets['img'].shape[-1],
                has_keypoints2d=has_keypoints2d)
        if self.loss_smplx_global_orient is not None:
            pred_global_orient = pred_param['global_orient']
            pred_global_orient = pose2rotmat(pred_global_orient)
            gt_global_orient = targets['smplx_global_orient']
            has_smplx_global_orient = targets[
                'has_smplx_global_orient'].squeeze(-1)
            losses['smplx_global_orient_loss'] = \
                self.compute_smplx_global_orient_loss(
                    pred_global_orient, gt_global_orient,
                    has_smplx_global_orient)
        if self.loss_smplx_body_pose is not None:
            pred_pose = pred_param['body_pose']
            pred_pose = pose2rotmat(pred_pose)
            gt_pose = targets['smplx_body_pose']
            has_smplx_body_pose = targets['has_smplx_body_pose'].squeeze(-1)
            losses['smplx_body_pose_loss'] = \
                self.compute_smplx_body_pose_loss(
                pred_pose, gt_pose, has_smplx_body_pose)
        if self.loss_smplx_jaw_pose is not None:
            pred_jaw_pose = pred_param['jaw_pose']
            pred_jaw_pose = pose2rotmat(pred_jaw_pose)
            gt_jaw_pose = targets['smplx_jaw_pose']
            face_conf = get_keypoint_idxs_by_part('head', self.convention)
            has_smplx_jaw_pose = targets['has_smplx_jaw_pose'].squeeze(-1)
            losses['smplx_jaw_pose_loss'] = self.compute_smplx_jaw_pose_loss(
                pred_jaw_pose, gt_jaw_pose, has_smplx_jaw_pose,
                gt_keypoints2d[:, face_conf, 2])
        if self.loss_smplx_hand_pose is not None:
            pred_right_hand_pose = pred_param['right_hand_pose']
            pred_right_hand_pose = pose2rotmat(pred_right_hand_pose)
            gt_right_hand_pose = targets['smplx_right_hand_pose']
            right_hand_conf = get_keypoint_idxs_by_part(
                'right_hand', self.convention)
            has_smplx_right_hand_pose = targets[
                'has_smplx_right_hand_pose'].squeeze(-1)
            losses['smplx_right_hand_pose_loss'] = \
                self.compute_smplx_hand_pose_loss(
                    pred_right_hand_pose, gt_right_hand_pose,
                    has_smplx_right_hand_pose,
                    gt_keypoints2d[:, right_hand_conf, 2])
            if 'left_hand_pose' in pred_param:
                pred_left_hand_pose = pred_param['left_hand_pose']
                pred_left_hand_pose = pose2rotmat(pred_left_hand_pose)
                gt_left_hand_pose = targets['smplx_left_hand_pose']
                left_hand_conf = get_keypoint_idxs_by_part(
                    'left_hand', self.convention)
                has_smplx_left_hand_pose = targets[
                    'has_smplx_left_hand_pose'].squeeze(-1)
                losses['smplx_left_hand_pose_loss'] = \
                    self.compute_smplx_hand_pose_loss(
                        pred_left_hand_pose, gt_left_hand_pose,
                        has_smplx_left_hand_pose,
                        gt_keypoints2d[:, left_hand_conf, 2])
        if self.loss_smplx_betas is not None:
            pred_betas = pred_param['betas']
            gt_betas = targets['smplx_betas']
            has_smplx_betas = targets['has_smplx_betas'].squeeze(-1)
            losses['smplx_betas_loss'] = self.compute_smplx_betas_loss(
                pred_betas, gt_betas, has_smplx_betas)
        if self.loss_smplx_expression is not None:
            pred_expression = pred_param['expression']
            gt_expression = targets['smplx_expression']
            face_conf = get_keypoint_idxs_by_part('head', self.convention)
            has_smplx_expression = targets['has_smplx_expression'].squeeze(-1)
            losses[
                'smplx_expression_loss'] = self.compute_smplx_expression_loss(
                    pred_expression, gt_expression, has_smplx_expression,
                    gt_keypoints2d[:, face_conf, 2])
        if self.loss_smplx_betas_piror is not None:
            pred_betas = pred_param['betas']
            losses['smplx_betas_prior_loss'] = \
                self.compute_smplx_betas_prior_loss(
                    pred_betas)
        if self.loss_camera is not None:
            losses['camera_loss'] = self.compute_camera_loss(pred_cam)
        if self.apply_hand_model and self.hand_crop_loss is not None:
            losses['hand_crop_loss'] = self.compute_hand_crop_loss(
                pred_keypoints3d, pred_cam, gt_keypoints2d,
                predictions['hand_crop_info'], targets['img'].shape[-1],
                has_keypoints2d)
        if self.apply_face_model and self.face_crop_loss is not None:
            losses['face_crop_loss'] = self.compute_face_crop_loss(
                pred_keypoints3d, pred_cam, gt_keypoints2d,
                predictions['face_crop_info'], targets['img'].shape[-1],
                has_keypoints2d)
        return losses

    @abstractmethod
    def prepare_targets(self, data_batch):
        pass

    def forward_train(self, **kwargs):
        """Forward function for general training.

        For mesh estimation, we do not use this interface.
        """
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule. Please use '
                                  '`train_step` for training.')

    @abstractmethod
    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        pass


class SMPLXImageBodyModelEstimator(SMPLXBodyModelEstimator):

    def prepare_targets(self, data_batch: dict):
        # Image Mesh Estimator does not need extra process for ground truth
        return data_batch

    def forward_test(self, img: torch.Tensor, img_metas: dict, **kwargs):
        """Defines the computation performed at every call when testing."""
        if self.backbone is not None:
            features = self.backbone(img)
        else:
            features = kwargs['features']

        if self.neck is not None:
            features = self.neck(features)

        predictions = self.head(features)
        if self.apply_hand_model:
            hand_input_img, hand_mean, hand_crop_info = self.crop_hand_func(
                predictions, img_metas)
            hand_features = self.hand_backbone(hand_input_img)
            if self.neck is not None:
                hand_features = self.hand_neck(hand_features)
            hand_predictions = self.hand_head(hand_features, cond=hand_mean)
            predictions = self.hand_merge_func(predictions, hand_predictions)
            predictions['hand_crop_info'] = hand_crop_info
        if self.apply_face_model:
            face_input_img, face_mean, face_crop_info = self.crop_face_func(
                predictions, img_metas)
            face_features = self.face_backbone(face_input_img)
            if self.neck is not None:
                face_features = self.face_neck(face_features)
            face_predictions = self.face_head(face_features, cond=face_mean)
            predictions = self.face_merge_func(predictions, face_predictions)
            predictions['face_crop_info'] = face_crop_info

        pred_param = predictions['pred_param']
        pred_cam = predictions['pred_cam']

        pred_output = self.body_model_test(**pred_param)

        pred_vertices = pred_output['vertices']
        pred_keypoints_3d = pred_output['joints']
        all_preds = {}
        all_preds['keypoints_3d'] = pred_keypoints_3d.detach().cpu().numpy()
        for value in pred_param.values():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
        all_preds['param'] = pred_param
        all_preds['camera'] = pred_cam.detach().cpu().numpy()
        all_preds['vertices'] = pred_vertices.detach().cpu().numpy()
        image_path = []
        for img_meta in img_metas:
            image_path.append(img_meta['image_path'])
        all_preds['image_path'] = image_path
        all_preds['image_idx'] = kwargs['sample_idx']
        return all_preds


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are
    fixed."""

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    @staticmethod
    def from_bn(module: nn.BatchNorm2d):
        """Initializes a frozen batch norm module from a batch norm module."""
        dim = len(module.weight.data)

        frozen_module = FrozenBatchNorm2d(dim)
        frozen_module.weight.data = module.weight.data

        missing, not_found = frozen_module.load_state_dict(
            module.state_dict(), strict=False)
        return frozen_module

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        return F.batch_norm(x, self.running_mean, self.running_var,
                            self.weight, self.bias, False)

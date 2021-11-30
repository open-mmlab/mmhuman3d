# isort: skip_file
from abc import ABCMeta

import torch

from mmhuman3d.data.datasets.pipelines.hybrik_transforms import heatmap2coord
from ..builder import (
    ARCHITECTURES,
    build_backbone,
    build_body_model,
    build_head,
    build_loss,
    build_neck,
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


@ARCHITECTURES.register_module()
class HybrIK_trainer(BaseArchitecture, metaclass=ABCMeta):
    """Hybrik_trainer Architecture.

    Args:
        backbone (dict | None, optional): Backbone config dict. Default: None.
        neck (dict | None, optional): Neck config dict. Default: None
        head (dict | None, optional): Regressor config dict. Default: None.
        body_model (dict | None, optional): SMPL config dict. Default: None.
        loss_beta (dict | None, optional): Losses config dict for
                beta (shape parameters) estimation. Default: None
        loss_theta (dict | None, optional): Losses config dict for
                theta (pose parameters) estimation. Default: None
        loss_twist (dict | None, optional): Losses config dict
                for twist angles estimation. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 body_model=None,
                 loss_beta=None,
                 loss_theta=None,
                 loss_twist=None,
                 loss_uvd=None,
                 init_cfg=None):
        super(HybrIK_trainer, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)

        self.neck = build_neck(neck)
        self.head = build_head(head)
        self.smpl = build_body_model(body_model)

        self.loss_beta = build_loss(loss_beta)
        self.loss_theta = build_loss(loss_theta)
        self.loss_twist = build_loss(loss_twist)
        self.loss_uvd = build_loss(loss_uvd)

        self.head._initialize()

    def forward_train(self, img, img_metas, **kwargs):
        """Train step function.

        In this function, train step is carried out
            with following the pipeline:
        1. extract features with the backbone
        2. feed the extracted features into the head to
            predicte beta, theta, twist angle, and heatmap (uvd map)
        3. compute regression losses of the predictions
            and optimize backbone and head
        Args:
            img (torch.Tensor): Batch of data as input.
            kwargs (dict): Dict with ground-truth
        Returns:
            output (dict): Dict with loss, information for logger,
            the number of samples.
        """
        labels = {}
        labels['trans_inv'] = kwargs['trans_inv']
        labels['intrinsic_param'] = kwargs['intrinsic_param']
        labels['joint_root'] = kwargs['joint_root']
        labels['depth_factor'] = kwargs['depth_factor']
        labels['target_uvd_29'] = kwargs['target_uvd_29']
        labels['target_xyz_24'] = kwargs['target_xyz_24']
        labels['target_weight_24'] = kwargs['target_weight_24']
        labels['target_weight_29'] = kwargs['target_weight_29']
        labels['target_xyz_17'] = kwargs['target_xyz_17']
        labels['target_weight_17'] = kwargs['target_weight_17']
        labels['target_theta'] = kwargs['target_theta']
        labels['target_beta'] = kwargs['target_beta']
        labels['target_smpl_weight'] = kwargs['target_smpl_weight']
        labels['target_theta_weight'] = kwargs['target_theta_weight']
        labels['target_twist'] = kwargs['target_twist']
        labels['target_twist_weight'] = kwargs['target_twist_weight']
        # flip_output = kwargs.pop('is_flipped', None)

        for k, _ in labels.items():
            labels[k] = labels[k].cuda()

        trans_inv = labels.pop('trans_inv')
        intrinsic_param = labels.pop('intrinsic_param')
        joint_root = labels.pop('joint_root')
        depth_factor = labels.pop('depth_factor')

        if self.backbone is not None:
            img = img.cuda().requires_grad_()
            features = self.backbone(img)
            features = features[0]
        else:
            features = img['features']

        if self.neck is not None:
            features = self.neck(features)

        predictions = self.head(features, trans_inv, intrinsic_param,
                                joint_root, depth_factor, self.smpl)

        losses = self.compute_losses(predictions, labels)

        loss, log_vars = self._parse_losses(losses)

        output = {
            'loss': loss,
        }
        return output

    def compute_losses(self, predictions, targets):
        """Compute regression losses for beta, theta, twist and uvd."""
        smpl_weight = targets['target_smpl_weight']

        losses = {}
        if self.loss_beta is not None:
            losses['loss_beta'] = self.loss_beta(
                predictions['pred_shape'] * smpl_weight,
                targets['target_beta'] * smpl_weight)
        if self.loss_theta is not None:
            losses['loss_theta'] = self.loss_theta(
                predictions['pred_theta_mats'] * smpl_weight *
                targets['target_theta_weight'], targets['target_theta'] *
                smpl_weight * targets['target_theta_weight'])
        if self.loss_twist is not None:
            losses['loss_twist'] = self.loss_twist(
                predictions['pred_phi'] * targets['target_twist_weight'],
                targets['target_twist'] * targets['target_twist_weight'])
        if self.loss_uvd is not None:
            pred_uvd = predictions['pred_uvd_jts']
            target_uvd = targets['target_uvd_29'][:, :pred_uvd.shape[1]]
            target_uvd_weight = targets['target_weight_29'][:, :pred_uvd.
                                                            shape[1]]
            losses['loss_uvd'] = self.loss_uvd(
                64 * predictions['pred_uvd_jts'],
                64 * target_uvd,
                target_uvd_weight,
                avg_factor=target_uvd_weight.sum())

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """Test step function.

        In this function, train step is carried out
            with following the pipeline:
        1. extract features with the backbone
        2. feed the extracted features into the head to
            predicte beta, theta, twist angle, and heatmap (uvd map)
        3. store predictions for evaluation
        Args:
            img (torch.Tensor): Batch of data as input.
            img_metas (dict): Dict with image metas i.e. path
            kwargs (dict): Dict with ground-truth
        Returns:
            all_preds (dict): Dict with image_path, vertices, xyz_17, uvd_jts,
            xyz_24 for predictions.
        """
        labels = {}
        labels['trans_inv'] = kwargs['trans_inv']
        labels['intrinsic_param'] = kwargs['intrinsic_param']
        labels['joint_root'] = kwargs['joint_root']
        labels['depth_factor'] = kwargs['depth_factor']
        labels['target_uvd_29'] = kwargs['target_uvd_29']
        labels['target_xyz_24'] = kwargs['target_xyz_24']
        labels['target_weight_24'] = kwargs['target_weight_24']
        labels['target_weight_29'] = kwargs['target_weight_29']
        labels['target_xyz_17'] = kwargs['target_xyz_17']
        labels['target_weight_17'] = kwargs['target_weight_17']
        labels['target_theta'] = kwargs['target_theta']
        labels['target_beta'] = kwargs['target_beta']
        labels['target_smpl_weight'] = kwargs['target_smpl_weight']
        labels['target_theta_weight'] = kwargs['target_theta_weight']
        labels['target_twist'] = kwargs['target_twist']
        labels['target_twist_weight'] = kwargs['target_twist_weight']

        bboxes = kwargs['bbox']

        for k, _ in labels.items():
            labels[k] = labels[k].cuda()

        trans_inv = labels.pop('trans_inv')
        intrinsic_param = labels.pop('intrinsic_param')
        joint_root = labels.pop('joint_root')
        depth_factor = labels.pop('depth_factor')
        if len(depth_factor.shape) != 2:
            depth_factor = torch.unsqueeze(depth_factor, dim=1)

        if self.backbone is not None:
            img = img.cuda().requires_grad_()
            features = self.backbone(img)
            features = features[0]
        else:
            features = img['features']

        if self.neck is not None:
            features = self.neck(features)

        output = self.head(features, trans_inv, intrinsic_param, joint_root,
                           depth_factor, self.smpl)

        pred_uvd_jts = output['pred_uvd_jts']
        batch_num = pred_uvd_jts.shape[0]
        pred_xyz_jts_24 = output['pred_xyz_jts_24'].reshape(batch_num, -1,
                                                            3)[:, :24, :]
        pred_xyz_jts_24_struct = output['pred_xyz_jts_24_struct'].reshape(
            batch_num, 24, 3)
        pred_xyz_jts_17 = output['pred_xyz_jts_17'].reshape(batch_num, 17, 3)
        pred_mesh = output['pred_vertices'].reshape(batch_num, -1, 3)

        pred_xyz_jts_24 = pred_xyz_jts_24.cpu().data.numpy()
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.cpu().data.numpy()
        pred_xyz_jts_17 = pred_xyz_jts_17.cpu().data.numpy()
        pred_uvd_jts = pred_uvd_jts.cpu().data
        pred_mesh = pred_mesh.cpu().data.numpy()

        assert pred_xyz_jts_17.ndim in [2, 3]
        pred_xyz_jts_17 = pred_xyz_jts_17.reshape(pred_xyz_jts_17.shape[0], 17,
                                                  3)
        pred_uvd_jts = pred_uvd_jts.reshape(pred_uvd_jts.shape[0], -1, 3)
        pred_xyz_jts_24 = pred_xyz_jts_24.reshape(pred_xyz_jts_24.shape[0], 24,
                                                  3)
        pred_scores = output['maxvals'].cpu().data[:, :29]

        hm_shape = [64, 64]
        pose_coords_list = []
        for i in range(pred_xyz_jts_17.shape[0]):
            bbox = bboxes[i].tolist()
            pose_coords, _ = heatmap2coord(
                pred_uvd_jts[i],
                pred_scores[i],
                hm_shape,
                bbox,
                mean_bbox_scale=None)
            pose_coords_list.append(pose_coords)

        all_preds = {}
        all_preds['vertices'] = pred_mesh
        all_preds['xyz_17'] = pred_xyz_jts_17
        all_preds['uvd_jts'] = pose_coords
        all_preds['xyz_24'] = pred_xyz_jts_24_struct
        image_path = []
        for img_meta in img_metas:
            image_path.append(img_meta['image_path'])
        all_preds['image_path'] = image_path
        all_preds['image_idx'] = kwargs['sample_idx']
        return all_preds

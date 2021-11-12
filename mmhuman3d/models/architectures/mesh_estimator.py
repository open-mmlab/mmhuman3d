from abc import ABCMeta, abstractmethod

import numpy as np
import torch

from mmhuman3d.core.parametric_model.builder import build_registrant
from mmhuman3d.models.utils.fits_dict import FitsDict
from mmhuman3d.utils.geometry import (
    batch_rodrigues,
    estimate_translation,
    project_points,
    rotation_matrix_to_angle_axis,
)
from ..builder import (
    ARCHITECTURES,
    build_backbone,
    build_body_model,
    build_discriminator,
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


class BodyModelEstimator(BaseArchitecture, metaclass=ABCMeta):
    """BodyModelEstimator Architecture.

    Args:
        backbone (dict | None, optional): Backbone config dict. Default: None.
        neck (dict | None, optional): Neck config dict. Default: None
        head (dict | None, optional): Regressor config dict. Default: None.
        disc (dict | None, optional): Discriminator config dict.
            Default: None.
        registrant ( dict | None, optional): Registrant config dict.
            Default: None.
        smpl (dict | None, optional): SMPL config dict. Default: None.
        loss_mesh (dict | None, optional): Losses config dict for supervised
            learning. Default: None
        loss_gan (dict | None, optional): Losses config for adversial
            training. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 disc=None,
                 registrant=None,
                 body_model=None,
                 loss_keypoints2d=None,
                 loss_keypoints3d=None,
                 loss_vertex=None,
                 loss_smpl_pose=None,
                 loss_smpl_betas=None,
                 loss_adv=None,
                 init_cfg=None):
        super(BodyModelEstimator, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)
        self.disc = build_discriminator(disc)

        self.smpl = build_body_model(body_model)

        # TODO: temp solution
        if registrant is None:
            self.registrant = None
        elif registrant == 'static_fits':
            # train with static fits, wo registration
            self.fits = 'static_fits'
            self.fits_dict = FitsDict(fits='static')
            self.registrant = build_registrant(registrant)
        elif registrant == 'final_fits':
            # train with final fits, wo registration
            self.fits = 'final_fits'
            self.fits_dict = FitsDict(fits='final')
            self.registrant = build_registrant(registrant)
        else:
            # train with static fits as initial fits, with registration
            self.fits = 'registration'
            self.fits_dict = FitsDict(fits='static')
            self.registrant = build_registrant(registrant)

        self.loss_keypoints2d = build_loss(loss_keypoints2d)
        self.loss_keypoints3d = build_loss(loss_keypoints3d)
        self.loss_vertex = build_loss(loss_vertex)
        self.loss_smpl_pose = build_loss(loss_smpl_pose)
        self.loss_smpl_betas = build_loss(loss_smpl_betas)
        self.loss_adv = build_loss(loss_adv)

        if self.smpl is not None:
            for m in self.smpl.parameters():
                m.requires_grad = False

    def train_step(self, data_batch, optimizer, **kwargs):
        """Train step function.

        In this function, the detector will finish the train step following
        the pipeline:
        1. get fake and real SMPL parameters
        2. optimize discriminator (if have)
        3. optimize generator
        If `self.train_cfg.disc_step > 1`, the train step will contain multiple
        iterations for optimizing discriminator with different input data and
        only one iteration for optimizing generator after `disc_step`
        iterations for discriminator.
        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (dict[torch.optim.Optimizer]): Dict with optimizers for
                generator and discriminator (if have).
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
        targets = self.prepare_targets(data_batch)

        # optimize discriminator (if have)
        if self.disc is not None:
            self.optimize_discrinimator(predictions, data_batch, optimizer)

        if self.registrant is not None:
            targets = self.run_registration(predictions, targets)

        losses = self.compute_losses(predictions, targets)

        # optimizer generator part
        if self.disc is not None:
            adv_loss = self.optimize_generator(predictions)
            losses.update(adv_loss)

        loss, log_vars = self._parse_losses(losses)

        if self.backbone is not None:
            optimizer['backbone'].zero_grad()
        if self.neck is not None:
            optimizer['neck'].zero_grad()
        if self.head is not None:
            optimizer['head'].zero_grad()
        loss.backward()
        if self.backbone is not None:
            optimizer['backbone'].step()
        if self.neck is not None:
            optimizer['neck'].step()
        if self.head is not None:
            optimizer['head'].step()

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def run_registration(self, predictions, targets):
        """Generates pseudo labels."""

        img_metas = targets['img_metas']
        dataset_name = [meta['dataset_name'] for meta in img_metas
                        ]  # name of the dataset the image comes from

        indices = targets['sample_idx'].squeeze()
        is_flipped = targets['is_flipped'].squeeze().bool(
        )  # flag that indicates whether image was flipped
        # during data augmentation
        rot_angle = targets['rotation'].squeeze(
        )  # rotation angle used for data augmentation Q
        gt_betas = targets['smpl_betas'].float()
        gt_global_orient = targets['smpl_global_orient'].float()
        gt_pose = targets['smpl_body_pose'].float().view(-1, 69)

        pred_rotmat = predictions['pred_pose'].detach().clone()
        pred_betas = predictions['pred_shape'].detach().clone()
        pred_cam_t = predictions['pred_cam'].detach().clone()

        keypoints2d = targets['keypoints2d'].float()
        keypoints2d_mask = targets['keypoints2d_mask'].float()

        try:
            gt_keypoints_2d = torch.cat(
                [keypoints2d, keypoints2d_mask.reshape(-1, 49, 1)], dim=-1)
        except Exception:
            gt_keypoints_2d = torch.cat(
                [keypoints2d, keypoints2d_mask.reshape(-1, 24, 1)], dim=-1)
        num_keypoints = gt_keypoints_2d.shape[1]

        has_smpl = targets['has_smpl'].squeeze().bool(
        )  # flag that indicates whether SMPL parameters are valid
        batch_size = has_smpl.shape[0]
        device = has_smpl.device

        # TODO: temp solution, hard code
        img_res = 224
        # smplify_threshold = 100.0
        smplify_threshold = 10.0  # our implementation uses 10.0
        focal_length = 5000.0

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as
        # it comes from SMPL
        # gt_out = self.smpl(betas=gt_betas,
        # body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
        gt_out = self.smpl(
            betas=gt_betas, body_pose=gt_pose, global_orient=gt_global_orient)
        if num_keypoints == 49:
            gt_model_joints = gt_out['joints']
            gt_vertices = gt_out['vertices']
        else:
            gt_model_joints = gt_out['joints'][:, 25:, :]
            gt_vertices = gt_out['vertices']
        # TODO: add joint mask

        # Get current best fits from the dictionary
        opt_pose, opt_betas = self.fits_dict[(dataset_name, indices.cpu(),
                                              rot_angle.cpu(),
                                              is_flipped.cpu())]

        opt_pose = opt_pose.to(device)
        opt_betas = opt_betas.to(device)
        opt_output = self.smpl(
            betas=opt_betas,
            body_pose=opt_pose[:, 3:],
            global_orient=opt_pose[:, :3])
        if num_keypoints == 49:
            opt_joints = opt_output['joints']
            opt_vertices = opt_output['vertices']
        else:
            opt_joints = opt_output['joints'][:, 25:, :]
            opt_vertices = opt_output['vertices']

        # # De-normalize 2D keypoints from [-1,1] to pixel space
        # gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        # gt_keypoints_2d_orig[:, :, :-1] = 0.5 * img_res *
        # (gt_keypoints_2d_orig[:, :, :-1] + 1)
        # TODO: current pipeline, the keypoints are already in the pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(
            gt_model_joints,
            gt_keypoints_2d_orig,
            focal_length=focal_length,
            img_size=img_res)

        opt_cam_t = estimate_translation(
            opt_joints,
            gt_keypoints_2d_orig,
            focal_length=focal_length,
            img_size=img_res)

        # opt_joint_loss = self.registrant.compute_loss(
        #     opt_pose, opt_betas, opt_cam_t,
        #     0.5 * img_res * torch.ones(batch_size, 2, device=device),
        #     gt_keypoints_2d_orig).mean(dim=-1)
        with torch.no_grad():
            loss_dict = self.registrant.evaluate(
                global_orient=opt_pose[:, :3],
                body_pose=opt_pose[:, 3:],
                betas=opt_betas,
                transl=opt_cam_t,
                keypoints2d=gt_keypoints_2d_orig[:, :, :2],
                keypoints2d_conf=gt_keypoints_2d_orig[:, :, 2],
                reduction_override='none')
        opt_joint_loss = loss_dict['keypoint2d_loss'].sum(dim=-1).sum(dim=-1)

        # Convert predicted rotation matrices to axis-angle
        pred_rotmat_hom = torch.cat([
            pred_rotmat.detach().view(-1, 3, 3).detach(),
            torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(
                1, 3, 1).expand(batch_size * 24, -1, -1)
        ],
                                    dim=-1)
        pred_pose = rotation_matrix_to_angle_axis(
            pred_rotmat_hom).contiguous().view(batch_size, -1)
        # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation,
        # so manually hack it
        pred_pose[torch.isnan(pred_pose)] = 0.0

        # TODO: temp solution
        if self.fits in ['static_fits', 'final_fits']:
            pass
        elif self.fits == 'registration':

            # Run SMPLify optimization starting from the network prediction
            # new_opt_vertices, new_opt_joints, \
            #     new_opt_pose, new_opt_betas, \
            #     new_opt_cam_t, new_opt_joint_loss = \
            registrant_output = self.registrant(
                keypoints2d=gt_keypoints_2d_orig[:, :, :2],
                keypoints2d_conf=gt_keypoints_2d_orig[:, :, 2],
                init_global_orient=pred_pose[:, :3],
                init_transl=pred_cam_t,
                init_body_pose=pred_pose[:, 3:],
                init_betas=pred_betas,
                return_joints=True,
                return_verts=True,
                return_losses=True)

            new_opt_vertices = registrant_output['vertices']
            new_opt_joints = registrant_output['joints']

            new_opt_global_orient = registrant_output['global_orient']
            new_opt_body_pose = registrant_output['body_pose']
            new_opt_pose = torch.cat(
                [new_opt_global_orient, new_opt_body_pose], dim=1)

            new_opt_betas = registrant_output['betas']
            new_opt_cam_t = registrant_output['transl']
            new_opt_joint_loss = registrant_output['keypoint2d_loss'].sum(
                dim=-1).sum(dim=-1)

            # new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)

            # Will update the dictionary for the examples where the new loss
            # is less than the current one
            update = (new_opt_joint_loss < opt_joint_loss)

            opt_joint_loss[update] = new_opt_joint_loss[update]
            opt_vertices[update, :] = new_opt_vertices[update, :]
            opt_joints[update, :] = new_opt_joints[update, :]
            opt_pose[update, :] = new_opt_pose[update, :]
            opt_betas[update, :] = new_opt_betas[update, :]
            opt_cam_t[update, :] = new_opt_cam_t[update, :]

            self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(),
                            is_flipped.cpu(),
                            update.cpu())] = (opt_pose.cpu(), opt_betas.cpu())

        # else:
        #     update = torch.zeros(batch_size, device=self.device).bool()

        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.

        # Replace the optimized parameters with the ground truth parameters,
        # if available
        opt_vertices[has_smpl, :, :] = gt_vertices[has_smpl, :, :]
        opt_cam_t[has_smpl, :] = gt_cam_t[has_smpl, :]
        opt_joints[has_smpl, :, :] = gt_model_joints[has_smpl, :, :]
        opt_pose[has_smpl, 3:] = gt_pose[has_smpl, :]
        opt_pose[has_smpl, :3] = gt_global_orient[has_smpl, :]
        opt_betas[has_smpl, :] = gt_betas[has_smpl, :]

        # Assert whether a fit is valid by comparing the joint loss with
        # the threshold
        valid_fit = (opt_joint_loss < smplify_threshold).to(device)
        # Add the examples with GT parameters to the list of valid fits
        # valid_fit = valid_fit | has_smpl

        valid_fit = valid_fit | has_smpl
        targets['valid_fit'] = valid_fit

        targets['opt_vertices'] = opt_vertices
        targets['opt_cam_t'] = opt_cam_t
        targets['opt_joints'] = opt_joints
        targets['opt_pose'] = opt_pose
        targets['opt_betas'] = opt_betas

        return targets

    def optimize_discrinimator(self, predictions, data_batch, optimizer):
        set_requires_grad(self.disc, True)
        fake_data = self.make_fake_data(predictions, requires_grad=False)
        real_data = self.make_real_data(data_batch)
        fake_score = self.disc(fake_data)
        real_score = self.disc(real_data)

        disc_losses = {}
        disc_losses['real_loss'] = self.loss_adv(
            real_score, target_is_real=True, is_disc=True)
        disc_losses['fake_loss'] = self.loss_adv(
            fake_score, target_is_real=False, is_disc=True)
        loss_disc, log_vars_d = self._parse_losses(disc_losses)

        optimizer['disc'].zero_grad()
        loss_disc.backward()
        optimizer['disc'].step()

    def optimize_generator(self, predictions):
        set_requires_grad(self.disc, False)
        fake_data = self.make_fake_data(predictions, requires_grad=True)
        pred_score = self.disc(fake_data)
        loss_adv = self.loss_adv(
            pred_score, target_is_real=True, is_disc=False)
        loss = dict(adv_loss=loss_adv)
        return loss

    def compute_keypoints3d_loss(self, pred_keypoints3d, gt_keypoints3d,
                                 keypoints3d_mask):
        pred_keypoints3d = pred_keypoints3d.float()
        gt_keypoints3d = gt_keypoints3d.float()
        keypoints3d_mask = keypoints3d_mask.float().unsqueeze(-1)

        # TODO
        num_keypoints = gt_keypoints3d.shape[1]
        assert num_keypoints in [24, 49]
        if num_keypoints == 24:
            gt_pelvis = (gt_keypoints3d[:, 2, :] + gt_keypoints3d[:, 3, :]) / 2
            pred_pelvis = (pred_keypoints3d[:, 2, :] +
                           pred_keypoints3d[:, 3, :]) / 2
        else:
            gt_pelvis = (gt_keypoints3d[:, 2 + 25, :] +
                         gt_keypoints3d[:, 3 + 25, :]) / 2
            pred_pelvis = (pred_keypoints3d[:, 2 + 25, :] +
                           pred_keypoints3d[:, 3 + 25, :]) / 2

        gt_keypoints3d = gt_keypoints3d - gt_pelvis[:, None, :]
        pred_keypoints3d = pred_keypoints3d - pred_pelvis[:, None, :]
        loss = self.loss_keypoints3d(
            pred_keypoints3d, gt_keypoints3d, weight=keypoints3d_mask)
        return loss

    def compute_keypoints2d_loss(self,
                                 pred_keypoints3d,
                                 pred_cam,
                                 gt_keypoints2d,
                                 keypoints2d_mask,
                                 img_res=224,
                                 focal_length=5000):
        gt_keypoints2d = gt_keypoints2d.float()
        keypoints2d_mask = keypoints2d_mask.float().unsqueeze(-1)
        pred_keypoints2d = project_points(
            pred_keypoints3d,
            pred_cam,
            focal_length=focal_length,
            img_res=img_res)
        # Normalize keypoints to [-1,1]
        # The coordinate origin of pred_keypoints_2d is
        # the center of the input image.
        pred_keypoints2d = 2 * pred_keypoints2d / (img_res - 1)
        # The coordinate origin of gt_keypoints_2d is
        # the top left corner of the input image.
        gt_keypoints2d = 2 * gt_keypoints2d / (img_res - 1) - 1
        loss = self.loss_keypoints2d(
            pred_keypoints2d, gt_keypoints2d, weight=keypoints2d_mask)
        return loss

    def compute_vertex_loss(self, pred_vertices, gt_vertices, has_smpl):
        gt_vertices = gt_vertices.float()
        conf = has_smpl.float().view(-1, 1, 1)
        loss = self.loss_vertex(pred_vertices, gt_vertices, weight=conf)
        return loss

    def compute_smpl_pose_loss(self, pred_rotmat, gt_pose, has_smpl):
        conf = has_smpl.float().view(-1, 1, 1, 1)
        gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
        loss = self.loss_smpl_pose(pred_rotmat, gt_rotmat, weight=conf)
        return loss

    def compute_smpl_betas_loss(self, pred_betas, gt_betas, has_smpl):
        conf = has_smpl.float().view(-1, 1)
        loss = self.loss_smpl_betas(pred_betas, gt_betas, weight=conf)
        return loss

    def compute_losses(self, predictions, targets):
        pred_betas = predictions['pred_shape'].view(-1, 10)
        pred_pose = predictions['pred_pose'].view(-1, 24, 3, 3)
        pred_cam = predictions['pred_cam'].view(-1, 3)

        keypoints3d_mask = targets['keypoints3d_mask']
        keypoints2d_mask = targets['keypoints2d_mask']
        gt_keypoints3d = targets['keypoints3d']
        gt_keypoints2d = targets['keypoints2d']

        # pred_pose N, 24, 3, 3
        pred_output = self.smpl(
            betas=pred_betas,
            body_pose=pred_pose[:, 1:],
            global_orient=pred_pose[:, 0].unsqueeze(1),
            pose2rot=False,
            num_joints=gt_keypoints2d.shape[1])
        pred_keypoints3d = pred_output['joints']
        pred_vertices = pred_output['vertices']

        # # TODO: temp. Should we multiply confs here?
        # pred_keypoints3d_mask = pred_output['joint_mask']
        # keypoints3d_mask = keypoints3d_mask * pred_keypoints3d_mask

        # TODO: temp solution
        if 'valid_fit' in targets:
            has_smpl = targets['valid_fit'].view(-1)
            # global_orient = targets['opt_pose'][:, :3].view(-1, 1, 3)
            gt_pose = targets['opt_pose']
            gt_betas = targets['opt_betas']
            gt_vertices = targets['opt_vertices']
        else:
            has_smpl = targets['has_smpl'].view(-1)
            gt_pose = targets['smpl_body_pose']
            global_orient = targets['smpl_global_orient'].view(-1, 1, 3)
            gt_pose = torch.cat((global_orient, gt_pose), dim=1).float()
            gt_betas = targets['smpl_betas'].float()

            # gt_pose N, 72
            gt_output = self.smpl(
                betas=gt_betas,
                body_pose=gt_pose[:, 3:],
                global_orient=gt_pose[:, :3],
                num_joints=gt_keypoints2d.shape[1])
            gt_vertices = gt_output['vertices']

        # TODO: temp
        num_joints = gt_keypoints3d.shape[1]
        num_joints_pred = pred_keypoints3d.shape[1]
        assert num_joints in [24, 49]
        if num_joints == 24:
            if num_joints_pred == 49:
                pred_keypoints3d = pred_keypoints3d[:, 25:, :]
            else:
                assert num_joints_pred == 24
        else:
            # openpose keypoints are not used for loss computation
            keypoints3d_mask[:, :25] = 0
            keypoints2d_mask[:, :25] = 0

        losses = {}
        if self.loss_keypoints3d is not None:
            losses['keypoints3d_loss'] = self.compute_keypoints3d_loss(
                pred_keypoints3d, gt_keypoints3d, keypoints3d_mask)
        if self.loss_keypoints2d is not None:
            losses['keypoints2d_loss'] = self.compute_keypoints2d_loss(
                pred_keypoints3d, pred_cam, gt_keypoints2d, keypoints2d_mask)
        if self.loss_vertex is not None:
            losses['vertex_loss'] = self.compute_vertex_loss(
                pred_vertices, gt_vertices, has_smpl)
        if self.loss_smpl_pose is not None:
            losses['smpl_pose_loss'] = self.compute_smpl_pose_loss(
                pred_pose, gt_pose, has_smpl)
        if self.loss_smpl_betas is not None:
            losses['smpl_betas_loss'] = self.compute_smpl_betas_loss(
                pred_betas, gt_betas, has_smpl)

        return losses

    @abstractmethod
    def make_fake_data(self, predictions, requires_grad):
        pass

    @abstractmethod
    def make_real_data(self, data_batch):
        pass

    @abstractmethod
    def prepare_targets(self, data_batch):
        pass

    def forward_train(self, **kwargs):
        """Forward function for training.

        For HMR, we do not use this interface.
        """
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule. Please use '
                                  '`train_step` for training.')

    @abstractmethod
    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        pass


@ARCHITECTURES.register_module()
class ImageBodyModelEstimator(BodyModelEstimator):

    def make_fake_data(self, predictions, requires_grad):
        pred_cam = predictions['pred_cam']
        pred_pose = predictions['pred_pose']
        pred_betas = predictions['pred_shape']
        if requires_grad:
            fake_data = (pred_cam, pred_pose, pred_betas)
        else:
            fake_data = (pred_cam.detach(), pred_pose.detach(),
                         pred_betas.detach())
        return fake_data

    def make_real_data(self, data_batch):
        transl = data_batch['adv_smpl_transl'].float()
        global_orient = data_batch['adv_smpl_global_orient']
        body_pose = data_batch['adv_smpl_body_pose']
        betas = data_batch['adv_smpl_betas'].float()
        pose = torch.cat((global_orient, body_pose), dim=-1).float()
        real_data = (transl, pose, betas)
        return real_data

    def prepare_targets(self, data_batch):
        # Image Mesh Estimator does not need extra process for ground truth
        return data_batch

    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        if self.backbone is not None:
            features = self.backbone(img)
        else:
            features = kwargs['features']

        if self.neck is not None:
            features = self.neck(features)

        predictions = self.head(features)
        pred_pose = predictions['pred_pose']
        pred_betas = predictions['pred_shape']
        pred_cam = predictions['pred_cam']

        pred_output = self.smpl(
            betas=pred_betas,
            body_pose=pred_pose[:, 1:],
            global_orient=pred_pose[:, 0].unsqueeze(1),
            pose2rot=False)

        pred_vertices = pred_output['vertices']
        pred_keypoints_3d = pred_output['joints']
        all_preds = {}
        all_preds['keypoints_3d'] = pred_keypoints_3d.detach().cpu().numpy()
        all_preds['smpl_pose'] = pred_pose.detach().cpu().numpy()
        all_preds['smpl_beta'] = pred_betas.detach().cpu().numpy()
        all_preds['camera'] = pred_cam.detach().cpu().numpy()
        all_preds['vertices'] = pred_vertices.detach().cpu().numpy()
        image_path = []
        for img_meta in img_metas:
            image_path.append(img_meta['image_path'])
        all_preds['image_path'] = image_path
        return all_preds


@ARCHITECTURES.register_module()
class VideoBodyModelEstimator(BodyModelEstimator):

    def make_fake_data(self, predictions, requires_grad):
        B, T = predictions['pred_cam'].shape[:2]
        pred_cam_vec = predictions['pred_cam']
        pred_betas_vec = predictions['pred_shape']
        pred_pose = predictions['pred_pose']
        pred_pose_vec = rotation_matrix_to_angle_axis(pred_pose.view(-1, 3, 3))
        pred_pose_vec = pred_pose_vec.contiguous().view(B, T, -1)
        pred_theta_vec = (pred_cam_vec, pred_pose_vec, pred_betas_vec)
        pred_theta_vec = torch.cat(pred_theta_vec, dim=-1)

        if not requires_grad:
            pred_theta_vec = pred_theta_vec.detach()
        return pred_theta_vec[:, :, 6:75]

    def make_real_data(self, data_batch):
        B, T = data_batch['adv_smpl_transl'].shape[:2]
        transl = data_batch['adv_smpl_transl'].view(B, T, -1)
        global_orient = \
            data_batch['adv_smpl_global_orient'].view(B, T, -1)
        body_pose = data_batch['adv_smpl_body_pose'].view(B, T, -1)
        betas = data_batch['adv_smpl_betas'].view(B, T, -1)
        real_data = (transl, global_orient, body_pose, betas)
        real_data = torch.cat(real_data, dim=-1).float()
        return real_data[:, :, 6:75]

    def prepare_targets(self, data_batch):
        # Video Mesh Estimator needs squeeze first two dimensions
        B, T = data_batch['smpl_body_pose'].shape[:2]

        output = {
            'smpl_body_pose': data_batch['smpl_body_pose'].view(-1, 23, 3),
            'smpl_global_orient': data_batch['smpl_global_orient'].view(-1, 3),
            'smpl_betas': data_batch['smpl_betas'].view(-1, 10),
            'has_smpl': data_batch['has_smpl'].view(-1),
            'keypoints3d_mask': data_batch['keypoints3d_mask'].view(B * T, -1),
            'keypoints2d_mask': data_batch['keypoints2d_mask'].view(B * T, -1),
            'keypoints3d': data_batch['keypoints3d'].view(B * T, -1, 3),
            'keypoints2d': data_batch['keypoints2d'].view(B * T, -1, 2)
        }
        return output

    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        if self.backbone is not None:
            features = self.backbone(img)
        else:
            features = kwargs['features']

        if self.neck is not None:
            features = self.neck(features)

        B, T = features.shape[:2]
        predictions = self.head(features)
        pred_pose = predictions['pred_pose'].view(-1, 24, 3, 3)
        pred_betas = predictions['pred_shape'].view(-1, 10)
        pred_cam = predictions['pred_cam'].view(-1, 3)

        pred_output = self.smpl(
            betas=pred_betas,
            body_pose=pred_pose[:, 1:],
            global_orient=pred_pose[:, 0].unsqueeze(1),
            pose2rot=False)

        pred_vertices = pred_output['vertices']
        pred_keypoints_3d = pred_output['joints']
        all_preds = {}
        all_preds['keypoints_3d'] = pred_keypoints_3d.detach().cpu().numpy()
        all_preds['smpl_pose'] = pred_pose.detach().cpu().numpy()
        all_preds['smpl_beta'] = pred_betas.detach().cpu().numpy()
        all_preds['camera'] = pred_cam.detach().cpu().numpy()
        all_preds['vertices'] = pred_vertices.detach().cpu().numpy()
        new_all_preds = {}
        new_all_preds['image_path'] = []
        for k, v in all_preds.items():
            new_v = []
            for i in range(B):
                frame_idx = img_metas[i]['frame_idx']
                valid_start_idx_bias = frame_idx[2] - frame_idx[0]
                valid_end_idx_bias = frame_idx[3] - frame_idx[0]
                start_idx = i * T + valid_start_idx_bias
                end_idx = i * T + valid_end_idx_bias
                new_v.append(all_preds[k][start_idx:end_idx + 1])
            new_v = np.concatenate(new_v, axis=0)
            new_all_preds[k] = new_v
        for i in range(B):
            item = img_metas[i]['image_path']
            new_all_preds['image_path'].extend(item)
        return new_all_preds

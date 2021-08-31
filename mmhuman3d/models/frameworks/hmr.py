import numpy as np
import torch
import torch.nn as nn
from mmcv.runner.base_module import BaseModule

from mmhuman3d.utils.geometry import batch_rodrigues, rot6d_to_rotmat
from ..builder import FRAMEWORKS, build_backbone, build_loss
from .base_image_estimator import BaseImageEstimator


class SimpleFC(BaseModule):

    def __init__(self, feat_dim, smpl_mean_params, npose=144, hdim=1024):
        super(SimpleFC, self).__init__()
        self.fc1 = nn.Linear(feat_dim + npose + 13, hdim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hdim, hdim)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(hdim, npose)
        self.decshape = nn.Linear(hdim, 10)
        self.deccam = nn.Linear(hdim, 3)

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(
            mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self,
                x,
                init_pose=None,
                init_shape=None,
                init_cam=None,
                n_iter=3):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam


@FRAMEWORKS.register_module()
class HMR(BaseImageEstimator):

    def __init__(self,
                 backbone,
                 head,
                 keypoint2d_criterion=None,
                 keypoint3d_criterion=None,
                 shape_criterion=None,
                 smpl_pose_criterion=None,
                 smpl_betas_criterion=None,
                 init_cfg=None):
        super(HMR, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.head = SimpleFC(feat_dim=self.backbone.feat_dim, **head)
        # TODO: support reweighing different categories of points
        # self.keypoint2d_criterion = build_loss(keypoint2d_criterion)
        # self.keypoint3d_criterion = build_loss(keypoint3d_criterion)
        # self.shape_criterion = build_loss(shape_criterion)
        self.smpl_pose_criterion = build_loss(smpl_pose_criterion)
        self.smpl_betas_criterion = build_loss(smpl_betas_criterion)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas,
                    has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1, 3))
        gt_rotmat_valid = gt_rotmat_valid.view(-1, 24, 3, 3)
        gt_rotmat_valid = gt_rotmat_valid[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.smpl_pose_criterion(
                pred_rotmat_valid.float(), gt_rotmat_valid.float())
            loss_regr_betas = self.smpl_betas_criterion(
                pred_betas_valid.float(), gt_betas_valid.float())
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def forward_train(self, img, img_metas, **kwargs):
        x = self.backbone(img)[-1]
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pred_rotmat, pred_shape, pred_cam = self.head(x)
        gt_pose = kwargs['smpl_body_pose']
        global_orient = kwargs['smpl_global_orient'].view(-1, 1, 3)
        gt_pose = torch.cat((gt_pose, global_orient), dim=1)
        gt_betas = kwargs['smpl_betas']
        has_smpl = kwargs['has_smpl'].view(-1)
        loss_regr_pose, loss_regr_betas = self.smpl_losses(
            pred_rotmat, pred_shape, gt_pose, gt_betas, has_smpl)
        output = {
            'loss_regr_pose': loss_regr_pose,
            'loss_regr_betas': loss_regr_betas
        }
        return output

    def forward_test(self, img, img_metas, **kwargs):
        x = self.backbone(img)[-1]
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pred_rotmat, pred_shape, pred_cam = self.head(x)
        output = {
            'pred_rotmat': pred_rotmat,
            'pred_shape': pred_shape,
            'pred_cam': pred_cam
        }
        return output

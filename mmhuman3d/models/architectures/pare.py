from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idx
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
    build_registrant,
)
from .base_architecture import BaseArchitecture
from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d
from mmhuman3d.utils.geometry import perspective_projection
from ..utils.pare_utils.pare_renderer import render_smpl
from mmhuman3d.core.conventions import convert_kps
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

def get_default_camera(focal_length, img_size):
    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[2, 2] = 1
    K[0, 2] = img_size / 2.
    K[1, 2] = img_size / 2.
    K = K[None, :, :]
    R = torch.eye(3)[None, :, :]
    return K, R

@ARCHITECTURES.register_module()
class PARE(BaseArchitecture, metaclass=ABCMeta):
    """PARE Architecture.

    Args:
        backbone (dict | None, optional): Backbone config dict. Default: None.
        neck (dict | None, optional): Neck config dict. Default: None
        head (dict | None, optional): Regressor config dict. Default: None.
        body_model_train (dict | None, optional): SMPL config dict during
            training. Default: None.
        body_model_test (dict | None, optional): SMPL config dict during
            test. Default: None.
        convention (str, optional): Keypoints convention for estimating 
            camera. Default: "human_data"
        loss_convention (str, optional): Keypoints convention for computing 
            key point losses. Default: "human_data"
        loss_keypoints2d (dict | None, optional): Losses config dict for
            2D keypoints. Default: None.
        loss_keypoints3d (dict | None, optional): Losses config dict for
            3D keypoints. Default: None.
        loss_smpl_pose (dict | None, optional): Losses config dict for smpl
            pose. Default: None
        loss_smpl_betas (dict | None, optional): Losses config dict for smpl
            betas. Default: None
        loss_camera (dict | None, optional): Losses config dict for predicted
            camera parameters. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        image_res (int, optional): image resolution for part silhouette rendering.
            Default: 224.
        focal_length (int, optional): focal length for part silhouette rendering.
            Default: 5000.
    """

    def __init__(self,
                 backbone: Optional[Union[dict, None]] = None,
                 neck: Optional[Union[dict, None]] = None,
                 head: Optional[Union[dict, None]] = None,
                 body_model_train: Optional[Union[dict, None]] = None,
                 body_model_test: Optional[Union[dict, None]] = None,
                 convention: Optional[str] = 'human_data',
                 loss_convention: Optional[str] = 'human_data',
                 loss_keypoints2d: Optional[Union[dict, None]] = None,
                 loss_keypoints3d: Optional[Union[dict, None]] = None,
                 loss_vertex: Optional[Union[dict, None]] = None,
                 loss_smpl_pose: Optional[Union[dict, None]] = None,
                 loss_smpl_betas: Optional[Union[dict, None]] = None,
                 loss_segm_mask: Optional[Union[dict, None]] = None, 
                 loss_camera: Optional[Union[dict, None]] = None,
                 init_cfg: Optional[Union[list, dict, None]] = None,
                 image_res: Optional[int] = 224,
                 focal_length: Optional[int] = 5000,):
        super(PARE, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)
        self.body_model_train = build_body_model(body_model_train)
        self.body_model_test = build_body_model(body_model_test)
        self.image_res = image_res
        self.focal_length = focal_length
        self.loss_keypoints2d = build_loss(loss_keypoints2d)

        

        self.loss_keypoints3d = build_loss(loss_keypoints3d)
        self.loss_smpl_pose = build_loss(loss_smpl_pose)
        self.loss_smpl_betas = build_loss(loss_smpl_betas)
        self.loss_segm_mask = build_loss(loss_segm_mask)
        self.loss_camera = build_loss(loss_camera) #             loss_cam = ((torch.exp(-pred_cam[:, 0] * 10)) ** 2).mean()
        self.loss_vertex = build_loss(loss_vertex) #
        self.convention = convention
        self.loss_convention = loss_convention
        K, R = get_default_camera(focal_length=self.focal_length,
                                      img_size=self.image_res)

        self.register_buffer('K', K)
        self.register_buffer('R', R)
        
        set_requires_grad(self.body_model_train, False)
        set_requires_grad(self.body_model_test, False)
        
    def get_smpl_result(self, 
                        rotmat: torch.Tensor, 
                        shape: torch.Tensor, 
                        camera: torch.Tensor, 
                        img_res: int = 224, 
                        focal_length: int = 5000, 
                        normalize_joints2d: bool = False):
        smpl_output = self.body_model_train(
            betas=shape,
            body_pose=rotmat[:, 1:].contiguous(),
            global_orient=rotmat[:, 0].unsqueeze(1).contiguous(),
            pose2rot=False,
        )

        output = {
            'smpl_vertices': smpl_output.vertices,
            'smpl_joints3d': smpl_output.joints,
        }

        if camera is not None:
            joints3d = smpl_output.joints
            batch_size = joints3d.shape[0]
            device = joints3d.device
            
            
            joints2d, cam_t = project_points(smpl_output.joints, camera, focal_length, img_res)

            if normalize_joints2d:
                # Normalize keypoints to [-1,1]
                joints2d = joints2d / (img_res / 2.)

            output['smpl_joints2d'] = joints2d
            output['pred_cam_t'] = cam_t

        return output

    def compute_keypoints3d_loss(self, pred_keypoints3d: torch.Tensor,
                                 gt_keypoints3d: torch.Tensor,
                                 has_keypoints3d: torch.Tensor,):
        """Compute loss for 3d keypoints."""
        keypoints3d_conf = gt_keypoints3d[:, :, 3].float().unsqueeze(-1)
        keypoints3d_conf = keypoints3d_conf[has_keypoints3d == 1]
        keypoints3d_conf = keypoints3d_conf.repeat(1, 1, 3)
        pred_keypoints3d = pred_keypoints3d.float()
        pred_keypoints3d = pred_keypoints3d[has_keypoints3d == 1]
        gt_keypoints3d = gt_keypoints3d[:, :, :3].float()
        gt_keypoints3d_orig = gt_keypoints3d.clone()
        gt_keypoints3d = gt_keypoints3d[has_keypoints3d == 1]

        # currently, only mpi_inf_3dhp and h36m have 3d keypoints
        # both datasets have right_hip_extra and left_hip_extra
        right_hip_idx = get_keypoint_idx('right_hip_extra', self.loss_convention)
        left_hip_idx = get_keypoint_idx('left_hip_extra', self.loss_convention)
        gt_pelvis = (gt_keypoints3d[:, right_hip_idx, :] +
                     gt_keypoints3d[:, left_hip_idx, :]) / 2
        pred_pelvis = (pred_keypoints3d[:, right_hip_idx, :] +
                       pred_keypoints3d[:, left_hip_idx, :]) / 2

        gt_keypoints3d = gt_keypoints3d - gt_pelvis[:, None, :]
        pred_keypoints3d = pred_keypoints3d - pred_pelvis[:, None, :]
        loss = self.loss_keypoints3d(
            pred_keypoints3d, gt_keypoints3d, reduction_override='none')
        keypoints3d_conf = keypoints3d_conf
        if gt_keypoints3d.shape[0] == 0:
            return torch.Tensor([0]).type_as(gt_keypoints3d_orig)
        loss = (loss * keypoints3d_conf).mean()
        return loss

    def compute_keypoints2d_loss(self,
                                 pred_keypoints3d: torch.Tensor,
                                 pred_cam: torch.Tensor,
                                 gt_keypoints2d: torch.Tensor,
                                 img_res: Optional[int] = 224,
                                 focal_length: Optional[int] = 5000):
        """Compute loss for 2d keypoints."""
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)

        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()
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
            pred_keypoints2d, gt_keypoints2d, reduction_override='none')
        valid_pos = keypoints2d_conf > 0
        if keypoints2d_conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_keypoints2d)
        loss = (loss * keypoints2d_conf).mean()  
        return loss

    def compute_smpl_pose_loss(self, pred_rotmat: torch.Tensor,
                               gt_pose: torch.Tensor, has_smpl: torch.Tensor):
        """Compute loss for smpl pose."""
        conf = has_smpl.float().view(-1, 1, 1, 1).repeat(1, 24, 3, 3)
        gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
        loss = self.loss_smpl_pose(
            pred_rotmat, gt_rotmat, reduction_override='none')
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_pose)
        loss = torch.sum(loss * conf) / conf[valid_pos].numel()
        return loss

    def compute_smpl_betas_loss(self, pred_betas: torch.Tensor,
                                gt_betas: torch.Tensor,
                                has_smpl: torch.Tensor):
        """Compute loss for smpl betas."""
        conf = has_smpl.float().view(-1, 1).repeat(1, 10)
        loss = self.loss_smpl_betas(
            pred_betas, gt_betas, reduction_override='none')
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_betas)
        loss = torch.sum(loss * conf) / conf[valid_pos].numel()
        return loss

    def compute_camera_loss(self, cameras: torch.Tensor):
        """Compute loss for predicted camera parameters."""
        loss = self.loss_camera(cameras)
        return loss
    def compute_segm_loss(self,pred_seg: torch.Tensor,
                          gt_seg: torch.Tensor,
                          has_smpl: torch.Tensor):

        """Compute loss between part segmentation map and part attention map"""
        pred_seg = pred_seg[has_smpl == 1]
        gt_seg = gt_seg[has_smpl == 1]

        ph, pw = pred_seg.size(2), pred_seg.size(3)
        h, w = gt_seg.size(1), gt_seg.size(2)
        if ph != h or pw != w:
            pred_seg = F.interpolate(input=pred_seg, size=(h, w), mode='bilinear')

        loss = self.loss_segm_mask(pred_seg,gt_seg)
        return loss
    def train_step(self, data_batch, optimizer, **kwargs):
        """Train step function.

        In this function, train step is carried out
            with following the pipeline:
        1. extract features with the backbone
        2. feed the extracted features into the head to
            predicte beta, pose and camera translation
        3. compute regression losses of the predictions
            and optimize backbone and head
        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (dict[torch.optim.Optimizer]): Dict with optimizers for
                generator and discriminator (if have).
        Returns:
            outputs (dict): Dict with loss, information for logger,
            the number of samples.
        """


        img = data_batch['img']
 
        features = self.backbone(img)
        
        pare_output = self.head(features)

        targets = self.prepare_targets(data_batch)

        losses = self.compute_losses(pare_output, targets)

        loss, log_vars = self._parse_losses(losses)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))
        return outputs
    
    def prepare_targets(self, data_batch: dict):

        gt_keypoints_2d = data_batch['keypoints2d']  # 2D keypoints
        gt_pose = data_batch['smpl_body_pose'].float()
        global_orient = data_batch['smpl_global_orient'].float().view(-1, 1, 3)
        gt_pose = torch.cat((global_orient, gt_pose), dim=1).float()
        gt_betas = data_batch['smpl_betas'].float() # SMPL beta parameters

        batch_size = gt_keypoints_2d.shape[0]
        device = gt_pose.device
        
        gt_out = self.body_model_train(
            betas=gt_betas,
            body_pose=gt_pose[:, 3:],
            global_orient=gt_pose[:, :3],
            
        )
        gt_model_joints = gt_out['joints']
        gt_vertices = gt_out['vertices']
        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        # gt_keypoints_2d_orig[:, :, :-1] = \
        #     0.5 * self.image_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)
        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(
            gt_model_joints,
            gt_keypoints_2d_orig,
            focal_length=self.focal_length,
            img_size=self.image_res,
        )



        

        data_batch['gt_cam_t'] = gt_cam_t
        data_batch['vertices'] = gt_vertices
        data_batch['gt_segm_mask'] = render_smpl(
                verts = gt_vertices,
                R = self.R,
                K = self.K,
                T = gt_cam_t,
                render_choice = 'part_silhouette',
                resolution =  self.image_res,
                return_tensor = True,
                body_model = self.body_model_train,
                device=device,
                in_ndc=False,
                convention='pytorch3d',
                projection = 'perspective',
                no_grad = True,
                batch_size = batch_size,
            )
        data_batch['gt_segm_mask'] = torch.flip(data_batch['gt_segm_mask'] ,[1,2]).squeeze(-1).detach()
        # Image Mesh Estimator does not need extra process for ground truth
        return data_batch
    def compute_losses(self, predictions: dict, targets: dict):
        """Compute losses."""
        pred_betas = predictions['pred_shape'].view(-1, 10)
        pred_pose = predictions['pred_pose'].view(-1, 24, 3, 3)
        pred_cam = predictions['pred_cam'].view(-1, 3)
        pred_seg = predictions['pred_segm_mask']

        gt_keypoints3d = targets['keypoints3d']
        gt_keypoints2d = targets['keypoints2d']
        has_keypoints3d = targets['has_keypoints3d'].squeeze(-1)
        gt_seg = targets['gt_segm_mask']

        if self.body_model_train is not None:
            pred_output = self.body_model_train(
                betas=pred_betas,
                body_pose=pred_pose[:, 1:],
                global_orient=pred_pose[:, 0].unsqueeze(1),
                pose2rot=False,
                num_joints=gt_keypoints2d.shape[1])
            pred_keypoints3d = pred_output['joints']
            pred_vertices = pred_output['vertices']


        has_smpl = targets['has_smpl'].view(-1)
        gt_pose = targets['smpl_body_pose']
        global_orient = targets['smpl_global_orient'].view(-1, 1, 3)
        gt_pose = torch.cat((global_orient, gt_pose), dim=1).float()
        gt_betas = targets['smpl_betas'].float()

        if self.body_model_train is not None:
            gt_output = self.body_model_train(
                betas=gt_betas,
                body_pose=gt_pose[:, 3:],
                global_orient=gt_pose[:, :3],
                num_joints=gt_keypoints2d.shape[1])
            gt_vertices = gt_output['vertices']

        losses = {}

        pred_keypoints3d,_ = convert_kps(pred_keypoints3d,src = self.convention, dst = self.loss_convention)
        gt_keypoints3d,_ = convert_kps(gt_keypoints3d,src = self.convention, dst = self.loss_convention)
        gt_keypoints2d,_ = convert_kps(gt_keypoints2d,src = self.convention, dst = self.loss_convention)

        if self.loss_keypoints3d is not None:
            losses['keypoints3d_loss'] = self.compute_keypoints3d_loss(
                pred_keypoints3d, gt_keypoints3d,has_keypoints3d)
        if self.loss_keypoints2d is not None:
            losses['keypoints2d_loss'] = self.compute_keypoints2d_loss(
                pred_keypoints3d, pred_cam, gt_keypoints2d)
        if self.loss_smpl_pose is not None:
            losses['smpl_pose_loss'] = self.compute_smpl_pose_loss(
                pred_pose, gt_pose, has_smpl)
        if self.loss_smpl_betas is not None:
            losses['smpl_betas_loss'] = self.compute_smpl_betas_loss(
                pred_betas, gt_betas, has_smpl)
        if self.loss_camera is not None:
            losses['camera_loss'] = self.compute_camera_loss(pred_cam)
        if self.loss_segm_mask is not None:
            losses['part_seg_loss'] = self.compute_segm_loss(pred_seg, gt_seg, has_smpl)
        return losses


    def forward_train(self, **kwargs):
        """Forward function for general training.

        For mesh estimation, we do not use this interface.
        """
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule. Please use '
                                  '`train_step` for training.')

    
    def forward_test(self, img, img_metas, **kwargs):
        """Test step function.

        In this function, train step is carried out
            with following the pipeline:
        1. extract features with the backbone
        2. feed the extracted features into the head to
            predicte beta, pose, camera translation
        3. store predictions for evaluation
        Args:
            img (torch.Tensor): Batch of data as input.
            img_metas (dict): Dict with image metas i.e. path
            kwargs (dict): Dict with ground-truth
        Returns:
            all_preds (dict): Dict with image_path, image_idx, vertices, 
                                        keypoints_3d, smpl_pose, smpl_beta,camera

            xyz_24 for predictions.
        """
        features = self.backbone(img)

        

        predictions = self.head(features)
        pred_pose = predictions['pred_pose']
        pred_betas = predictions['pred_shape']
        pred_cam = predictions['pred_cam']

        pred_output = self.body_model_test(
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
        all_preds['image_idx'] = kwargs['sample_idx']

        return all_preds
        
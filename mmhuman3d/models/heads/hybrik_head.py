import numpy as np
import torch
import torch.cuda.comm
import torch.nn as nn
from mmcv.runner.base_module import BaseModule
from torch.nn import functional as F

from mmhuman3d.core.conventions.keypoints_mapping import get_flip_pairs
from ..builder import HEADS


def norm_heatmap(norm_type, heatmap):
    """Normalize heatmap.

    Args:
        norm_type (str):
            type of normalization. Currently only 'softmax' is supported
        heatmap (torch.Tensor):
            model output heatmap with shape (Bx29xF^2) where F^2 refers to
            number of squared feature channels F

    Returns:
        heatmap (torch.Tensor):
            normalized heatmap according to specified type with
            shape (Bx29xF^2)
    """

    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError


@HEADS.register_module()
class HybrIKHead(BaseModule):
    """HybrIK parameters regressor head.

    Args:
        feature_channel (int):
            Number of input channels
        deconv_dim (List[int]):
            List of deconvolution dimensions
        num_joints (int):
            Number of keypoints
        depth_dim (int):
            Depth dimension
        height_dim (int):
            Height dimension
        width_dim (int):
            Width dimension
        smpl_mean_params (str):
            file name of the mean SMPL parameters
    """

    def __init__(
        self,
        feature_channel=512,
        deconv_dim=[256, 256, 256],
        num_joints=29,
        depth_dim=64,
        height_dim=64,
        width_dim=64,
        smpl_mean_params=None,
    ):

        super(HybrIKHead, self).__init__()

        self.deconv_dim = deconv_dim
        self._norm_layer = nn.BatchNorm2d
        self.num_joints = num_joints
        self.norm_type = 'softmax'
        self.depth_dim = depth_dim
        self.height_dim = height_dim
        self.width_dim = width_dim
        self.smpl_dtype = torch.float32
        self.feature_channel = feature_channel

        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(
            self.deconv_dim[2],
            self.num_joints * self.depth_dim,
            kernel_size=1,
            stride=1,
            padding=0)

        self.joint_pairs_24 = get_flip_pairs('smpl')
        self.joint_pairs_29 = get_flip_pairs('hybrik_29')

        self.leaf_pairs = ((0, 1), (3, 4))
        self.root_idx_smpl = 0

        # mean shape
        init_shape = np.load(smpl_mean_params)
        self.register_buffer('init_shape', torch.Tensor(init_shape).float())

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.feature_channel, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]

    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(
            self.feature_channel,
            self.deconv_dim[0],
            kernel_size=4,
            stride=2,
            padding=int(4 / 2) - 1,
            bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(
            self.deconv_dim[0],
            self.deconv_dim[1],
            kernel_size=4,
            stride=2,
            padding=int(4 / 2) - 1,
            bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(
            self.deconv_dim[1],
            self.deconv_dim[2],
            kernel_size=4,
            stride=2,
            padding=int(4 / 2) - 1,
            bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])

        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*deconv_layers)

    def _initialize(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def uvd_to_cam(self,
                   uvd_jts,
                   trans_inv,
                   intrinsic_param,
                   joint_root,
                   depth_factor,
                   return_relative=True):
        """Project uvd coordinates to camera frame.

        Args:
            uvd_jts (torch.Tensor):
                uvd coordinates with shape (BxNum_jointsx3)
            trans_inv (torch.Tensor):
                inverse affine transformation matrix with shape (Bx2x3)
            intrinsic_param (torch.Tensor):
                camera intrinsic matrix with shape (Bx3x3)
            joint_root (torch.Tensor):
                root joint coordinate with shape (Bx3)
            depth_factor (float):
                depth factor with shape (Bx1)
            return_relative (bool):
                Store True to return root normalized relative coordinates.
                Default: True.

        Returns:
            xyz_jts (torch.Tensor):
                uvd coordinates in camera frame with shape (BxNum_jointsx3)
        """
        assert uvd_jts.dim() == 3 and uvd_jts.shape[2] == 3, uvd_jts.shape
        uvd_jts_new = uvd_jts.clone()
        # if torch.sum(torch.isnan(uvd_jts)) > 0:
        #     aaa= 1
        assert torch.sum(torch.isnan(uvd_jts)) == 0, ('uvd_jts', uvd_jts)

        # remap uv coordinate to input space
        uvd_jts_new[:, :, 0] = (uvd_jts[:, :, 0] + 0.5) * self.width_dim * 4
        uvd_jts_new[:, :, 1] = (uvd_jts[:, :, 1] + 0.5) * self.height_dim * 4
        # remap d to mm
        uvd_jts_new[:, :, 2] = uvd_jts[:, :, 2] * depth_factor
        assert torch.sum(torch.isnan(uvd_jts_new)) == 0, ('uvd_jts_new',
                                                          uvd_jts_new)

        dz = uvd_jts_new[:, :, 2]

        # transform in-bbox coordinate to image coordinate
        uv_homo_jts = torch.cat(
            (uvd_jts_new[:, :, :2], torch.ones_like(uvd_jts_new)[:, :, 2:]),
            dim=2)
        # batch-wise matrix multiply : (B,1,2,3) * (B,K,3,1) -> (B,K,2,1)
        uv_jts = torch.matmul(
            trans_inv.unsqueeze(1), uv_homo_jts.unsqueeze(-1))
        # transform (u,v,1) to (x,y,z)
        cam_2d_homo = torch.cat((uv_jts, torch.ones_like(uv_jts)[:, :, :1, :]),
                                dim=2)
        # batch-wise matrix multiply : (B,1,3,3) * (B,K,3,1) -> (B,K,3,1)
        xyz_jts = torch.matmul(intrinsic_param.unsqueeze(1), cam_2d_homo)
        xyz_jts = xyz_jts.squeeze(dim=3)
        # recover absolute z : (B,K) + (B,1)
        abs_z = dz + joint_root[:, 2].unsqueeze(-1)
        # multiply absolute z : (B,K,3) * (B,K,1)
        xyz_jts = xyz_jts * abs_z.unsqueeze(-1)

        if return_relative:
            # (B,K,3) - (B,1,3)
            xyz_jts = xyz_jts - joint_root.unsqueeze(1)

        xyz_jts = xyz_jts / depth_factor.unsqueeze(-1)

        return xyz_jts

    def flip_uvd_coord(self, pred_jts, flip=False, flatten=True):
        """Flip uvd coordinates.

        Args:
            pred_jts (torch.Tensor):
                predicted uvd coordinates with shape (Bx87)
            flip (bool):
                Store True to flip uvd coordinates. Default: False.
            flatten (bool):
                Store True to reshape uvd_coordinates to shape (Bx29x3)
                Default: True

        Returns:
            pred_jts (torch.Tensor):
                flipped uvd coordinates with shape (Bx29x3)
        """
        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, self.num_joints, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]

        # flip
        if flip:
            pred_jts[:, :, 0] = -pred_jts[:, :, 0]
        else:
            pred_jts[:, :, 0] = -1 / self.width_dim - pred_jts[:, :, 0]

        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_jts[:, idx] = pred_jts[:, inv_idx]

        return pred_jts

    def flip_phi(self, pred_phi):
        """Flip phi.

        Args:
            pred_phi (torch.Tensor): phi in shape (Num_twistx2)

        Returns:
            pred_phi (torch.Tensor): flipped phi in shape (Num_twistx2)
        """
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]

        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]

        return pred_phi

    def forward(self,
                feature,
                trans_inv,
                intrinsic_param,
                joint_root,
                depth_factor,
                smpl_layer,
                flip_item=None,
                flip_output=False):
        """Forward function.

        Args:
            feature (torch.Tensor): features extracted from backbone
            trans_inv (torch.Tensor):
                inverse affine transformation matrix with shape (Bx2x3)
            intrinsic_param (torch.Tensor):
                camera intrinsic matrix with shape (Bx3x3)
            joint_root (torch.Tensor):
                root joint coordinate with shape (Bx3)
            depth_factor (float):
                depth factor with shape (Bx1)
            smpl_layer (torch.Tensor):
                smpl body model
            flip_item (List[torch.Tensor]|None):
                list containing items to flip
            flip_output (bool):
                Store True to flip output. Default: False

        Returns:
            output (dict): Dict containing model predictions.
        """
        batch_size = feature.shape[0]

        x0 = feature
        out = self.deconv_layers(x0)
        out = self.final_layer(out)

        out = out.reshape((out.shape[0], self.num_joints, -1))
        out = norm_heatmap(self.norm_type, out)
        assert out.dim() == 3, out.shape

        if self.norm_type == 'sigmoid':
            maxvals, _ = torch.max(out, dim=2, keepdim=True)
        else:
            maxvals = torch.ones((*out.shape[:2], 1),
                                 dtype=torch.float,
                                 device=out.device)

        heatmaps = out / out.sum(dim=2, keepdim=True)

        heatmaps = heatmaps.reshape(
            (heatmaps.shape[0], self.num_joints, self.depth_dim,
             self.height_dim, self.width_dim))

        hm_x = heatmaps.sum((2, 3))
        hm_y = heatmaps.sum((2, 4))
        hm_z = heatmaps.sum((3, 4))

        hm_x = hm_x * torch.cuda.comm.broadcast(
            torch.arange(hm_x.shape[-1]).type(torch.cuda.FloatTensor),
            devices=[hm_x.device.index])[0]
        hm_y = hm_y * torch.cuda.comm.broadcast(
            torch.arange(hm_y.shape[-1]).type(torch.cuda.FloatTensor),
            devices=[hm_y.device.index])[0]
        hm_z = hm_z * torch.cuda.comm.broadcast(
            torch.arange(hm_z.shape[-1]).type(torch.cuda.FloatTensor),
            devices=[hm_z.device.index])[0]
        coord_x = hm_x.sum(dim=2, keepdim=True)
        coord_y = hm_y.sum(dim=2, keepdim=True)
        coord_z = hm_z.sum(dim=2, keepdim=True)

        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5
        coord_z = coord_z / float(self.depth_dim) - 0.5

        #  -0.5 ~ 0.5
        pred_uvd_jts_29 = torch.cat((coord_x, coord_y, coord_z), dim=2)

        pred_uvd_jts_29_flat = pred_uvd_jts_29.reshape(
            (batch_size, self.num_joints * 3))

        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)  # (B, 10,)

        xc = x0

        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)

        delta_shape = self.decshape(xc)
        pred_shape = delta_shape + init_shape
        pred_phi = self.decphi(xc)

        if flip_item is not None:
            assert flip_output
            pred_uvd_jts_29_orig, pred_phi_orig, pred_leaf_orig, \
                pred_shape_orig = flip_item

        if flip_output:
            pred_uvd_jts_29 = self.flip_uvd_coord(
                pred_uvd_jts_29, flatten=False, shift=True)
        if flip_output and flip_item is not None:
            pred_uvd_jts_29 = (pred_uvd_jts_29 + pred_uvd_jts_29_orig.reshape(
                batch_size, 29, 3)) / 2

        pred_uvd_jts_29_flat = pred_uvd_jts_29.reshape(
            (batch_size, self.num_joints * 3))

        #  -0.5 ~ 0.5
        # Rotate back
        pred_xyz_jts_29 = self.uvd_to_cam(pred_uvd_jts_29, trans_inv,
                                          intrinsic_param, joint_root,
                                          depth_factor)
        assert torch.sum(
            torch.isnan(pred_xyz_jts_29)) == 0, ('pred_xyz_jts_29',
                                                 pred_xyz_jts_29)

        pred_xyz_jts_29 = pred_xyz_jts_29 - \
            pred_xyz_jts_29[:, self.root_idx_smpl, :].unsqueeze(1)

        pred_phi = pred_phi.reshape(batch_size, 23, 2)

        if flip_output:
            pred_phi = self.flip_phi(pred_phi)

        if flip_output and flip_item is not None:
            pred_phi = (pred_phi + pred_phi_orig) / 2
            pred_shape = (pred_shape + pred_shape_orig) / 2

        hybrik_output = smpl_layer(
            pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) * 2,
            betas=pred_shape.type(self.smpl_dtype),
            phis=pred_phi.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True)
        pred_vertices = hybrik_output['vertices'].float()
        #  -0.5 ~ 0.5
        pred_xyz_jts_24_struct = hybrik_output['joints'].float() / 2
        #  -0.5 ~ 0.5
        pred_xyz_jts_17 = hybrik_output['joints_from_verts'].float() / 2
        pred_theta_mats = hybrik_output['rot_mats'].float().reshape(
            batch_size, 24 * 4)
        pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17 = pred_xyz_jts_17.reshape(batch_size, 17 * 3)

        output = {
            'pred_phi': pred_phi,
            'pred_delta_shape': delta_shape,
            'pred_shape': pred_shape,
            'pred_theta_mats': pred_theta_mats,
            'pred_uvd_jts': pred_uvd_jts_29_flat,
            'pred_xyz_jts_24': pred_xyz_jts_24,
            'pred_xyz_jts_24_struct': pred_xyz_jts_24_struct,
            'pred_xyz_jts_17': pred_xyz_jts_17,
            'pred_vertices': pred_vertices,
            'maxvals': maxvals,
        }

        return output

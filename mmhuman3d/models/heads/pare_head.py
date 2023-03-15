"""This script is modified from [PARE](https://github.com/
mkocabas/PARE/tree/master/pare/models/layers).

Original license please see docs/additional_licenses.md.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule
from torch.nn.modules.utils import _pair

from mmhuman3d.utils.geometry import rot6d_to_rotmat


class LocallyConnected2d(nn.Module):
    """Locally Connected Layer.

    Args:
        in_channels (int):
            the in channel of the features.
        out_channels (int):
            the out channel of the features.
        output_size (List[int]):
            the output size of the features.
        kernel_size (int):
            the size of the kernel.
        stride (int):
            the stride of the kernel.
    Returns:
        attended_features (torch.Tensor):
            attended feature maps
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 output_size,
                 kernel_size,
                 stride,
                 bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0],
                        output_size[1], kernel_size**2),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1]),
                requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        """Forward function."""
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class KeypointAttention(nn.Module):
    """Keypoint Attention Layer.

    Args:
        use_conv (bool):
            whether to use conv for the attended feature map.
            Default: False
        in_channels (List[int]):
            the in channel of shape_cam features and pose features.
            Default: (256, 64)
        out_channels (List[int]):
            the out channel of shape_cam features and pose features.
            Default: (256, 64)
    Returns:
        attended_features (torch.Tensor):
            attended feature maps
    """

    def __init__(self,
                 use_conv=False,
                 in_channels=(256, 64),
                 out_channels=(256, 64),
                 act='softmax',
                 use_scale=False):
        super(KeypointAttention, self).__init__()
        self.use_conv = use_conv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.use_scale = use_scale
        if use_conv:
            self.conv1x1_pose = nn.Conv1d(
                in_channels[0], out_channels[0], kernel_size=1)
            self.conv1x1_shape_cam = nn.Conv1d(
                in_channels[1], out_channels[1], kernel_size=1)

    def forward(self, features, heatmaps):
        """Forward function."""
        batch_size, num_joints, height, width = heatmaps.shape

        if self.use_scale:
            scale = 1.0 / np.sqrt(height * width)
            heatmaps = heatmaps * scale

        if self.act == 'softmax':
            normalized_heatmap = F.softmax(
                heatmaps.reshape(batch_size, num_joints, -1), dim=-1)
        elif self.act == 'sigmoid':
            normalized_heatmap = torch.sigmoid(
                heatmaps.reshape(batch_size, num_joints, -1))
        features = features.reshape(batch_size, -1, height * width)

        attended_features = torch.matmul(normalized_heatmap,
                                         features.transpose(2, 1))
        attended_features = attended_features.transpose(2, 1)

        if self.use_conv:
            if attended_features.shape[1] == self.in_channels[0]:
                attended_features = self.conv1x1_pose(attended_features)
            else:
                attended_features = self.conv1x1_shape_cam(attended_features)

        return attended_features


def interpolate(feat, uv):
    """
    Args:
        feat (torch.Tensor): [B, C, H, W] image features
        uv (torch.Tensor): [B, 2, N] uv coordinates
            in the image plane, range [-1, 1]
    Returns:
        samples[:, :, :, 0] (torch.Tensor):
            [B, C, N] image features at the uv coordinates
    """
    if uv.shape[-1] != 2:
        uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training
    # results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    if int(torch.__version__.split('.')[1]) < 4:
        samples = torch.nn.functional.grid_sample(feat, uv)  # [B, C, N, 1]
    else:
        samples = torch.nn.functional.grid_sample(
            feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]


def _softmax(tensor, temperature, dim=-1):
    return F.softmax(tensor * temperature, dim=dim)


def softargmax2d(
    heatmaps,
    temperature=None,
    normalize_keypoints=True,
):
    """Softargmax layer for heatmaps."""
    dtype, device = heatmaps.dtype, heatmaps.device
    if temperature is None:
        temperature = torch.tensor(1.0, dtype=dtype, device=device)
    batch_size, num_channels, height, width = heatmaps.shape
    x = torch.arange(
        0, width, device=device,
        dtype=dtype).reshape(1, 1, 1, width).expand(batch_size, -1, height, -1)
    y = torch.arange(
        0, height, device=device,
        dtype=dtype).reshape(1, 1, height, 1).expand(batch_size, -1, -1, width)
    # Should be Bx2xHxW
    points = torch.cat([x, y], dim=1)
    normalized_heatmap = _softmax(
        heatmaps.reshape(batch_size, num_channels, -1),
        temperature=temperature.reshape(1, -1, 1),
        dim=-1)

    # Should be BxJx2
    keypoints = (
        normalized_heatmap.reshape(batch_size, -1, 1, height * width) *
        points.reshape(batch_size, 1, 2, -1)).sum(dim=-1)

    if normalize_keypoints:
        # Normalize keypoints to [-1, 1]
        keypoints[:, :, 0] = (keypoints[:, :, 0] / (width - 1) * 2 - 1)
        keypoints[:, :, 1] = (keypoints[:, :, 1] / (height - 1) * 2 - 1)

    return keypoints, normalized_heatmap.reshape(batch_size, -1, height, width)


class PareHead(BaseModule):

    def __init__(
        self,
        num_joints=24,
        num_input_features=480,
        softmax_temp=1.0,
        num_deconv_layers=3,
        num_deconv_filters=(256, 256, 256),
        num_deconv_kernels=(4, 4, 4),
        num_camera_params=3,
        num_features_smpl=64,
        final_conv_kernel=1,
        pose_mlp_num_layers=1,
        shape_mlp_num_layers=1,
        pose_mlp_hidden_size=256,
        shape_mlp_hidden_size=256,
        bn_momentum=0.1,
        use_heatmaps='part_segm',
        use_keypoint_attention=False,
        use_postconv_keypoint_attention=False,
        keypoint_attention_act='softmax',  # softmax, sigmoid
        use_scale_keypoint_attention=False,
        backbone='hrnet_w32-conv',  # hrnet, resnet
        smpl_mean_params=None,
        deconv_with_bias=False,
    ):
        """PARE parameters regressor head. This class is modified from.

        [PARE](hhttps://github.com/
        mkocabas/PARE/blob/master/pare/models/head/pare_head.py). Original
        license please see docs/additional_licenses.md.

        Args:
            num_joints (int):
                Number of joints, should be 24 for smpl.
            num_input_features (int):
                Number of input featuremap channels.
            softmax_temp (float):
                Softmax tempreture
            num_deconv_layers (int):
                Number of deconvolution layers.
            num_deconv_filters (List[int]):
                Number of filters for each deconvolution layer,
                len(num_deconv_filters) == num_deconv_layers.
            num_deconv_kernels (List[int]):
                Kernel size  for each deconvolution layer,
                len(num_deconv_kernels) == num_deconv_layers.
            num_camera_params (int):
                Number of predicted camera parameter dimension.
            num_features_smpl (int):
                Number of feature map channels.
            final_conv_kernel (int):
                Kernel size for the final deconvolution feature map channels.
            pose_mlp_num_layers (int):
                Number of mpl layers for pose parameter regression.
            shape_mlp_num_layers (int):
                Number of mpl layers for pose parameter regression.
            pose_mlp_hidden_size (int):
                Hidden size for pose mpl layers.
            shape_mlp_hidden_size (int):
                Hidden size for pose mpl layers.
            bn_momemtum (float):
                Momemtum for batch normalization.
            use_heatmaps (str):
                Types of heat maps to use.
            use_keypoint_attention (bool)
                Whether to use attention based on heat maps.
            keypoint_attention_act (str):
                Types of activation function for attention layers.
            use_scale_keypoint_attention (str):
                Whether to scale the attention
                according to the size of the attention map.
            deconv_with_bias (bool)
                Whether to deconv with bias.
            backbone (str):
                Types of the backbone.
            smpl_mean_params (str):
                File name of the mean SMPL parameters
        """

        super(PareHead, self).__init__()
        self.backbone = backbone
        self.num_joints = num_joints
        self.deconv_with_bias = deconv_with_bias
        self.use_heatmaps = use_heatmaps
        self.pose_mlp_num_layers = pose_mlp_num_layers
        self.shape_mlp_num_layers = shape_mlp_num_layers
        self.pose_mlp_hidden_size = pose_mlp_hidden_size
        self.shape_mlp_hidden_size = shape_mlp_hidden_size
        self.use_keypoint_attention = use_keypoint_attention

        self.num_input_features = num_input_features
        self.bn_momentum = bn_momentum
        if self.use_heatmaps == 'part_segm':

            self.use_keypoint_attention = True

        if backbone.startswith('hrnet'):

            self.keypoint_deconv_layers = self._make_conv_layer(
                num_deconv_layers,
                num_deconv_filters,
                (3, ) * num_deconv_layers,
            )
            self.num_input_features = num_input_features
            self.smpl_deconv_layers = self._make_conv_layer(
                num_deconv_layers,
                num_deconv_filters,
                (3, ) * num_deconv_layers,
            )
        else:
            # part branch that estimates 2d keypoints

            conv_fn = self._make_deconv_layer

            self.keypoint_deconv_layers = conv_fn(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
            # reset inplanes to 2048 -> final resnet layer
            self.num_input_features = num_input_features
            self.smpl_deconv_layers = conv_fn(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )

        pose_mlp_inp_dim = num_deconv_filters[-1]
        smpl_final_dim = num_features_smpl
        shape_mlp_inp_dim = num_joints * smpl_final_dim

        self.keypoint_final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=num_joints +
            1 if self.use_heatmaps in ('part_segm',
                                       'part_segm_pool') else num_joints,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=1 if final_conv_kernel == 3 else 0,
        )

        self.smpl_final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=smpl_final_dim,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=1 if final_conv_kernel == 3 else 0,
        )

        # temperature for softargmax function
        self.register_buffer('temperature', torch.tensor(softmax_temp))
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(
            mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        self.pose_mlp_inp_dim = pose_mlp_inp_dim
        self.shape_mlp_inp_dim = shape_mlp_inp_dim

        self.shape_mlp = self._get_shape_mlp(output_size=10)
        self.cam_mlp = self._get_shape_mlp(output_size=num_camera_params)

        self.pose_mlp = self._get_pose_mlp(
            num_joints=num_joints, output_size=6)

        self.keypoint_attention = KeypointAttention(
            use_conv=use_postconv_keypoint_attention,
            in_channels=(self.pose_mlp_inp_dim, smpl_final_dim),
            out_channels=(self.pose_mlp_inp_dim, smpl_final_dim),
            act=keypoint_attention_act,
            use_scale=use_scale_keypoint_attention,
        )

    def _get_shape_mlp(self, output_size):
        """mlp layers for shape regression."""
        if self.shape_mlp_num_layers == 1:
            return nn.Linear(self.shape_mlp_inp_dim, output_size)

        module_list = []
        for i in range(self.shape_mlp_num_layers):
            if i == 0:
                module_list.append(
                    nn.Linear(self.shape_mlp_inp_dim,
                              self.shape_mlp_hidden_size))
            elif i == self.shape_mlp_num_layers - 1:
                module_list.append(
                    nn.Linear(self.shape_mlp_hidden_size, output_size))
            else:
                module_list.append(
                    nn.Linear(self.shape_mlp_hidden_size,
                              self.shape_mlp_hidden_size))
        return nn.Sequential(*module_list)

    def _get_pose_mlp(self, num_joints, output_size):
        """mlp layers for pose regression."""
        if self.pose_mlp_num_layers == 1:

            return LocallyConnected2d(
                in_channels=self.pose_mlp_inp_dim,
                out_channels=output_size,
                output_size=[num_joints, 1],
                kernel_size=1,
                stride=1,
            )

        module_list = []
        for i in range(self.pose_mlp_num_layers):
            if i == 0:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_inp_dim,
                        out_channels=self.pose_mlp_hidden_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    ))
            elif i == self.pose_mlp_num_layers - 1:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_hidden_size,
                        out_channels=output_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    ))
            else:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_hidden_size,
                        out_channels=self.pose_mlp_hidden_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    ))
        return nn.Sequential(*module_list)

    def _get_deconv_cfg(self, deconv_kernel):
        """get deconv padding, output padding according to kernel size."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_conv_layer(self, num_layers, num_filters, num_kernels):
        """make convolution layers."""
        assert num_layers == len(num_filters), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.Conv2d(
                    in_channels=self.num_input_features,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """make deconvolution layers."""
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.num_input_features,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            # if self.use_self_attention:
            #     layers.append(SelfAttention(planes))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def forward(self, features):
        """Forward function."""
        batch_size = features.shape[0]

        init_pose = self.init_pose.expand(batch_size, -1)  # N, Jx6
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        output = {}

        part_feats = self._get_2d_branch_feats(features)

        part_attention = self._get_part_attention_map(part_feats, output)

        smpl_feats = self._get_3d_smpl_feats(features, part_feats)

        point_local_feat, cam_shape_feats = self._get_local_feats(
            smpl_feats, part_attention, output)

        pred_pose, pred_shape, pred_cam = self._get_final_preds(
            point_local_feat, cam_shape_feats, init_pose, init_shape, init_cam)

        pred_rotmat = rot6d_to_rotmat(pred_pose).reshape(batch_size, 24, 3, 3)

        output.update({
            'pred_pose': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
        })
        return output

    def _get_local_feats(self, smpl_feats, part_attention, output):
        # 1x1 conv
        """get keypoints and camera features from backbone features."""

        cam_shape_feats = self.smpl_final_layer(smpl_feats)

        if self.use_keypoint_attention:
            point_local_feat = self.keypoint_attention(smpl_feats,
                                                       part_attention)
            cam_shape_feats = self.keypoint_attention(cam_shape_feats,
                                                      part_attention)
        else:
            point_local_feat = interpolate(smpl_feats, output['pred_kp2d'])
            cam_shape_feats = interpolate(cam_shape_feats, output['pred_kp2d'])
        return point_local_feat, cam_shape_feats

    def _get_2d_branch_feats(self, features):
        """get part features from backbone features."""
        part_feats = self.keypoint_deconv_layers(features)

        return part_feats

    def _get_3d_smpl_feats(self, features, part_feats):
        """get smpl feature maps from backbone features."""

        smpl_feats = self.smpl_deconv_layers(features)

        return smpl_feats

    def _get_part_attention_map(self, part_feats, output):
        """get attention map from part feature map."""
        heatmaps = self.keypoint_final_layer(part_feats)

        if self.use_heatmaps == 'part_segm':

            output['pred_segm_mask'] = heatmaps
            # remove the the background channel
            heatmaps = heatmaps[:, 1:, :, :]
        else:
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature)
            output['pred_kp2d'] = pred_kp2d
            output['pred_heatmaps_2d'] = heatmaps
        return heatmaps

    def _get_final_preds(self, pose_feats, cam_shape_feats, init_pose,
                         init_shape, init_cam):
        """get final preds."""
        return self._pare_get_final_preds(pose_feats, cam_shape_feats,
                                          init_pose, init_shape, init_cam)

    def _pare_get_final_preds(self, pose_feats, cam_shape_feats, init_pose,
                              init_shape, init_cam):
        """get final preds."""
        pose_feats = pose_feats.unsqueeze(-1)  #

        if init_pose.shape[-1] == 6:
            # This means init_pose comes from a previous iteration
            init_pose = init_pose.transpose(2, 1).unsqueeze(-1)
        else:
            # This means init pose comes from mean pose
            init_pose = init_pose.reshape(init_pose.shape[0], 6,
                                          -1).unsqueeze(-1)

        shape_feats = cam_shape_feats

        shape_feats = torch.flatten(shape_feats, start_dim=1)

        pred_pose = self.pose_mlp(pose_feats)
        pred_cam = self.cam_mlp(shape_feats)
        pred_shape = self.shape_mlp(shape_feats)

        pred_pose = pred_pose.squeeze(-1).transpose(2, 1)  # N, J, 6
        return pred_pose, pred_shape, pred_cam

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner.base_module import BaseModule

from mmhuman3d.models.backbones.resnet import BasicBlock
from mmhuman3d.utils.geometry import rot6d_to_rotmat
from ..builder import HEADS
from ..utils.pare_utils import (
    KeypointAttention,
    LocallyConnected2d,
    interpolate,
    softargmax2d,
)

BN_MOMENTUM = 0.1


@HEADS.register_module()
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
        use_heatmaps='part_segm',
        use_keypoint_attention=False,
        use_postconv_keypoint_attention=False,
        keypoint_attention_act='softmax',  # softmax, sigmoid
        use_scale_keypoint_attention=False,
        backbone='hrnet_w32-conv',  # hrnet, resnet
        smpl_mean_params=None,
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
            use_heatmaps (str):
                Types of heat maps to use.
            use_keypoint_attention (bool)
                Whether to use attention based on heat maps.
            keypoint_attention_act (str):
                Types of activation function for attention layers.
            use_scale_keypoint_attention (str):
                Whether to scale the attention
                according to the size of the attention map.
            backbone (str):
                Types of the backbone.
            smpl_mean_params (str):
                File name of the mean SMPL parameters
        """

        super(PareHead, self).__init__()
        self.backbone = backbone
        self.num_joints = num_joints
        self.deconv_with_bias = False
        self.use_heatmaps = use_heatmaps
        self.pose_mlp_num_layers = pose_mlp_num_layers
        self.shape_mlp_num_layers = shape_mlp_num_layers
        self.pose_mlp_hidden_size = pose_mlp_hidden_size
        self.shape_mlp_hidden_size = shape_mlp_hidden_size
        self.use_keypoint_attention = use_keypoint_attention

        self.num_input_features = num_input_features

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
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _make_res_conv_layers(self,
                              input_channels,
                              num_channels=64,
                              num_heads=1,
                              num_basic_blocks=2):
        head_layers = []

        head_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1),
                nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)))

        for i in range(num_heads):
            layers = []
            for _ in range(num_basic_blocks):
                layers.append(
                    nn.Sequential(BasicBlock(num_channels, num_channels)))
            head_layers.append(nn.Sequential(*layers))

        return nn.Sequential(*head_layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
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
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            # if self.use_self_attention:
            #     layers.append(SelfAttention(planes))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _make_upsample_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_layers is different len(num_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_layers is different len(num_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(
                nn.Conv2d(
                    in_channels=self.num_input_features,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            # if self.use_self_attention:
            #     layers.append(SelfAttention(planes))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def forward(self, features):
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
        part_feats = self.keypoint_deconv_layers(features)

        return part_feats

    def _get_3d_smpl_feats(self, features, part_feats):

        smpl_feats = self.smpl_deconv_layers(features)

        return smpl_feats

    def _get_part_attention_map(self, part_feats, output):

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
        return self._pare_get_final_preds(pose_feats, cam_shape_feats,
                                          init_pose, init_shape, init_cam)

    def _pare_get_final_preds(self, pose_feats, cam_shape_feats, init_pose,
                              init_shape, init_cam):
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

# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, ModuleList, Sequential
from torch.nn.modules.batchnorm import _BatchNorm

from .resnet import BasicBlock, Bottleneck


class HRModule(BaseModule):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output=True,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 block_init_cfg=None,
                 init_cfg=None):
        super(HRModule, self).__init__(init_cfg)
        self.block_init_cfg = block_init_cfg
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, num_blocks, in_channels,
                        num_channels):
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_BLOCKS({len(num_blocks)})'
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_CHANNELS({len(num_channels)})'
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.in_channels[branch_index] != \
                num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, num_channels[branch_index] *
                                 block.expansion)[1])

        layers = []
        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index],
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                init_cfg=self.block_init_cfg))
        self.in_channels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    init_cfg=self.block_init_cfg))

        return Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.in_channels

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class PoseHighResolutionNet(BaseModule):
    """HRNet backbone.
    `High-Resolution Representations for Labeling Pixels and Regions
    arXiv: <https://arxiv.org/abs/1904.04514>`_.
    Args:
        extra (dict): Detailed configuration for each stage of HRNet.
            There must be 4 stages, the configuration for each stage must have
            5 keys:
                - num_modules(int): The number of HRModule in this stage.
                - num_branches(int): The number of branches in the HRModule.
                - block(str): The type of convolution block.
                - num_blocks(tuple): The number of blocks in each branch.
                    The length must be equal to num_branches.
                - num_channels(tuple): The number of channels in each branch.
                    The length must be equal to num_branches.
        in_channels (int): Number of input image channels. Default: 3.
        conv_cfg (dict): Dictionary to construct and config conv layer.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: False.
        multiscale_output (bool): Whether to output multi-level features
            produced by multiple branches. If False, only the first level
            feature will be output. Default: True.
        num_joints(int): the number of output for the final layer. Default: 24.
        pretrained (str, optional): Model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=True,
                 with_cp=False,
                 num_joints=24,
                 zero_init_residual=False,
                 multiscale_output=True,
                 pretrained=None,
                 init_cfg=None):
        super(PoseHighResolutionNet, self).__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        # Assert configurations of 4 stages are in extra
        assert 'stage1' in extra and 'stage2' in extra \
               and 'stage3' in extra and 'stage4' in extra
        # Assert whether the length of `num_blocks` and `num_channels` are
        # equal to `num_branches`
        for i in range(4):
            cfg = extra[f'stage{i + 1}']
            assert len(cfg['num_blocks']) == cfg['num_branches'] and \
                   len(cfg['num_channels']) == cfg['num_branches']

        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        # stem net
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)

        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition1 = self._make_transition_layer([stage1_out_channels],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multiscale_output=multiscale_output)
        # self.pretrained_layers = extra['pretrained_layers']
        self.final_layer = build_conv_layer(
            cfg=self.conv_cfg,
            in_channels=pre_stage_channels[0],
            out_channels=num_joints,
            kernel_size=extra['final_conv_kernel'],
            stride=1,
            padding=1 if extra['final_conv_kernel'] == 3 else 0)
        if extra['downsample'] and extra['use_conv']:
            self.downsample_stage_1 = self._make_downsample_layer(
                3, num_channel=self.stage2_cfg['num_channels'][0])
            self.downsample_stage_2 = self._make_downsample_layer(
                2, num_channel=self.stage2_cfg['num_channels'][-1])
            self.downsample_stage_3 = self._make_downsample_layer(
                1, num_channel=self.stage3_cfg['num_channels'][-1])
        elif not extra['downsample'] and extra['use_conv']:
            self.upsample_stage_2 = self._make_upsample_layer(
                1, num_channel=self.stage2_cfg['num_channels'][-1])
            self.upsample_stage_3 = self._make_upsample_layer(
                2, num_channel=self.stage3_cfg['num_channels'][-1])
            self.upsample_stage_4 = self._make_upsample_layer(
                3, num_channel=self.stage4_cfg['num_channels'][-1])

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = []
        block_init_cfg = None
        if self.pretrained is None and not hasattr(
                self, 'init_cfg') and self.zero_init_residual:
            if block is BasicBlock:
                block_init_cfg = dict(
                    type='Constant', val=0, override=dict(name='norm2'))
            elif block is Bottleneck:
                block_init_cfg = dict(
                    type='Constant', val=0, override=dict(name='norm3'))
        layers.append(
            block(
                inplanes,
                planes,
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                init_cfg=block_init_cfg,
            ))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    init_cfg=block_init_cfg))

        return Sequential(*layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        hr_modules = []
        block_init_cfg = None
        if self.pretrained is None and not hasattr(
                self, 'init_cfg') and self.zero_init_residual:
            if block is BasicBlock:
                block_init_cfg = dict(
                    type='Constant', val=0, override=dict(name='norm2'))
            elif block is Bottleneck:
                block_init_cfg = dict(
                    type='Constant', val=0, override=dict(name='norm3'))

        for i in range(num_modules):
            # multiscale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    block_init_cfg=block_init_cfg))

        return Sequential(*hr_modules), in_channels

    def _make_upsample_layer(self, num_layers, num_channel, kernel_size=3):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(
                build_conv_layer(
                    cfg=self.conv_cfg,
                    in_channels=num_channel,
                    out_channels=num_channel,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1,
                    bias=False,
                ))
            layers.append(build_norm_layer(self.norm_cfg, num_channel)[1])
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_downsample_layer(self, num_layers, num_channel, kernel_size=3):
        layers = []
        for i in range(num_layers):
            layers.append(
                build_conv_layer(
                    cfg=self.conv_cfg,
                    in_channels=num_channel,
                    out_channels=num_channel,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                    bias=False,
                ))
            layers.append(build_norm_layer(self.norm_cfg, num_channel)[1])
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        if self.extra['return_list']:
            return y_list
        elif self.extra['downsample']:
            if self.extra['use_conv']:
                # Downsampling with strided convolutions
                x1 = self.downsample_stage_1(y_list[0])
                x2 = self.downsample_stage_2(y_list[1])
                x3 = self.downsample_stage_3(y_list[2])
                x = torch.cat([x1, x2, x3, y_list[3]], 1)
            else:
                # Downsampling with interpolation
                x0_h, x0_w = y_list[3].size(2), y_list[3].size(3)
                x1 = F.interpolate(
                    y_list[0],
                    size=(x0_h, x0_w),
                    mode='bilinear',
                    align_corners=True)
                x2 = F.interpolate(
                    y_list[1],
                    size=(x0_h, x0_w),
                    mode='bilinear',
                    align_corners=True)
                x3 = F.interpolate(
                    y_list[2],
                    size=(x0_h, x0_w),
                    mode='bilinear',
                    align_corners=True)
                x = torch.cat([x1, x2, x3, y_list[3]], 1)
        else:
            if self.extra['use_conv']:
                # Upsampling with interpolations + convolutions
                x1 = self.upsample_stage_2(y_list[1])
                x2 = self.upsample_stage_3(y_list[2])
                x3 = self.upsample_stage_4(y_list[3])
                x = torch.cat([y_list[0], x1, x2, x3], 1)
            else:
                # Upsampling with interpolation
                x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
                x1 = F.interpolate(
                    y_list[1],
                    size=(x0_h, x0_w),
                    mode='bilinear',
                    align_corners=True)
                x2 = F.interpolate(
                    y_list[2],
                    size=(x0_h, x0_w),
                    mode='bilinear',
                    align_corners=True)
                x3 = F.interpolate(
                    y_list[3],
                    size=(x0_h, x0_w),
                    mode='bilinear',
                    align_corners=True)
                x = torch.cat([y_list[0], x1, x2, x3], 1)
        return x

    def train(self, mode=True):
        """Convert the model into training mode will keeping the normalization
        layer freezed."""
        super(PoseHighResolutionNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


class PoseHighResolutionNetExpose(PoseHighResolutionNet):
    """HRNet backbone for expose."""

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=True,
                 with_cp=False,
                 num_joints=24,
                 zero_init_residual=False,
                 multiscale_output=True,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(extra, in_channels, conv_cfg, norm_cfg, norm_eval,
                         with_cp, num_joints, zero_init_residual,
                         multiscale_output, pretrained, init_cfg)
        in_dims = (2**2 * self.stage2_cfg['num_channels'][-1] +
                   2**1 * self.stage3_cfg['num_channels'][-1] +
                   self.stage4_cfg['num_channels'][-1])
        self.conv_layers = self._make_conv_layer(
            in_channels=in_dims, num_layers=5)
        self.subsample_3 = self._make_subsample_layer(
            in_channels=self.stage2_cfg['num_channels'][-1], num_layers=2)
        self.subsample_2 = self._make_subsample_layer(
            in_channels=self.stage3_cfg['num_channels'][-1], num_layers=1)

    def _make_conv_layer(self,
                         in_channels=2048,
                         num_layers=3,
                         num_filters=2048,
                         stride=1):

        layers = []
        for i in range(num_layers):

            downsample = nn.Conv2d(
                in_channels, num_filters, stride=1, kernel_size=1, bias=False)
            layers.append(
                Bottleneck(
                    in_channels, num_filters // 4, downsample=downsample))
            in_channels = num_filters

        return nn.Sequential(*layers)

    def _make_subsample_layer(self, in_channels=96, num_layers=3, stride=2):

        layers = []
        for i in range(num_layers):

            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1))
            in_channels = 2 * in_channels
            layers.append(nn.BatchNorm2d(in_channels, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x3 = self.subsample_3(x_list[1])
        x2 = self.subsample_2(x_list[2])
        x1 = x_list[3]
        xf = self.conv_layers(torch.cat([x3, x2, x1], dim=1))
        xf = xf.mean(dim=(2, 3))
        xf = xf.view(xf.size(0), -1)
        return xf


class PoseHighResolutionNetPyMAFX(PoseHighResolutionNet):
    """HRNet backbone for pymaf-x."""

    def __init__(self,
                 extra,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 pretrained=True,
                 global_mode=False,
                 with_cp=False,
                 init_cfg=None):
        super(PoseHighResolutionNet, self).__init__(init_cfg=init_cfg)
        self.inplanes = 64
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.pretrained = pretrained
        self.with_cp = with_cp

        # stem net
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.bn1 = build_norm_layer(self.norm_cfg, 64)[1]
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.bn2 = build_norm_layer(self.norm_cfg, 64)[1]
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, self.inplanes, 64, 4)

        self.stage2_cfg = extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block = self.blocks_dict[self.stage2_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block = self.blocks_dict[self.stage3_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block = self.blocks_dict[self.stage4_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multiscale_output=True)

        # Classification Head
        self.global_mode = global_mode
        if self.global_mode:
            self.incre_modules, self.downsamp_modules, self.final_layer = \
                self._make_head(pre_stage_channels)

        self.pretrained_layers = extra['pretrained_layers']

    def _make_head(self, pre_stage_channels):
        """make head."""
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(
                head_block, channels, head_channels[i], 1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True))

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                head_channels[3] * head_block.expansion,
                2048,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            build_norm_layer(self.norm_cfg, 2048)[1], nn.ReLU(inplace=True))

        return incre_modules, downsamp_modules, final_layer

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        s_feat = [y_list[-2], y_list[-3], y_list[-4]]

        # Classification Head
        if self.global_mode:
            y = self.incre_modules[0](y_list[0])
            for i in range(len(self.downsamp_modules)):
                y = self.incre_modules[i + 1](y_list[i + 1]) + \
                    self.downsamp_modules[i](y)

            y = self.final_layer(y)

            if torch._C._get_tracing_state():
                xf = y.flatten(start_dim=2).mean(dim=2)
            else:
                xf = F.avg_pool2d(
                    y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        else:
            xf = None

        return s_feat, xf

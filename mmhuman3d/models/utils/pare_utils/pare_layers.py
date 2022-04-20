"""This script is modified from [PARE](https://github.com/
mkocabas/PARE/tree/master/pare/models/layers).

Original license please see docs/additional_licenses.md.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class LocallyConnected2d(nn.Module):

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
    '''

    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
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

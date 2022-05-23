# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

from .hrnet import PoseHighResolutionNet
from .resnet import ResNet, ResNetV1d

BACKBONES = Registry('backbones')

BACKBONES.register_module(name='ResNet', module=ResNet)
BACKBONES.register_module(name='ResNetV1d', module=ResNetV1d)
BACKBONES.register_module(
    name='PoseHighResolutionNet', module=PoseHighResolutionNet)


def build_backbone(cfg):
    """Build backbone."""
    if cfg is None:
        return None
    return BACKBONES.build(cfg)

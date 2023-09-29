# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

from .hrnet import (
    PoseHighResolutionNet,
    PoseHighResolutionNetExpose,
    PoseHighResolutionNetPyMAFX,
)
from .resnet import PoseResNet, ResNet, ResNetV1d
from .vit import ViT

BACKBONES = Registry('backbones')

BACKBONES.register_module(name='ResNet', module=ResNet)
BACKBONES.register_module(name='ResNetV1d', module=ResNetV1d)
BACKBONES.register_module(name='PoseResNet', module=PoseResNet)
BACKBONES.register_module(
    name='PoseHighResolutionNet', module=PoseHighResolutionNet)
BACKBONES.register_module(
    name='PoseHighResolutionNetExpose', module=PoseHighResolutionNetExpose)
BACKBONES.register_module(
    name='PoseHighResolutionNetPyMAFX', module=PoseHighResolutionNetPyMAFX)
BACKBONES.register_module(name='ViT', module=ViT)

def build_backbone(cfg):
    """Build backbone."""
    if cfg is None:
        return None
    return BACKBONES.build(cfg)

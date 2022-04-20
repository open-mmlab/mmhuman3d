# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

from .pose_discriminator import (
    FullPoseDiscriminator,
    PoseDiscriminator,
    ShapeDiscriminator,
    SMPLDiscriminator,
)

DISCRIMINATORS = Registry('discriminators')

DISCRIMINATORS.register_module(
    name='ShapeDiscriminator', module=ShapeDiscriminator)
DISCRIMINATORS.register_module(
    name='PoseDiscriminator', module=PoseDiscriminator)
DISCRIMINATORS.register_module(
    name='FullPoseDiscriminator', module=FullPoseDiscriminator)
DISCRIMINATORS.register_module(
    name='SMPLDiscriminator', module=SMPLDiscriminator)


def build_discriminator(cfg):
    """Build discriminator."""
    if cfg is None:
        return None
    return DISCRIMINATORS.build(cfg)

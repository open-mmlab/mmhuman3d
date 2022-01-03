# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry


def build_from_cfg(cfg, registry, default_args=None):
    if cfg is None:
        return None
    return MMCV_MODELS.build_func(cfg, registry, default_args)


MODELS = Registry('models', parent=MMCV_MODELS, build_func=build_from_cfg)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
ARCHITECTURES = MODELS
BODY_MODELS = MODELS
DISCRIMINATORS = MODELS
REGISTRANTS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_architecture(cfg):
    """Build framework."""
    return ARCHITECTURES.build(cfg)


def build_body_model(cfg):
    """Build body model."""
    return BODY_MODELS.build(cfg)


def build_discriminator(cfg):
    """Build discriminator."""
    return DISCRIMINATORS.build(cfg)


def build_registrant(cfg):
    """Build registrant."""
    return REGISTRANTS.build(cfg)

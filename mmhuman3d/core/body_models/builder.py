# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry


def build_from_cfg(cfg, registry, default_args=None):
    if cfg is None:
        return None
    return MMCV_MODELS.build_func(cfg, registry, default_args)


BODY_MODELS = Registry('body_models', parent=MMCV_MODELS, build_func=build_from_cfg)


def build_body_model(cfg):
    """Build body model."""
    return BODY_MODELS.build(cfg)




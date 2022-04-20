# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

from .smpl import SMPL, GenderedSMPL, HybrIKSMPL
from .smplx import SMPLX

BODY_MODELS = Registry('body_models')

BODY_MODELS.register_module(name=['SMPL', 'smpl'], module=SMPL)
BODY_MODELS.register_module(name='GenderedSMPL', module=GenderedSMPL)
BODY_MODELS.register_module(
    name=['HybrIKSMPL', 'HybrIKsmpl', 'hybriksmpl', 'hybrik', 'hybrIK'],
    module=HybrIKSMPL)
BODY_MODELS.register_module(name=['SMPLX', 'smplx'], module=SMPLX)


def build_body_model(cfg):
    """Build body_models."""
    if cfg is None:
        return None
    return BODY_MODELS.build(cfg)

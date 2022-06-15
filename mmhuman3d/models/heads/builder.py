# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

from .hmr_head import HMRHead
from .hybrik_head import HybrIKHead
from .pare_head import PareHead

HEADS = Registry('heads')

HEADS.register_module(name='HybrIKHead', module=HybrIKHead)
HEADS.register_module(name='HMRHead', module=HMRHead)
HEADS.register_module(name='PareHead', module=PareHead)


def build_head(cfg):
    """Build head."""
    if cfg is None:
        return None
    return HEADS.build(cfg)

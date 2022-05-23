# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

from .smplify import SMPLify
from .smplifyx import SMPLifyX

REGISTRANTS = Registry('registrants')

REGISTRANTS.register_module(name='SMPLify', module=SMPLify)
REGISTRANTS.register_module(name='SMPLifyX', module=SMPLifyX)


def build_registrant(cfg):
    """Build registrant."""
    if cfg is None:
        return None
    return REGISTRANTS.build(cfg)

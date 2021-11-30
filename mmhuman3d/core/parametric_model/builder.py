from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry


def build_from_cfg(cfg, registry, default_args=None):
    if cfg is None:
        return None
    return MMCV_MODELS.build_func(cfg, registry, default_args)


REGISTRANTS = Registry('registrants', build_func=build_from_cfg)


def build_registrant(cfg):
    """Build registrant."""
    return REGISTRANTS.build(cfg)

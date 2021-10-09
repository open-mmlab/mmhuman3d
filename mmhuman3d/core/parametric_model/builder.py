from mmcv.utils import Registry

REGISTRANTS = Registry('registrants')


def build_registrant(cfg):
    """Build registrant."""
    return REGISTRANTS.build(cfg)

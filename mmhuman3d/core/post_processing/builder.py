from mmcv.utils import Registry

POST_PROCESSING = Registry('post_processing')


def build_post_processing(cfg):
    """Build post processing function."""
    return POST_PROCESSING.build(cfg)

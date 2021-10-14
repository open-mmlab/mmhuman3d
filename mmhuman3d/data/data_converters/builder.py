from mmcv.utils import Registry

DATA_CONVERTERS = Registry('data_converters')


def build_data_converter(cfg):
    """Build data converter."""
    return DATA_CONVERTERS.build(cfg)

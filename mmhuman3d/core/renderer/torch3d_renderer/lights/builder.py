from mmcv.utils import Registry

from .lights import AmbientLights, DirectionalLights, PointLights  # noqa:E401

LIGHTS = Registry('lights')
LIGHTS.register_module(
    name=['directional', 'directional_lights', 'DirectionalLights'],
    module=DirectionalLights)
LIGHTS.register_module(
    name=['point', 'point_lights', 'PointLights'], module=PointLights)
LIGHTS.register_module(
    name=['ambient', 'ambient_lights', 'AmbientLights'], module=AmbientLights)


def build_lights(cfg):
    """Build lights."""
    return LIGHTS.build(cfg)

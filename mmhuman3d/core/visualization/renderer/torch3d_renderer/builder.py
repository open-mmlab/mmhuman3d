from mmcv.utils import Registry
from pytorch3d.renderer import (
    DirectionalLights,
    HardFlatShader,
    PointLights,
    SoftGouraudShader,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesAtlas,
    TexturesUV,
    TexturesVertex,
)

from .shader import NoLightShader
from .textures import TexturesClosest

LIGHTS = Registry('lights')
LIGHTS.register_module(
    name=['directional', 'directional_lights', 'DirectionalLights'],
    module=DirectionalLights)
LIGHTS.register_module(
    name=['point', 'point_lights', 'PointLights'], module=PointLights)

SHADER = Registry('shader')

SHADER.register_module(name=['flat', 'HardFlatShader'], module=HardFlatShader)
SHADER.register_module(
    name=['gouraud', 'SoftGouraudShader'], module=SoftGouraudShader)
SHADER.register_module(
    name=['phong', 'SoftPhongShader'], module=SoftPhongShader)
SHADER.register_module(
    name=['silhouette', 'SoftSilhouetteShader'], module=SoftSilhouetteShader)
SHADER.register_module(name=['nolight', 'NoLightShader'], module=NoLightShader)

TEXTURES = Registry('textures')

TEXTURES.register_module(name=['TexturesAtlas', 'atlas'], module=TexturesAtlas)
TEXTURES.register_module(
    name=['TexturesClosest', 'closest'], module=TexturesClosest)
TEXTURES.register_module(name=['TexturesUV', 'uv'], module=TexturesUV)
TEXTURES.register_module(
    name=['TexturesVertex', 'vertex', 'vc'], module=TexturesVertex)


def build_textures(cfg):
    """Build textures."""
    return TEXTURES.build(cfg)


def build_shader(cfg):
    """Build shader."""
    return SHADER.build(cfg)


def build_lights(cfg):
    """Build lights."""
    return LIGHTS.build(cfg)

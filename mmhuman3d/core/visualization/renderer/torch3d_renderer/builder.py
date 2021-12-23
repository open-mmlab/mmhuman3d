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

RENDERER = Registry('renderer')
LIGHTS = Registry('lights')
LIGHTS.register_module(
    name=['directional', 'directional_lights', 'DirectionalLights'],
    module=DirectionalLights)
LIGHTS.register_module(
    name=['point', 'point_lights', 'PointLights'], module=PointLights)

SHADER = Registry('shader')

SHADER.register_module(
    name=[
        'flat', 'hard_flat_shader', 'hard_flat', 'HardFlat', 'HardFlatShader'
    ],
    module=HardFlatShader)
SHADER.register_module(
    name=['gouraud', 'soft_gouraud', 'SoftGouraud', 'SoftGouraudShader'],
    module=SoftGouraudShader)
SHADER.register_module(
    name=['phong', 'soft_phong', 'SoftPhong', 'SoftPhongShader'],
    module=SoftPhongShader)
SHADER.register_module(
    name=[
        'silhouette', 'soft_silhouette', 'SoftSilhouette',
        'SoftSilhouetteShader'
    ],
    module=SoftSilhouetteShader)
SHADER.register_module(
    name=['nolight', 'nolight_shader', 'NoLight', 'NoLightShader'],
    module=NoLightShader)

TEXTURES = Registry('textures')

TEXTURES.register_module(
    name=['TexturesAtlas', 'textures_atlas', 'atlas', 'Atlas'],
    module=TexturesAtlas)
TEXTURES.register_module(
    name=['TexturesClosest', 'textures_closest', 'closest', 'Closest'],
    module=TexturesClosest)
TEXTURES.register_module(
    name=['TexturesUV', 'textures_uv', 'uv'], module=TexturesUV)
TEXTURES.register_module(
    name=['TexturesVertex', 'textures_vertex', 'vertex', 'vc'],
    module=TexturesVertex)


def build_textures(cfg):
    """Build textures."""
    return TEXTURES.build(cfg)


def build_shader(cfg):
    """Build shader."""
    return SHADER.build(cfg)


def build_lights(cfg):
    """Build lights."""
    return LIGHTS.build(cfg)


def build_renderer(cfg):
    """Build renderers."""
    return RENDERER.build(cfg)

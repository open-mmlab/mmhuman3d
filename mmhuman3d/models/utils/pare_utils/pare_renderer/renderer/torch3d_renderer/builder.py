from mmcv.utils import Registry
from pytorch3d.renderer import (
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    SoftGouraudShader,
    SoftPhongShader,
    TexturesAtlas,
    TexturesUV,
    TexturesVertex,
)

from .lights import AmbientLights, DirectionalLights, PointLights
from .shader import (
    DepthShader,
    NoLightShader,
    NormalShader,
    SegmentationShader,
    SilhouetteShader,
)
from .textures import TexturesNearest

RENDERER = Registry('renderer')

LIGHTS = Registry('lights')
LIGHTS.register_module(
    name=['directional', 'directional_lights', 'DirectionalLights'],
    module=DirectionalLights)
LIGHTS.register_module(
    name=['point', 'point_lights', 'PointLights'], module=PointLights)
LIGHTS.register_module(
    name=['ambient', 'ambient_lights', 'AmbientLights'], module=AmbientLights)

SHADER = Registry('shader')
SHADER.register_module(
    name=[
        'flat', 'hard_flat_shader', 'hard_flat', 'HardFlat', 'HardFlatShader'
    ],
    module=HardFlatShader)
SHADER.register_module(
    name=['hard_phong', 'HardPhong', 'HardPhongShader'],
    module=HardPhongShader)
SHADER.register_module(
    name=['hard_gouraud', 'HardGouraud', 'HardGouraudShader'],
    module=HardGouraudShader)
SHADER.register_module(
    name=['soft_gouraud', 'SoftGouraud', 'SoftGouraudShader'],
    module=SoftGouraudShader)
SHADER.register_module(
    name=['soft_phong', 'SoftPhong', 'SoftPhongShader'],
    module=SoftPhongShader)
SHADER.register_module(
    name=['silhouette', 'Silhouette', 'SilhouetteShader'],
    module=SilhouetteShader)
SHADER.register_module(
    name=['nolight', 'nolight_shader', 'NoLight', 'NoLightShader'],
    module=NoLightShader)
SHADER.register_module(
    name=['normal', 'normal_shader', 'Normal', 'NormalShader'],
    module=NormalShader)
SHADER.register_module(
    name=['depth', 'depth_shader', 'Depth', 'DepthShader'], module=DepthShader)
SHADER.register_module(
    name=[
        'segmentation', 'segmentation_shader', 'Segmentation',
        'SegmentationShader'
    ],
    module=SegmentationShader)

TEXTURES = Registry('textures')
TEXTURES.register_module(
    name=['TexturesAtlas', 'textures_atlas', 'atlas', 'Atlas'],
    module=TexturesAtlas)
TEXTURES.register_module(
    name=['TexturesNearest', 'textures_nearest', 'nearest', 'Nearest'],
    module=TexturesNearest)
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

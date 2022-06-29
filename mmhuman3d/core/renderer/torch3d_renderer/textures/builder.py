from mmcv.utils import Registry
from pytorch3d.renderer import TexturesAtlas, TexturesUV, TexturesVertex

from .textures import TexturesNearest

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

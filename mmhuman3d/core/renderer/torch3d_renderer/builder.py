from mmcv.utils import Registry

from .base_renderer import BaseRenderer
from .depth_renderer import DepthRenderer
from .mesh_renderer import MeshRenderer
from .normal_renderer import NormalRenderer
from .pointcloud_renderer import PointCloudRenderer
from .segmentation_renderer import SegmentationRenderer
from .silhouette_renderer import SilhouetteRenderer
from .uv_renderer import UVRenderer

RENDERER = Registry('renderer')
RENDERER.register_module(
    name=['base', 'Base', 'base_renderer', 'BaseRenderer'],
    module=BaseRenderer)
RENDERER.register_module(
    name=['Depth', 'depth', 'depth_renderer', 'DepthRenderer'],
    module=DepthRenderer)
RENDERER.register_module(
    name=['Mesh', 'mesh', 'mesh_renderer', 'MeshRenderer'],
    module=MeshRenderer)
RENDERER.register_module(
    name=['Normal', 'normal', 'normal_renderer', 'NormalRenderer'],
    module=NormalRenderer)
RENDERER.register_module(
    name=[
        'PointCloud', 'pointcloud', 'point_cloud', 'pointcloud_renderer',
        'PointCloudRenderer'
    ],
    module=PointCloudRenderer)
RENDERER.register_module(
    name=[
        'segmentation', 'segmentation_renderer', 'Segmentation',
        'SegmentationRenderer'
    ],
    module=SegmentationRenderer)
RENDERER.register_module(
    name=[
        'silhouette', 'silhouette_renderer', 'Silhouette', 'SilhouetteRenderer'
    ],
    module=SilhouetteRenderer)
RENDERER.register_module(
    name=['uv_renderer', 'uv', 'UV', 'UVRenderer'], module=UVRenderer)


def build_renderer(cfg):
    """Build renderers."""
    return RENDERER.build(cfg)

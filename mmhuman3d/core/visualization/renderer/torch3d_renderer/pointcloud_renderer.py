import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures import Pointclouds

from mmhuman3d.core.cameras import build_cameras
from .render_factory import CAMERA_FACTORY

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class PointCloudRenderer(nn.Module):

    def __init__(self,
                 resolution: Tuple[int, int],
                 device: Union[torch.device, str] = 'cpu',
                 ply_path: Optional[str] = None,
                 return_tensor: bool = False,
                 frames_folder: Optional[str] = None,
                 img_format: str = '%06d.png',
                 projection: Literal['weakperspective', 'fovperspective',
                                     'orthographics', 'perspective',
                                     'fovorthographics'] = 'weakperspective',
                 radius: float = 0.008,
                 **kwargs) -> None:
        self.device = device
        self.resolution = resolution
        self.ply_path = ply_path
        self.return_tensor = return_tensor
        self.img_format = img_format
        self.radius = radius
        self.frames_folder = frames_folder
        self.projection = projection
        self.set_render_params(**kwargs)
        super().__init__()

    def set_render_params(self, **kwargs):
        self.bg_color = torch.tensor(
            kwargs.get('bg_color', [
                1.0,
                1.0,
                1.0,
                0.0,
            ]),
            dtype=torch.float32,
            device=self.device)
        self.raster_settings = PointsRasterizationSettings(
            image_size=self.resolution,
            radius=kwargs.get('radius', self.radius),
            points_per_pixel=kwargs.get('points_per_pixel', 10))

    def forward(
        self,
        pointclouds: Optional[Pointclouds] = None,
        vertices: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        verts_rgba: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
    ):
        if pointclouds is None:
            assert vertices is not None
            if isinstance(vertices, torch.Tensor):
                if vertices.ndim == 2:
                    vertices = vertices[None]
            if isinstance(verts_rgba, torch.Tensor):
                if verts_rgba.ndim == 2:
                    verts_rgba = verts_rgba[None]
            pointclouds = Pointclouds(points=vertices, features=verts_rgba)
        else:
            if vertices is not None or verts_rgba is not None:
                warnings.warn(
                    'Redundant input, will ignore `vertices` and `verts_rgb`.')
        pointclouds = pointclouds.to(self.device)
        cameras = build_cameras(
            dict(type=CAMERA_FACTORY[self.projection], K=K, R=R,
                 T=T)).to(self.device)

        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=cameras, raster_settings=self.raster_settings),
            compositor=AlphaCompositor(background_color=self.bg_color))

        rendered_images = renderer(pointclouds)

        if self.return_tensor:
            return rendered_images
        else:
            return None

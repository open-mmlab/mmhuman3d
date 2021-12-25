import warnings
from typing import Iterable, List, Optional, Tuple, Union

import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
)
from pytorch3d.structures import Meshes, Pointclouds

from mmhuman3d.utils.mesh_utils import mesh_to_pointcloud_vc
from .base_renderer import MeshBaseRenderer
from .builder import RENDERER, build_raster

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@RENDERER.register_module(name=[
    'PointCloud', 'pointcloud', 'point_cloud', 'pointcloud_renderer',
    'PointCloudRenderer'
])
class PointCloudRenderer(MeshBaseRenderer):

    def __init__(self,
                 resolution: Tuple[int, int],
                 device: Union[torch.device, str] = 'cpu',
                 return_tensor: bool = False,
                 out_img_format: str = '%06d.png',
                 projection: Literal['weakperspective', 'fovperspective',
                                     'orthographics', 'perspective',
                                     'fovorthographics'] = 'weakperspective',
                 in_ndc: bool = True,
                 radius: Optional[float] = None,
                 **kwargs) -> None:
        """PointCloud renderer.

        Args:
            resolution (Iterable[int]):
                (width, height) of the rendered images resolution.
            device (Union[torch.device, str], optional):
                You can pass a str or torch.device for cpu or gpu render.
                Defaults to 'cpu'.
            return_tensor (bool, optional):
                Boolean of whether return the rendered tensor.
                Defaults to False.
            out_img_format (str, optional): name format for temp images.
                Defaults to '%06d.png'.
            projection (Literal[, optional): projection type of camera.
                Defaults to 'weakperspective'.
            in_ndc (bool, optional): cameras whether defined in NDC.
                Defaults to True.
            radius (float, optional): radius of points. Defaults to None.

        Returns:
            None
        """
        self.device = device
        self.resolution = resolution
        self.return_tensor = return_tensor
        self.out_img_format = out_img_format
        self.radius = radius
        self.projection = projection
        self.set_render_params(**kwargs)
        self.in_ndc = in_ndc

    def set_render_params(self, **kwargs):
        """Set render params."""
        raster_params = kwargs.get('raster')
        self.bg_color = torch.tensor(
            kwargs.get('bg_color', [
                1.0,
                1.0,
                1.0,
                0.0,
            ]),
            dtype=torch.float32,
            device=self.device)
        self.raster_type = raster_params.pop('type', None)
        self.raster_settings = PointsRasterizationSettings(
            image_size=self.resolution,
            radius=self.radius
            if self.radius is not None else kwargs.get('radius'),
            points_per_pixel=kwargs.get('points_per_pixel', 10))

    def init_renderer(self, cameras):
        renderer = PointsRenderer(
            rasterizer=build_raster(
                dict(
                    type=self.raster_type,
                    cameras=cameras,
                    raster_settings=self.raster_settings)),
            compositor=AlphaCompositor(background_color=self.bg_color))
        return renderer

    def forward(
        self,
        pointclouds: Optional[Pointclouds] = None,
        vertices: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        verts_rgba: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        meshes: Meshes = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        indexs: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> Union[None, torch.Tensor]:
        """Render pointclouds.

        Args:
            pointclouds (Optional[Pointclouds], optional): pytorch3d data
                structure. If not None, `vertices` and `verts_rgba` will
                be ignored.
                Defaults to None.
            vertices (Optional[Union[torch.Tensor, List[torch.Tensor]]],
                optional): coordinate tensor of points. Defaults to None.
            verts_rgba (Optional[Union[torch.Tensor, List[torch.Tensor]]],
                optional): color tensor of points. Defaults to None.
            K (Optional[torch.Tensor], optional): Camera intrinsic matrix.
                Defaults to None.
            R (Optional[torch.Tensor], optional): Camera rotation matrix.
                Defaults to None.
            T (Optional[torch.Tensor], optional): Camera translation matrix.
                Defaults to None.
            images (Optional[torch.Tensor], optional): background images.
                Defaults to None.
            indexs (Optional[Iterable[int]], optional): indexs for the images.
                Defaults to None.

        Returns:
            Union[None, torch.Tensor]: Return tensor or None.
        """
        if pointclouds is None:
            if meshes is not None:
                pointclouds = mesh_to_pointcloud_vc(meshes)
            else:
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
        cameras = self.init_cameras(K=K, R=R, T=T)
        renderer = self.init_renderer(cameras)

        rendered_images = renderer(pointclouds)

        if self.output_path is not None:
            self.write_images(rendered_images, images, indexs)

        if self.return_tensor:
            return rendered_images
        else:
            return None

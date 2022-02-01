import warnings
from typing import Iterable, List, Optional, Tuple, Union

import torch
from pytorch3d.renderer import (AlphaCompositor, PointsRasterizationSettings,
                                PointsRenderer, PointsRasterizer)
import torch.nn as nn
from pytorch3d.structures import Meshes, Pointclouds

from mmhuman3d.core.cameras import NewAttributeCameras
from mmhuman3d.utils.mesh_utils import mesh_to_pointcloud_vc
from .base_renderer import MeshBaseRenderer
from .builder import RENDERER

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
                 output_path: Optional[str] = None,
                 return_type: Optional[List] = None,
                 out_img_format: str = '%06d.png',
                 projection: Literal['weakperspective', 'fovperspective',
                                     'orthographics', 'perspective',
                                     'fovorthographics'] = 'weakperspective',
                 in_ndc: bool = True,
                 radius: Optional[float] = None,
                 **kwargs) -> None:
        """Point cloud renderer.

        Args:
            resolution (Iterable[int]):
                (width, height) of the rendered images resolution.
            device (Union[torch.device, str], optional):
                You can pass a str or torch.device for cpu or gpu render.
                Defaults to 'cpu'.
            output_path (Optional[str], optional):
                Output path of the video or images to be saved.
                Defaults to None.
            return_type (List, optional): the type of tensor to be
                returned. 'tensor' denotes return the determined tensor. E.g.,
                return silhouette tensor of (B, H, W) for SilhouetteRenderer.
                'rgba' denotes the colorful RGBA tensor to be written.
                Will be same for MeshBaseRenderer.
                Will return a pointcloud image for 'tensor' and for 'rgba'.
                Defaults to None.
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
        self.radius = radius
        super().__init__(
            resolution=resolution,
            device=device,
            output_path=output_path,
            obj_path=None,
            return_type=return_type,
            out_img_format=out_img_format,
            projection=projection,
            in_ndc=in_ndc,
            **kwargs)

    def _init_renderer(self, rasterizer, compositor, **kwargs):
        """Set render params."""

        if isinstance(rasterizer, nn.Module):
            rasterizer.raster_settings.image_size = self.resolution
            self.rasterizer = rasterizer
        elif isinstance(rasterizer, dict):
            rasterizer.udpate(image_size=self.resolution)
            if self.radius is not None:
                rasterizer.update(radius=self.radius)
            raster_settings = PointsRasterizationSettings(**rasterizer)
            self.rasterizer = PointsRasterizer(raster_settings=raster_settings)

        compositor = AlphaCompositor(
            **compositor) if isinstance(compositor, dict) else compositor
        self.shader_type = None
        self.renderer = PointsRenderer(
            rasterizer=self.rasterizer, compositor=compositor).to(self.device)

    def forward(
        self,
        pointclouds: Optional[Pointclouds] = None,
        vertices: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        verts_rgba: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        meshes: Meshes = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
        cameras: Optional[NewAttributeCameras] = None,
        images: Optional[torch.Tensor] = None,
        indexes: Optional[Iterable[int]] = None,
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
            indexes (Optional[Iterable[int]], optional): indexes for the
                images.
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
        cameras = self.init_cameras(
            K=K, R=R, T=T) if cameras is None else cameras

        rendered_images = self.renderer(pointclouds, cameras=cameras)

        if self.output_path is not None or 'rgba' in self.return_type:
            rgba = self.tensor2rgba(rendered_images)
            if self.output_path is not None:
                self.write_images(rgba, images, indexes)

        return rendered_images

import os.path as osp
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import mmcv
import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
)
from pytorch3d.structures import Meshes, Pointclouds

from mmhuman3d.utils.mesh_utils import mesh_to_pointcloud_vc
from mmhuman3d.utils.path_utils import check_path_suffix
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
        self.device = device
        self.output_path = output_path
        self.resolution = resolution
        self.return_type = return_type
        self.out_img_format = out_img_format
        self.radius = radius
        self.projection = projection
        self.in_ndc = in_ndc
        if output_path is not None:
            if check_path_suffix(output_path, ['.mp4', '.gif']):
                self.temp_path = osp.join(
                    Path(output_path).parent,
                    Path(output_path).name + '_output_temp')
                mmcv.mkdir_or_exist(self.temp_path)
                print('make dir', self.temp_path)
            else:
                self.temp_path = output_path
        self.set_render_params(**kwargs)
        super(MeshBaseRenderer, self).__init__()

    def set_render_params(self, **kwargs):
        """Set render params."""
        self.raster_type = kwargs.get('raster_type', 'point')
        self.shader_type = None
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
        cameras = self.init_cameras(K=K, R=R, T=T)
        renderer = self.init_renderer(cameras)

        rendered_images = renderer(pointclouds)

        if self.output_path is not None or 'rgba' in self.return_type:
            rgbs, valid_masks = rendered_images[
                ..., :3], (rendered_images[..., 3:] > 0) * 1.0
            if self.output_path is not None:
                self.write_images(rgbs, valid_masks, images, indexes)

        results = {}
        if 'tensor' in self.return_type:
            results.update(tensor=rendered_images)
        if 'rgba' in self.return_type:
            results.update(rgba=rendered_images)
        return results

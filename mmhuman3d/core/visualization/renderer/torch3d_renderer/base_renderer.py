import os.path as osp
import shutil
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import (
    BlendParams,
    Materials,
    MeshRenderer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.utils.ffmpeg_utils import images_to_gif, images_to_video
from mmhuman3d.utils.path_utils import check_path_suffix
from .builder import RENDERER, build_lights, build_raster, build_shader

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@RENDERER.register_module(
    name=['base', 'Base', 'base_renderer', 'MeshBaseRenderer'])
class MeshBaseRenderer(nn.Module):

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
                 **kwargs) -> None:
        """MeshBaseRenderer for neural rendering and visualization.

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
                Defaults to None.
            out_img_format (str, optional): The image format string for
                saving the images.
                Defaults to '%06d.png'.
            projection (Literal[, optional): Projection type of the cameras.
                Defaults to 'weakperspective'.
            in_ndc (bool, optional): Whether defined in NDC.
                Defaults to True.

        **kwargs is used for render setting.
        You can set up your render kwargs like:
            {
                'shader_type': 'flat',
                'texture_type': 'closet',
                'light': {
                    'light_type': 'directional',
                    'direction': [[1.0, 1.0, 1.0]],
                    'ambient_color': [[0.5, 0.5, 0.5]],
                    'diffuse_color': [[0.5, 0.5, 0.5]],
                    'specular_color': [[1.0, 1.0, 1.0]],
                },
                'material': {
                    'ambient_color': [[1, 1, 1]],
                    'diffuse_color': [[0.5, 0.5, 0.5]],
                    'specular_color': [[0.5, 0.5, 0.5]],
                    'shininess': 60.0,
                },
                'raster': {
                    'type': 'mesh',
                    'blur_radius': 0.0,
                    'faces_per_pixel': 1,
                    'cull_to_frustum': True,
                    'cull_backfaces': True,
                },
                'blend': {
                    'background_color': (1.0, 1.0, 1.0)
                },
            }
        You can change any parameter in the suitable range, please check
        configs/render/smpl.py.

        Returns:
            None
        """
        super().__init__()
        self.device = device
        self.output_path = output_path
        self.return_type = return_type
        self.resolution = resolution
        self.projection = projection
        self.temp_path = None
        self.in_ndc = in_ndc
        self.out_img_format = out_img_format
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

    def set_render_params(self, **kwargs):
        """Set render params."""
        material_params = kwargs.get('material', {})
        light_params = kwargs.get('light', {'type': 'directional'})
        self.raster_type = kwargs.get('raster_type', 'mesh')
        raster_params = kwargs.get('raster_setting', {})
        blend_params = kwargs.get('blend', {})
        self.shader_type = kwargs.get('shader_type', 'phong')

        default_resolution = raster_params.pop('resolution', [1024, 1024])
        if self.resolution is None:
            self.resolution = default_resolution

        self.materials = Materials(device=self.device, **material_params)
        self.raster_settings = RasterizationSettings(
            image_size=self.resolution, **raster_params)
        self.lights = build_lights(light_params).to(
            self.device) if light_params is not None else None
        self.blend_params = BlendParams(**blend_params)

    def export(self):
        """Export output video if need."""
        if self.output_path is not None:
            folder = self.temp_path if self.temp_path is not None else\
                 self.output_path
            if Path(self.output_path).suffix == '.mp4':
                images_to_video(
                    input_folder=folder,
                    output_path=self.output_path,
                    img_format=self.out_img_format)
            elif Path(self.output_path).suffix == '.gif':
                images_to_gif(
                    input_folder=folder,
                    output_path=self.output_path,
                    img_format=self.out_img_format)

    def __del__(self):
        """remove_temp_files."""
        if self.output_path is not None:
            if Path(self.output_path).is_file():
                self.remove_temp_frames()

    def remove_temp_frames(self):
        """Remove temp files."""
        if self.temp_path:
            if osp.exists(self.temp_path) and osp.isdir(self.temp_path):
                shutil.rmtree(self.temp_path)

    def init_cameras(self, K, R, T):
        """Build cameras."""
        cameras = build_cameras(
            dict(
                type=self.projection,
                K=K,
                R=R,
                T=T,
                image_size=self.resolution,
                in_ndc=self.in_ndc)).to(self.device)
        return cameras

    def init_renderer(self, cameras, lights):
        """Initial renderer."""
        raster = build_raster(
            dict(
                type=self.raster_type,
                cameras=cameras,
                raster_settings=self.raster_settings))
        renderer = MeshRenderer(
            rasterizer=raster,
            shader=build_shader(
                dict(
                    type=self.shader_type,
                    cameras=cameras,
                    device=self.device,
                    lights=lights,
                    materials=self.materials,
                    blend_params=self.blend_params))
            if self.shader_type != 'silhouette' else build_shader(
                dict(type=self.shader_type, blend_params=self.blend_params)))
        return renderer

    @staticmethod
    def rgb2bgr(rgbs):
        if isinstance(rgbs, torch.Tensor):
            bgrs = torch.cat(
                [rgbs[..., 0, None], rgbs[..., 1, None], rgbs[..., 2, None]],
                -1)
        elif isinstance(rgbs, np.ndarray):
            bgrs = np.concatenate(
                [rgbs[..., 0, None], rgbs[..., 1, None], rgbs[..., 2, None]],
                -1)
        return bgrs

    @staticmethod
    def image_tensor2numpy(image):
        return (image.detach().cpu().numpy() * 255).astype(np.uint8)

    def write_images(self, rgbs, valid_masks, images, indexes):
        """Write output/temp images."""
        rgbs = rgbs.clone()[..., :3] / rgbs[..., :3].max()
        bgrs = torch.cat(
            [rgbs[..., 0, None], rgbs[..., 1, None], rgbs[..., 2, None]], -1)
        if images is not None:
            output_images = bgrs * 255 * valid_masks + (1 -
                                                        valid_masks) * images
            output_images = output_images.detach().cpu().numpy().astype(
                np.uint8)
        else:
            output_images = (bgrs.detach().cpu().numpy() * 255).astype(
                np.uint8)
        for idx, real_idx in enumerate(indexes):
            folder = self.temp_path if self.temp_path is not None else\
                self.output_path
            cv2.imwrite(
                osp.join(folder, self.out_img_format % real_idx),
                output_images[idx])

    def prepare_meshes(self, meshes, vertices, faces):
        if meshes is None:
            assert (vertices is not None) and (faces is not None),\
                'No mesh data input.'
            meshes = Meshes(
                verts=vertices.to(self.device), faces=faces.to(self.device))
        else:
            if (vertices is not None) or (faces is not None):
                warnings.warn('Redundant input, will only use meshes.')
            meshes = meshes.to(self.device)
        return meshes

    def forward(self,
                meshes: Optional[Meshes] = None,
                vertices: Optional[torch.Tensor] = None,
                faces: Optional[torch.Tensor] = None,
                K: Optional[torch.Tensor] = None,
                R: Optional[torch.Tensor] = None,
                T: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                lights: Optional[torch.Tensor] = None,
                indexes: Optional[Iterable[int]] = None,
                **kwargs) -> Union[torch.Tensor, None]:
        """Render Meshes.

        Args:
            meshes (Optional[Meshes], optional): meshes to be rendered.
                Defaults to None.
            vertices (Optional[torch.Tensor], optional): vertices to be
                rendered. Should be passed together with faces.
                Defaults to None.
            faces (Optional[torch.Tensor], optional): faces of the meshes,
                should be passed together with the vertices.
                Defaults to None.
            K (Optional[torch.Tensor], optional): Camera intrinsic matrixs.
                Defaults to None.
            R (Optional[torch.Tensor], optional): Camera rotation matrixs.
                Defaults to None.
            T (Optional[torch.Tensor], optional): Camera tranlastion matrixs.
                Defaults to None.
            images (Optional[torch.Tensor], optional): background images.
                Defaults to None.
            indexes (Optional[Iterable[int]], optional): indexes for images.
                Defaults to None.

        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """
        meshes = self.prepare_meshes(meshes, vertices, faces)
        cameras = self.init_cameras(K=K, R=R, T=T)
        renderer = self.init_renderer(cameras, lights)

        rendered_images = renderer(meshes)

        if self.output_path is not None or 'rgba' in self.return_type:
            valid_masks = (rendered_images[..., 3:] > 0
                           ) * 1.0 if images is not None else None
            if self.output_path is not None:
                self.write_images(rendered_images, valid_masks, images,
                                  indexes)

        results = {}
        if 'tensor' in self.return_type:
            results.update(tensor=rendered_images)
        if 'rgba' in self.return_type:
            results.update(rgba=rendered_images)
        return results

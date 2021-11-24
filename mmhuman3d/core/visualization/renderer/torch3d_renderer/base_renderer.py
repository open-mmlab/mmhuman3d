import os.path as osp
import shutil
import warnings
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import (
    BlendParams,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.utils.ffmpeg_utils import images_to_gif, images_to_video
from mmhuman3d.utils.path_utils import check_path_suffix
from .builder import build_lights, build_shader

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class MeshBaseRenderer(nn.Module):

    def __init__(self,
                 resolution: Tuple[int, int] = (1024, 1024),
                 device: Union[torch.device, str] = 'cpu',
                 obj_path: Optional[str] = None,
                 output_path: Optional[str] = None,
                 return_tensor: bool = False,
                 img_format: str = '%06d.png',
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
            return_tensor (bool, optional):
                Boolean of whether return the rendered tensor.
                Defaults to False.

        **kwargs is used for render setting.
        You can set up your render kwargs like:
            {
                'light': {
                    'light_type': 'directional',
                    'direction': [[1.0, 1.0, 1.0]],
                    'ambient_color': [[0.5, 0.5, 0.5]],
                    'diffuse_color': [[0.5, 0.5, 0.5]],
                    'specular_color': [[1.0, 1.0, 1.0]],
                },
                'camera': {
                    'camera_type': 'weakpespective',
                    'orbit_speed': None,
                },
                'material': {
                    'ambient_color': [[1, 1, 1]],
                    'diffuse_color': [[0.5, 0.5, 0.5]],
                    'specular_color': [[0.5, 0.5, 0.5]],
                    'shininess': 60.0,
                },
                'raster': {
                    'resolution': (256, 256),
                    'blur_radius': 0.0,
                    'faces_per_pixel': 1,
                    'cull_to_frustum': True,
                    'cull_backfaces': True,
                },
                'shader': {
                    'shader_type': 'flat',
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
        self.return_tensor = return_tensor
        self.resolution = resolution
        self.projection = projection
        self.temp_path = None
        self.obj_path = obj_path
        self.in_ndc = in_ndc
        if self.obj_path is not None:
            mmcv.mkdir_or_exist(self.obj_path)
        self.img_format = img_format
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
        material_params = kwargs.get('material')
        light_params = kwargs.get('light')
        shader_params = kwargs.get('shader')
        raster_params = kwargs.get('raster')
        blend_params = kwargs.get('blend')
        assert light_params is not None
        assert shader_params is not None
        assert raster_params is not None
        assert material_params is not None
        assert blend_params is not None
        self.shader_type = shader_params.pop('shader_type', 'phong')

        self.materials = Materials(device=self.device, **material_params)
        default_resolution = raster_params.pop('resolution', None)
        if self.resolution is None:
            self.resolution = default_resolution

        self.raster_settings = RasterizationSettings(
            image_size=self.resolution, **raster_params)
        self.lights = build_lights(light_params).to(self.device)
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
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, raster_settings=self.raster_settings),
            shader=build_shader(
                dict(
                    type=self.shader_type,
                    cameras=cameras,
                    device=self.device,
                    lights=lights,
                    materials=self.materials,
                    blend_params=self.blend_params))
            if self.shader_type != 'silhouette' else build_shader(
                dict(type=self.shader_type)))
        return renderer

    def write_images(self, rgbs, valid_masks, images, indexs):
        """Write output/temp images."""
        if images is not None:
            output_images = rgbs * 255 * valid_masks + (1 -
                                                        valid_masks) * images
            output_images = output_images.detach().cpu().numpy().astype(
                np.uint8)
        else:
            output_images = (rgbs.detach().cpu().numpy() * 255).astype(
                np.uint8)
        for idx, real_idx in enumerate(indexs):
            folder = self.temp_path if self.temp_path is not None else\
                self.output_path
            cv2.imwrite(
                osp.join(folder, self.out_img_format % real_idx),
                output_images[idx])

    def forward(
            self,
            meshes: Optional[Meshes] = None,
            vertices: Optional[torch.Tensor] = None,
            faces: Optional[torch.Tensor] = None,
            K: Optional[torch.Tensor] = None,
            R: Optional[torch.Tensor] = None,
            T: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            indexs: Optional[Iterable[int]] = None
    ) -> Union[torch.Tensor, None]:
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
            indexs (Optional[Iterable[int]], optional): indexs for the images.
                Defaults to None.

        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """
        if meshes is None:
            assert (vertices is not None) and (faces is not None),\
                'No mesh data input.'
            meshes = Meshes(
                verts=vertices.to(self.device), faces=faces.to(self.device))
        else:
            if (vertices is not None) or (faces is not None):
                warnings.warn('Redundant input, will only use meshes.')
            meshes = meshes.to(self.device)
            vertices = meshes.verts_padded()
        cameras = self.init_cameras(K=K, R=R, T=T)
        renderer = self.init_renderer(cameras, self.lights)

        rendered_images = renderer(meshes)
        rgbs = rendered_images.clone()[..., :3] / rendered_images[
            ..., :3].max()
        valid_masks = (rendered_images[..., 3:] > 0) * 1.0

        if self.output_path is not None:
            self.write_images(rgbs, valid_masks, images, indexs)

        if self.return_tensor:
            return rendered_images
        else:
            return None

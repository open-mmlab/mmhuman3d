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
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (DirectionalLights, PointLights, Materials,
                                AmbientLights, RasterizationSettings,
                                MeshRasterizer, BlendParams)
from mmhuman3d.core.cameras import NewAttributeCameras, build_cameras
from mmhuman3d.utils.ffmpeg_utils import images_to_gif, images_to_video
from mmhuman3d.utils.path_utils import check_path_suffix
from .builder import RENDERER, build_lights, build_shader

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@RENDERER.register_module(
    name=['base', 'Base', 'base_renderer', 'MeshBaseRenderer'])
class MeshBaseRenderer(nn.Module):

    def __init__(self,
                 resolution: Tuple[int, int] = None,
                 device: Union[torch.device, str] = 'cpu',
                 output_path: Optional[str] = None,
                 out_img_format: str = '%06d.png',
                 in_ndc: bool = True,
                 projection: Literal['weakperspective', 'fovperspective',
                                     'orthographics', 'perspective',
                                     'fovorthographics'] = 'weakperspective',
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
            out_img_format (str, optional): The image format string for
                saving the images.
                Defaults to '%06d.png'.
            in_ndc (bool, optional): Whether defined in NDC.
                Defaults to True.
            projection (Literal[, optional): Projection type of the cameras.
                Defaults to 'weakperspective'.

        **kwargs is used for render setting.
        You can set up your render kwargs like:
            {
                'shader': {
                    'type': 'soft_phong'
                },
                'lights': {
                        'type': 'directional',
                        'direction': [[10.0, 10.0, 10.0]],
                        'ambient_color': [[0.5, 0.5, 0.5]],
                        'diffuse_color': [[0.5, 0.5, 0.5]],
                        'specular_color': [[0.5, 0.5, 0.5]],
                    },
                'materials': {
                        'ambient_color': [[1, 1, 1]],
                        'diffuse_color': [[0.5, 0.5, 0.5]],
                        'specular_color': [[0.5, 0.5, 0.5]],
                        'shininess': 60.0,
                    },
                'rasterizer': {
                    'bin_size': 0,
                    'blur_radius': 0.0,
                    'faces_per_pixel': 1,
                    'perspective_correct': False,
                    'bin_size': 0,
                },
                'blend_params': {'background_color': (1.0, 1.0, 1.0)},
            },
        You can change any parameter in the suitable range, please check
        configs/render/smpl.py.

        Returns:
            None
        """
        super().__init__()
        self.device = device
        self.output_path = output_path
        self.resolution = resolution
        self.temp_path = None
        self.in_ndc = in_ndc
        self.projection = projection
        self.out_img_format = out_img_format
        self._set_output_path(output_path)

        self._init_renderer(**kwargs)

    def _init_renderer(self,
                       rasterizer: Union[dict, nn.Module] = None,
                       shader: Union[dict, nn.Module] = None,
                       materials: Union[dict, Materials] = None,
                       lights: Union[dict, DirectionalLights, PointLights,
                                     AmbientLights] = None,
                       blend_params: Union[dict, BlendParams] = None,
                       **kwargs):
        """Initial renderer."""
        if isinstance(materials, dict):
            materials = Materials(**materials)
        elif materials is None:
            materials = Materials()
        elif not isinstance(materials, Materials):
            raise TypeError(f'Wrong type of materials: {type(materials)}.')

        if isinstance(lights, dict):
            self.lights = build_lights(lights)
        elif lights is None:
            self.lights = AmbientLights()
        elif not isinstance(
                lights, Union[AmbientLights, PointLights, DirectionalLights]):
            raise TypeError(f'Wrong type of lights: {type(lights)}.')

        if isinstance(blend_params, dict):
            blend_params = BlendParams(**blend_params)
        elif blend_params is None:
            blend_params = BlendParams()
        elif not isinstance(blend_params, BlendParams):
            raise TypeError(
                f'Wrong type of blend_params: {type(blend_params)}.')

        if isinstance(rasterizer, nn.Module):
            if self.resolution is not None:
                rasterizer.raster_settings.image_size = self.resolution
            self.rasterizer = rasterizer
        elif isinstance(rasterizer, dict):
            if self.resolution is not None:
                rasterizer['image_size'] = self.resolution
            raster_settings = RasterizationSettings(**rasterizer)
            self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        else:
            raise TypeError(
                f'Wrong type of rasterizer: {type(self.rasterizer)}.')
        if self.resolution is None:
            self.resolution = tuple(self.rasterizer.raster_settings.image_size)
            self.resolution = (self.resolution, self.resolution) if isinstance(
                self.resolution, int) else self.resolution
        if isinstance(shader, nn.Module):
            self.shader = shader
        elif isinstance(shader, dict):
            shader.update(
                materials=materials,
                lights=self.lights,
                blend_params=blend_params)
            self.shader = build_shader(shader)
        else:
            raise TypeError(f'Wrong type of shader: {type(self.shader)}.')
        self = self.to(self.device)

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if self.rasterizer.cameras is not None:
            self.rasterizer.cameras = self.rasterizer.cameras.to(device)
        if self.shader.cameras is not None:
            self.shader.cameras = self.shader.cameras.to(device)
        self.shader.materials = self.shader.materials.to(device)
        self.shader.lights = self.shader.lights.to(device)
        return self

    def _set_output_path(self, output_path):
        if output_path is not None:
            if check_path_suffix(output_path, ['.mp4', '.gif']):
                self.temp_path = osp.join(
                    Path(output_path).parent,
                    Path(output_path).name + '_output_temp')
                mmcv.mkdir_or_exist(self.temp_path)
                print('make dir', self.temp_path)
            else:
                self.temp_path = output_path

    def _update_resolution(self, **kwargs):
        if 'resolution' in kwargs:
            self.resolution = kwargs.get('resolution')
            self.rasterizer.raster_settings.image_size = self.resolution

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
                self._remove_temp_frames()

    def _remove_temp_frames(self):
        """Remove temp files."""
        if self.temp_path:
            if osp.exists(self.temp_path) and osp.isdir(self.temp_path):
                shutil.rmtree(self.temp_path)

    def _init_cameras(self, K, R, T):
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

    @staticmethod
    def rgb2bgr(rgbs) -> Union[torch.Tensor, np.ndarray]:
        """Convert color channels."""
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
    def _normalize(value,
                   origin_value_range=None,
                   out_value_range=(0, 1),
                   dtype=None,
                   clip=False) -> Union[torch.Tensor, np.ndarray]:
        """Normalize the tensor or array."""
        if origin_value_range is not None:
            value = (value - origin_value_range[0]) / (
                origin_value_range[1] - origin_value_range[0] + 1e-9)

        else:
            value = (value - value.min()) / (value.max() - value.min())
        value = value * (out_value_range[1] -
                         out_value_range[0]) + out_value_range[0]
        if clip:
            value = torch.clip(
                value, min=out_value_range[0], max=out_value_range[1])
        if isinstance(value, torch.Tensor):
            if dtype is not None:
                return value.type(dtype)
            else:
                return value
        elif isinstance(value, np.ndarray):
            if dtype is not None:
                return value.astype(dtype)
            else:
                return value

    def _tensor2array(self, image) -> np.ndarray:
        """Convert image tensor to array."""
        image = image.detach().cpu().numpy()
        image = self._normalize(
            image,
            origin_value_range=(0, 1),
            out_value_range=(0, 255),
            dtype=np.uint8)
        return image

    def _array2tensor(self, image) -> torch.Tensor:
        """Convert image array to tensor."""
        image = torch.Tensor(image)
        image = self._normalize(
            image,
            origin_value_range=(0, 255),
            out_value_range=(0, 1),
            dtype=torch.float32)
        return image

    def _write_images(self, rgba, images, indexes):
        """Write output/temp images."""
        rgbs, valid_masks = rgba[..., :3], rgba[..., 3:]
        rgbs = self._normalize(rgbs, origin_value_range=(0, 1), clip=True)
        bgrs = self.rgb2bgr(rgbs)
        if images is not None:
            image_max = 1.0 if images.max() <= 1.0 else 255
            images = self._normalize(
                images,
                origin_value_range=(0, image_max),
                out_value_range=(0, 1))
            output_images = bgrs * valid_masks + (1 - valid_masks) * images
            output_images = self._tensor2array(output_images)

        else:
            output_images = self._tensor2array(bgrs)
        for idx, real_idx in enumerate(indexes):
            folder = self.temp_path if self.temp_path is not None else\
                self.output_path
            cv2.imwrite(
                osp.join(folder, self.out_img_format % real_idx),
                output_images[idx])

    def _prepare_meshes(self, meshes, vertices, faces):
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
                cameras: Optional[NewAttributeCameras] = None,
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
        self._update_resolution(**kwargs)
        meshes = self._prepare_meshes(meshes, vertices, faces)
        cameras = self._init_cameras(
            K=K, R=R, T=T) if cameras is None else cameras

        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        rendered_images = self.shader(
            fragments=fragments, meshes=meshes, cameras=cameras, lights=lights)

        if self.output_path is not None:
            rgba = self.tensor2rgba(rendered_images)
            self._write_images(rgba, images, indexes)
        return rendered_images

    def tensor2rgba(self, tensor: torch.Tensor):
        valid_masks = (tensor[..., 3:] > 0) * 1.0
        rgbs = tensor[..., :3]
        rgbs = self._normalize(rgbs, out_value_range=[0, 1])
        rgba = torch.cat([rgbs, valid_masks], -1)
        return rgba

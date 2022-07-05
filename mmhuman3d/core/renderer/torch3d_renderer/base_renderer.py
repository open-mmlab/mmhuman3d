import os.path as osp
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import mmcv
import torch
import torch.nn as nn
from pytorch3d.renderer import (
    AmbientLights,
    BlendParams,
    DirectionalLights,
    Materials,
    MeshRasterizer,
    PointLights,
    RasterizationSettings,
)

from mmhuman3d.core.cameras import MMCamerasBase
from mmhuman3d.utils.ffmpeg_utils import images_to_gif, images_to_video
from mmhuman3d.utils.path_utils import check_path_suffix
from .lights import build_lights
from .shader import build_shader
from .utils import normalize, rgb2bgr, tensor2array


class BaseRenderer(nn.Module):

    def __init__(self,
                 resolution: Tuple[int, int] = None,
                 device: Union[torch.device, str] = 'cpu',
                 output_path: Optional[str] = None,
                 out_img_format: str = '%06d.png',
                 **kwargs) -> None:
        """BaseRenderer for differentiable rendering and visualization.

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
        elif isinstance(lights,
                        (AmbientLights, PointLights, DirectionalLights)):
            self.lights = lights
        else:
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
        elif rasterizer is None:
            self.rasterizer = MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.resolution,
                    bin_size=0,
                    blur_radius=0,
                    faces_per_pixel=1,
                    perspective_correct=False))
        else:
            raise TypeError(
                f'Wrong type of rasterizer: {type(self.rasterizer)}.')

        if self.resolution is None:
            self.resolution = self.rasterizer.raster_settings.image_size
        assert self.resolution is not None
        self.resolution = (self.resolution, self.resolution) if isinstance(
            self.resolution, int) else tuple(self.resolution)
        if isinstance(shader, nn.Module):
            self.shader = shader
        elif isinstance(shader, dict):
            shader.update(
                materials=materials,
                lights=self.lights,
                blend_params=blend_params)
            self.shader = build_shader(shader)
        elif shader is None:
            self.shader = build_shader(
                dict(
                    type=self.shader_type,
                    materials=materials,
                    lights=self.lights,
                    blend_params=blend_params))
        else:
            raise TypeError(f'Wrong type of shader: {type(self.shader)}.')
        self = self.to(self.device)

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if getattr(self.rasterizer, 'cameras', None) is not None:
            self.rasterizer.cameras = self.rasterizer.cameras.to(device)

        if getattr(self.shader, 'cameras', None) is not None:
            self.shader.cameras = self.shader.cameras.to(device)
        if getattr(self.shader, 'materials', None) is not None:
            self.shader.materials = self.shader.materials.to(device)
        if getattr(self.shader, 'lights', None) is not None:
            self.shader.lights = self.shader.lights.to(device)
        return self

    def _set_output_path(self, output_path):
        if output_path is not None:
            self.output_path = output_path
            if check_path_suffix(output_path, ['.mp4', '.gif']):
                self.temp_path = osp.join(
                    Path(output_path).parent,
                    Path(output_path).name + '_output_temp')
            else:
                self.temp_path = output_path
            mmcv.mkdir_or_exist(self.temp_path)
            print('Make dir', self.temp_path)

    def _update_resolution(self, cameras, **kwargs):
        if isinstance(cameras, MMCamerasBase):
            self.resolution = (int(cameras.resolution[0][0]),
                               int(cameras.resolution[0][1]))
        if 'resolution' in kwargs:
            self.resolution = kwargs.get('resolution')
        self.rasterizer.raster_settings.image_size = self.resolution

    def export(self):
        """Export output video if need."""
        if self.output_path is not None:
            folder = self.temp_path if self.temp_path is not None else\
                 self.output_path
            if check_path_suffix(self.output_path, ['.mp4']):
                images_to_video(
                    input_folder=folder,
                    output_path=self.output_path,
                    img_format=self.out_img_format)
            elif check_path_suffix(self.output_path, ['.gif']):
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

    def _write_images(self, rgba, backgrounds, indexes):
        """Write output/temp images."""
        if rgba.shape[-1] > 3:
            rgbs, valid_masks = rgba[..., :3], rgba[..., 3:]
        else:
            rgbs = rgba[..., :3]
            valid_masks = torch.ones_like(rgbs[..., :1])
        rgbs = normalize(rgbs, origin_value_range=(0, 1), clip=True)
        bgrs = rgb2bgr(rgbs)
        if backgrounds is not None:
            image_max = 1.0 if backgrounds.max() <= 1.0 else 255
            backgrounds = normalize(
                backgrounds,
                origin_value_range=(0, image_max),
                out_value_range=(0, 1))
            output_images = bgrs * valid_masks + (1 -
                                                  valid_masks) * backgrounds
            output_images = tensor2array(output_images)

        else:
            output_images = tensor2array(bgrs)
        for idx, real_idx in enumerate(indexes):
            folder = self.temp_path if self.temp_path is not None else\
                self.output_path
            cv2.imwrite(
                osp.join(folder, self.out_img_format % real_idx),
                output_images[idx])

    def forward(self):
        """"Should be called by each sub renderer class."""
        raise NotImplementedError()

    def tensor2rgba(self, tensor: torch.Tensor):
        valid_masks = (tensor[..., 3:] > 0) * 1.0
        rgbs = tensor[..., :3]

        rgbs = normalize(
            rgbs, origin_value_range=[0, 1], out_value_range=[0, 1])
        rgba = torch.cat([rgbs, valid_masks], -1)
        return rgba

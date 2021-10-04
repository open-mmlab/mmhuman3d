import os.path as osp
import shutil
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

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
    SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes

from mmhuman3d.utils.ffmpeg_utils import images_to_gif, images_to_video
from mmhuman3d.utils.path_utils import check_path_suffix
from .render_factory import CAMERA_FACTORY, LIGHTS_FACTORY, SHADER_FACTORY


class MeshBaseRenderer(nn.Module):

    def __init__(self,
                 resolution: Tuple[int, int] = (1024, 1024),
                 device: Union[torch.device, str] = 'cpu',
                 obj_path: Optional[str] = None,
                 output_path: Optional[str] = None,
                 return_tensor: bool = False,
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
        self.conv = nn.Conv2d(1, 1, 1)
        self.output_path = output_path
        self.return_tensor = return_tensor
        self.resolution = resolution
        self.temp_path = None
        self.obj_path = obj_path
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
        material_params = kwargs.get('material')
        light_params = kwargs.get('light')
        shader_params = kwargs.get('shader')
        raster_params = kwargs.get('raster')
        camera_params = kwargs.get('camera')
        blend_params = kwargs.get('blend')
        assert light_params is not None
        assert shader_params is not None
        assert raster_params is not None
        assert camera_params is not None
        assert material_params is not None
        assert blend_params is not None
        self.shader = SHADER_FACTORY[shader_params.pop('shader_type', 'phong')]

        self.materials = Materials(device=self.device, **material_params)
        default_resolution = raster_params.pop('resolution', None)
        if self.resolution is None:
            self.resolution = default_resolution

        self.raster_settings = RasterizationSettings(
            image_size=self.resolution, **raster_params)
        light_type = light_params.pop('light_type', 'directional')
        lights = LIGHTS_FACTORY[light_type]
        self.lights = lights(device=self.device, **light_params)

        self.camera_type = camera_params.get('camera_type', 'weakperspective')
        self.camera_register = CAMERA_FACTORY
        self.blend_params = BlendParams(**blend_params)

    def export(self):
        if self.output_path is not None:
            folder = self.temp_path if self.temp_path is not None else\
                 self.output_path
            if Path(self.output_path).suffix == '.mp4':
                images_to_video(
                    input_folder=folder, output_path=self.output_path)
            elif Path(self.output_path).suffix == '.gif':
                images_to_gif(
                    input_folder=folder, output_path=self.output_path)

    def __del__(self):
        if self.output_path is not None:
            if Path(self.output_path).is_file():
                self.removeTempFrames()

    def removeTempFrames(self):
        if self.temp_path:
            if osp.exists(self.temp_path) and osp.isdir(self.temp_path):
                shutil.rmtree(self.temp_path)

    def forward(self,
                meshes: Optional[Meshes] = None,
                vertices: Optional[torch.Tensor] = None,
                faces: Optional[torch.Tensor] = None,
                K: Optional[torch.Tensor] = None,
                R: Optional[torch.Tensor] = None,
                T: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                file_names: Union[List[str], Tuple[str]] = []):
        if meshes is None:
            assert (vertices is not None) and (faces is not None),\
                'No mesh data input.'
            meshes = Meshes(
                verts=vertices.to(self.device), faces=faces.to(self.device))
        else:
            vertices = meshes.verts_padded()
            if (vertices is not None) or (faces is not None):
                warnings.warn('Redundant input, will only use meshes.')

        K = K.to(self.device) if K is not None else None
        R = R.to(self.device) if R is not None else None
        T = T.to(self.device) if T is not None else None
        if K is None:
            self.camera_type = 'fovperspective'
        camera_params = {
            'device': self.device,
            'K': K,
            'R': R,
            'T': T,
        }
        cameras = self.camera_register[self.camera_type](**camera_params)

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, raster_settings=self.raster_settings),
            shader=self.shader(
                device=self.device,
                cameras=cameras,
                lights=self.lights,
                materials=self.materials,
                blend_params=self.blend_params) if
            (self.shader is not SoftSilhouetteShader) else self.shader())

        rendered_images = renderer(meshes)
        rgbs, valid_masks = rendered_images[
            ..., :3], (rendered_images[..., 3:] > 0) * 1.0

        if self.output_path is not None:
            if images is not None:
                output_images = rgbs * 255 * valid_masks + (
                    1 - valid_masks) * images
                output_images = output_images.detach().cpu().numpy().astype(
                    np.uint8)
            else:
                output_images = (rgbs.detach().cpu().numpy() * 255).astype(
                    np.uint8)
            for index in range(output_images.shape[0]):
                folder = self.temp_path if self.temp_path is not None else\
                    self.output_path
                cv2.imwrite(
                    osp.join(folder, file_names[index]), output_images[index])

        if self.return_tensor:
            return rendered_images
        else:
            return None

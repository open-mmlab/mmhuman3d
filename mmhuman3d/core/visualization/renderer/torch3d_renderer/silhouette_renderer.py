import os.path as osp
from pathlib import Path
from typing import Iterable, Optional, Union

import mmcv
import torch
import torch.nn as nn
from pytorch3d.renderer import BlendParams, Materials, RasterizationSettings
from pytorch3d.renderer.lighting import DirectionalLights
from pytorch3d.structures import Meshes

from mmhuman3d.utils.path_utils import check_path_suffix
from .base_renderer import MeshBaseRenderer
from .render_factory import CAMERA_FACTORY
from .shader import NoLightShader


class SilhouetteRenderer(MeshBaseRenderer):

    def __init__(self,
                 resolution: Iterable[int] = [1024, 1024],
                 device: Union[torch.device, str] = 'cpu',
                 obj_path: Optional[str] = None,
                 output_path: Optional[str] = None,
                 return_tensor: bool = True,
                 camera_type: str = 'weakperspetive') -> None:
        """SilhouetteRenderer for neural rendering and visualization.

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
            camera_type (str, optional):
                Projection type of camera.
                Defaults to 'weakperspetive'.

        Returns:
            None
        """
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.conv = nn.Conv2d(1, 1, 1)
        self.output_path = output_path
        self.return_tensor = return_tensor
        self.resolution = resolution
        self.temp_path = None
        self.obj_path = obj_path
        if output_path is not None:
            if check_path_suffix(output_path, ['.mp4', 'gif']):
                self.temp_path = osp.join(
                    Path(output_path).parent,
                    Path(output_path).name + '_output_temp')
                mmcv.mkdir_or_exist(self.temp_path)
                print('make dir', self.temp_path)
        self.camera_type = camera_type
        self.set_render_params()

    def set_render_params(self):
        self.shader = NoLightShader
        self.materials = Materials(device=self.device)
        self.raster_settings = RasterizationSettings(
            image_size=self.resolution)
        self.lights = DirectionalLights(device=self.device)
        self.camera_register = CAMERA_FACTORY[self.camera_type]
        self.blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))

    def forward(self,
                meshes: Meshes,
                K: Optional[torch.Tensor] = None,
                R: Optional[torch.Tensor] = None,
                T: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                file_names: Iterable[str] = []):
        rendered_images = super().forward(meshes, K, R, T, images, file_names)
        alpha = rendered_images[..., 3:]
        return alpha

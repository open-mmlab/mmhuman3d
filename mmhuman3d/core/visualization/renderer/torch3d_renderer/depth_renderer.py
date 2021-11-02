import os.path as osp
import warnings
from typing import Iterable, Optional, Union

import cv2
import numpy as np
import torch
from pytorch3d.renderer import (
    BlendParams,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
)
from pytorch3d.renderer.lighting import DirectionalLights
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
from skimage import exposure

from mmhuman3d.core.cameras import build_cameras
from .base_renderer import MeshBaseRenderer
from .render_factory import CAMERA_FACTORY
from .shader import NoLightShader

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class DepthRenderer(MeshBaseRenderer):

    def __init__(
        self,
        resolution: Iterable[int] = [1024, 1024],
        device: Union[torch.device, str] = 'cpu',
        output_path: Optional[str] = None,
        return_tensor: bool = True,
        img_format: str = '%06d.png',
        projection: Literal['weakperspective', 'fovperspective',
                            'orthographics', 'perspective',
                            'fovorthographics'] = 'weakperspective',
        **kwargs,
    ) -> None:
        super().__init__(
            resolution=resolution,
            device=device,
            output_path=output_path,
            obj_path=None,
            return_tensor=return_tensor,
            img_format=img_format,
            projection=projection,
            **kwargs)

    def set_render_params(self):
        self.shader = NoLightShader
        self.materials = Materials(device=self.device)
        self.raster_settings = RasterizationSettings(
            image_size=self.resolution)
        self.lights = DirectionalLights(device=self.device)
        self.blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))

    def forward(self,
                meshes: Optional[Meshes] = None,
                vertices: Optional[torch.Tensor] = None,
                faces: Optional[torch.Tensor] = None,
                K: Optional[torch.Tensor] = None,
                R: Optional[torch.Tensor] = None,
                T: Optional[torch.Tensor] = None,
                indexs: Optional[Iterable[int]] = None):
        cameras = build_cameras(
            dict(type=CAMERA_FACTORY[self.projection], K=K, R=R,
                 T=T)).to(self.device)
        if meshes is None:
            assert (vertices is not None) and (faces is not None),\
                'No mesh data input.'

            meshes = Meshes(
                verts=vertices.to(self.device),
                faces=faces.to(self.device),
            )
        else:
            vertices = meshes.verts_padded()
            if (vertices is not None) or (faces is not None):
                warnings.warn('Redundant input, will only use meshes.')
        verts_depth = cameras.compute_depth_of_points(vertices)
        verts_depth_rgb = verts_depth.repeat(1, 1, 3)
        norm_scale = torch.max(verts_depth).clone()
        verts_depth_rgb /= norm_scale
        meshes.textures = TexturesVertex(verts_features=verts_depth_rgb)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, raster_settings=self.raster_settings),
            shader=self.shader(
                cameras=cameras,
                device=self.device,
                lights=self.lights,
                materials=self.materials,
                blend_params=self.blend_params))

        rendered_images = renderer(meshes)
        rgbs, _ = rendered_images[
            ..., :3], (rendered_images[..., 3:] > 0) * 1.0
        if self.output_path is not None:
            output_images = rgbs.detach().cpu().numpy()
            p2, p98 = np.percentile(output_images, (2, 98))
            img_rescale = exposure.rescale_intensity(
                output_images, in_range=(p2, p98))

            # Adaptive Equalization
            output_images = (
                exposure.equalize_adapthist(img_rescale, clip_limit=0.03) *
                255).astype(np.uint8)

            for idx, real_idx in enumerate(indexs):
                folder = self.temp_path if self.temp_path is not None else\
                    self.output_path
                cv2.imwrite(
                    osp.join(folder, self.img_format % real_idx),
                    output_images[idx])

        if self.return_tensor:
            return rendered_images * norm_scale
        else:
            return None

from typing import Iterable, Optional, Union

import torch
from pytorch3d.renderer import BlendParams, Materials, RasterizationSettings
from pytorch3d.renderer.lighting import DirectionalLights
from pytorch3d.structures import Meshes

from .base_renderer import MeshBaseRenderer
from .shader import NoLightShader

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class SilhouetteRenderer(MeshBaseRenderer):

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
            projection (str, optional):
                Projection type of camera.
                Defaults to 'weakperspetive'.

        Returns:
            None
        """
        super(SilhouetteRenderer).__init__(
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
        self.lights = DirectionalLights(
            ambient_color=((1.0, 1.0, 1.0)),
            diffuse_color=((.0, .0, .0)),
            specular_color=((.0, .0, .0)),
            device=self.device)
        self.blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))

    def forward(self,
                meshes: Optional[Meshes] = None,
                vertices: Optional[torch.Tensor] = None,
                faces: Optional[torch.Tensor] = None,
                K: Optional[torch.Tensor] = None,
                R: Optional[torch.Tensor] = None,
                T: Optional[torch.Tensor] = None,
                indexs: Iterable[str] = None):
        rendered_images = super().forward(
            meshes=meshes,
            vertices=vertices,
            faces=faces,
            K=K,
            R=R,
            T=T,
            images=None,
            indexs=indexs)
        alpha = rendered_images[..., 3:]
        return alpha

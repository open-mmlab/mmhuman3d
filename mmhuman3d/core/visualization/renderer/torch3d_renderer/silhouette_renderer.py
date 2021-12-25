from typing import Iterable, Optional, Tuple, Union

import torch
from pytorch3d.structures import Meshes

from .base_renderer import MeshBaseRenderer
from .builder import RENDERER

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@RENDERER.register_module(name=[
    'silhouette', 'silhouette_renderer', 'Silhouette', 'SilhouetteRenderer'
])
class SilhouetteRenderer(MeshBaseRenderer):
    """Silhouette renderer."""

    def __init__(
        self,
        resolution: Tuple[int, int],
        device: Union[torch.device, str] = 'cpu',
        output_path: Optional[str] = None,
        return_tensor: bool = True,
        out_img_format: str = '%06d.png',
        projection: Literal['weakperspective', 'fovperspective',
                            'orthographics', 'perspective',
                            'fovorthographics'] = 'weakperspective',
        in_ndc: bool = True,
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
            out_img_format (str, optional): The image format string for
                saving the images.
                Defaults to '%06d.png'.
            projection (str, optional):
                Projection type of camera.
                Defaults to 'weakperspetive'.
            in_ndc (bool, optional): Whether defined in NDC.
                Defaults to True.

        Returns:
            None
        """
        super(SilhouetteRenderer).__init__(
            resolution=resolution,
            device=device,
            output_path=output_path,
            obj_path=None,
            return_tensor=return_tensor,
            out_img_format=out_img_format,
            projection=projection,
            in_ndc=in_ndc,
            **kwargs)

    def set_render_params(self, **kwargs):
        super().set_render_params(**kwargs)
        self.shader_type = 'silhouette'

    def forward(self,
                meshes: Optional[Meshes] = None,
                vertices: Optional[torch.Tensor] = None,
                faces: Optional[torch.Tensor] = None,
                K: Optional[torch.Tensor] = None,
                R: Optional[torch.Tensor] = None,
                T: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                indexs: Iterable[str] = None,
                **kwargs):
        """The params are the same as MeshBaseRenderer."""
        rendered_images = super().forward(
            meshes=meshes,
            vertices=vertices,
            faces=faces,
            K=K,
            R=R,
            T=T,
            images=images,
            indexs=indexs)
        if self.return_tensor:
            return rendered_images[..., 0:1].long()
        else:
            return None

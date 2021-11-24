from typing import Iterable, Optional, Union

import torch
from pytorch3d.structures import Meshes

from .base_renderer import MeshBaseRenderer

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class SilhouetteRenderer(MeshBaseRenderer):
    """Silhouette renderer."""

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
            in_ndc=in_ndc,
            **kwargs)

    def forward(self,
                meshes: Optional[Meshes] = None,
                vertices: Optional[torch.Tensor] = None,
                faces: Optional[torch.Tensor] = None,
                K: Optional[torch.Tensor] = None,
                R: Optional[torch.Tensor] = None,
                T: Optional[torch.Tensor] = None,
                indexs: Iterable[str] = None):
        """The params are the same as MeshBaseRenderer."""
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

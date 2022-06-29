from typing import Iterable, Optional, Tuple, Union

import torch
from pytorch3d.structures import Meshes

from mmhuman3d.core.cameras import MMCamerasBase
from .base_renderer import BaseRenderer
from .utils import normalize


class SilhouetteRenderer(BaseRenderer):
    """Silhouette renderer."""
    shader_type = 'SilhouetteShader'

    def __init__(
        self,
        resolution: Tuple[int, int] = None,
        device: Union[torch.device, str] = 'cpu',
        output_path: Optional[str] = None,
        out_img_format: str = '%06d.png',
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
            out_img_format (str, optional): The image format string for
                saving the images.
                Defaults to '%06d.png'.

        Returns:
            None
        """
        super().__init__(
            resolution=resolution,
            device=device,
            output_path=output_path,
            out_img_format=out_img_format,
            **kwargs)

    def forward(self,
                meshes: Optional[Meshes] = None,
                cameras: Optional[MMCamerasBase] = None,
                images: Optional[torch.Tensor] = None,
                indexes: Iterable[str] = None,
                backgrounds: Optional[torch.Tensor] = None,
                **kwargs):
        """Render silhouette map.

        Args:
            meshes (Optional[Meshes], optional): meshes to be rendered.
                Require the textures type is `TexturesClosest`.
                The color indicates the class index of the triangle.
                Defaults to None.
            cameras (Optional[MMCamerasBase], optional): cameras for render.
                Defaults to None.
            indexes (Optional[Iterable[int]], optional): indexes for images.
                Defaults to None.
            backgrounds (Optional[torch.Tensor], optional): background images.
                Defaults to None.

        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """
        meshes = meshes.to(self.device)
        self._update_resolution(cameras, **kwargs)
        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        silhouette_map = self.shader(
            fragments=fragments, meshes=meshes, cameras=cameras)

        if self.output_path is not None:
            rgba = self.tensor2rgba(silhouette_map)
            self._write_images(rgba, backgrounds, indexes)

        return silhouette_map

    def tensor2rgba(self, tensor: torch.Tensor):
        silhouette = tensor[..., 3:]
        rgbs = silhouette.repeat(1, 1, 1, 3)
        valid_masks = (silhouette > 0) * 1.0
        rgbs = normalize(rgbs, out_value_range=(0, 1))
        return torch.cat([rgbs, valid_masks], -1)

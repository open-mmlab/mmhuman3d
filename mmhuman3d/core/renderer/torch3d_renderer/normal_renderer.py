from typing import Iterable, Optional, Union

import torch
from pytorch3d.structures import Meshes

from mmhuman3d.core.cameras import MMCamerasBase
from .base_renderer import BaseRenderer
from .utils import normalize


class NormalRenderer(BaseRenderer):
    """Render normal map with the help of camera system."""
    shader_type = 'NormalShader'

    def __init__(
        self,
        resolution: Iterable[int] = None,
        device: Union[torch.device, str] = 'cpu',
        output_path: Optional[str] = None,
        out_img_format: str = '%06d.png',
        **kwargs,
    ) -> None:
        """Renderer for normal map of meshes.

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
            obj_path=None,
            out_img_format=out_img_format,
            **kwargs)

    def forward(self,
                meshes: Optional[Meshes] = None,
                cameras: Optional[MMCamerasBase] = None,
                indexes: Optional[Iterable[int]] = None,
                backgrounds: Optional[torch.Tensor] = None,
                **kwargs):
        """Render Meshes.

        Args:
            meshes (Optional[Meshes], optional): meshes to be rendered.
                Defaults to None.
            cameras (Optional[MMCamerasBase], optional): cameras for render.
                Defaults to None.
            indexes (Optional[Iterable[int]], optional): indexes for the
                images.
                Defaults to None.
            backgrounds (Optional[torch.Tensor], optional): background images.
                Defaults to None.

        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """

        meshes = meshes.to(self.device)
        self._update_resolution(cameras, **kwargs)
        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        normal_map = self.shader(
            fragments=fragments, meshes=meshes, cameras=cameras)

        if self.output_path is not None:
            rgba = self.tensor2rgba(normal_map)
            self._write_images(rgba, backgrounds, indexes)

        return normal_map

    def tensor2rgba(self, tensor: torch.Tensor):
        rgbs, valid_masks = tensor[..., :3], (tensor[..., 3:] > 0) * 1.0
        rgbs = normalize(
            rgbs, origin_value_range=(-1, 1), out_value_range=(0, 1))
        return torch.cat([rgbs, valid_masks], -1)

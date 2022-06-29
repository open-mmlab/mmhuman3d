from typing import Iterable, Optional, Tuple, Union

import torch
from pytorch3d.structures import Meshes

from mmhuman3d.core.cameras import MMCamerasBase
from .base_renderer import BaseRenderer
from .lights import MMLights


class MeshRenderer(BaseRenderer):
    """Render RGBA image with the help of camera system."""
    shader_type = 'SoftPhongShader'

    def __init__(
        self,
        resolution: Tuple[int, int] = None,
        device: Union[torch.device, str] = 'cpu',
        output_path: Optional[str] = None,
        out_img_format: str = '%06d.png',
        **kwargs,
    ) -> None:
        """Renderer for RGBA image of meshes.

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
        """
        super().__init__(
            resolution=resolution,
            device=device,
            output_path=output_path,
            out_img_format=out_img_format,
            **kwargs)

    def forward(self,
                meshes: Meshes,
                cameras: Optional[MMCamerasBase] = None,
                lights: Optional[MMLights] = None,
                indexes: Optional[Iterable[int]] = None,
                backgrounds: Optional[torch.Tensor] = None,
                **kwargs) -> Union[torch.Tensor, None]:
        """Render Meshes.

        Args:
            meshes (Meshes): meshes to be rendered.
            cameras (Optional[MMCamerasBase], optional): cameras for render.
                Defaults to None.
            lights (Optional[MMLights], optional): lights for render.
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

        rendered_images = self.shader(
            fragments=fragments,
            meshes=meshes,
            cameras=cameras,
            lights=self.lights if lights is None else lights)

        if self.output_path is not None:
            rgba = self.tensor2rgba(rendered_images)
            self._write_images(rgba, backgrounds, indexes)
        return rendered_images

from typing import Iterable, Optional, Union

import torch
from pytorch3d.structures import Meshes

from mmhuman3d.core.cameras import MMCamerasBase
from .base_renderer import BaseRenderer
from .builder import RENDERER


@RENDERER.register_module(
    name=['Normal', 'normal', 'normal_renderer', 'NormalRenderer'])
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
                vertices: Optional[torch.Tensor] = None,
                faces: Optional[torch.Tensor] = None,
                cameras: Optional[MMCamerasBase] = None,
                images: Optional[torch.Tensor] = None,
                indexes: Optional[Iterable[int]] = None,
                **kwargs):
        """Render Meshes.

        Args:
            meshes (Optional[Meshes], optional): meshes to be rendered.
                Defaults to None.
            vertices (Optional[torch.Tensor], optional): vertices to be
                rendered. Should be passed together with faces.
                Defaults to None.
            faces (Optional[torch.Tensor], optional): faces of the meshes,
                should be passed together with the vertices.
                Defaults to None.

            images (Optional[torch.Tensor], optional): background images.
                Defaults to None.
            indexes (Optional[Iterable[int]], optional): indexes for the
                images.
                Defaults to None.

        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """

        meshes = self._prepare_meshes(
            meshes, vertices, faces, device=self.device)
        self._update_resolution(cameras, **kwargs)
        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        normal_map = self.shader(
            fragments=fragments, meshes=meshes, cameras=cameras)

        if self.output_path is not None:
            rgba = self.tensor2rgba(normal_map)
            self.write_images(rgba, images, indexes)

        return normal_map

    def tensor2rgba(self, tensor: torch.Tensor):
        rgbs, valid_masks = tensor[..., :3], (tensor[..., 3:] > 0) * 1.0
        rgbs = self._normalize(
            rgbs, origin_value_range=(-1, 1), out_value_range=(0, 1))
        return torch.cat([rgbs, valid_masks], -1)

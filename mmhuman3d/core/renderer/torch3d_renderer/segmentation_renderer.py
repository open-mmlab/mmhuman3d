from typing import Iterable, Optional, Tuple, Union

import torch
from pytorch3d.structures import Meshes

from mmhuman3d.core.cameras import MMCamerasBase
from mmhuman3d.utils.demo_utils import get_different_colors
from .base_renderer import BaseRenderer
from .utils import normalize


class SegmentationRenderer(BaseRenderer):
    """Render segmentation map into a segmentation index tensor."""
    shader_type = 'SegmentationShader'

    def __init__(self,
                 resolution: Tuple[int, int] = None,
                 device: Union[torch.device, str] = 'cpu',
                 output_path: Optional[str] = None,
                 out_img_format: str = '%06d.png',
                 num_class: int = 1,
                 **kwargs) -> None:
        """Render vertex-color mesh into a segmentation map of a (B, H, W)
        tensor. For visualization, the output rgba image will be (B, H, W, 4),
        and the color palette comes from `get_different_colors`. The
        segmentation map is a tensor each pixel saves the classification index.
        Please make sure you have allocate each pixel a correct classification
        index by defining a textures of vertex color.

        [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.
        CrossEntropyLoss.html)

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
            num_class (int, optional): number of segmentation parts.
                Defaults to 1.

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
        self.num_class = num_class

    def forward(self,
                meshes: Meshes,
                cameras: Optional[MMCamerasBase] = None,
                indexes: Optional[Iterable[int]] = None,
                backgrounds: Optional[torch.Tensor] = None,
                **kwargs):
        """Render segmentation map.

        Args:
            meshes (Meshes): meshes to be rendered.
                Require the textures type is `TexturesClosest`.
                The color indicates the class index of the triangle.
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
        segmentation_map = self.shader(
            fragments=fragments, meshes=meshes, cameras=cameras)

        if self.output_path is not None:
            rgba = self.tensor2rgba(segmentation_map)
            if self.output_path is not None:
                self._write_images(rgba, backgrounds, indexes)

        return segmentation_map

    def tensor2rgba(self, tensor: torch.Tensor):
        valid_masks = (tensor[..., :] > 0) * 1.0
        color = torch.Tensor(get_different_colors(self.num_class))
        color = torch.cat([torch.zeros(1, 3), color]).to(self.device)
        B, H, W, _ = tensor.shape
        rgbs = color[tensor.view(-1)].view(B, H, W, 3) * valid_masks
        rgbs = normalize(
            rgbs.float(), origin_value_range=(0, 255), out_value_range=(0, 1))
        rgba = torch.cat([rgbs, valid_masks], -1)
        return rgba

from typing import Iterable, Optional, Tuple, Union

import torch
from pytorch3d.structures import Meshes

from mmhuman3d.core.cameras import MMCamerasBase
from mmhuman3d.utils import get_different_colors
from .base_renderer import BaseRenderer
from .builder import RENDERER

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@RENDERER.register_module(name=[
    'segmentation', 'segmentation_renderer', 'Segmentation',
    'SegmentationRenderer'
])
class SegmentationRenderer(BaseRenderer):
    """Render segmentation map into a segmentation index tensor."""
    shader_type = 'SegmentationShader'

    def __init__(self,
                 resolution: Tuple[int, int] = None,
                 device: Union[torch.device, str] = 'cpu',
                 output_path: Optional[str] = None,
                 out_img_format: str = '%06d.png',
                 projection: Literal['weakperspective', 'fovperspective',
                                     'orthographics', 'perspective',
                                     'fovorthographics'] = 'weakperspective',
                 in_ndc: bool = True,
                 num_class: int = 1,
                 differentiable: bool = True,
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
            projection (Literal[, optional): Projection type of the cameras.
                Defaults to 'weakperspective'.
            in_ndc (bool, optional): Whether defined in NDC.
                Defaults to True.
            num_class (int, optional): number of segmentation parts.
                Defaults to 1.
            differentiable (bool, optional): Some renderer need smplified
                parameters if do not need differentiable.
                Defaults to True.

        Returns:
            None
        """
        super().__init__(
            resolution=resolution,
            device=device,
            output_path=output_path,
            obj_path=None,
            out_img_format=out_img_format,
            projection=projection,
            in_ndc=in_ndc,
            differentiable=differentiable,
            **kwargs)
        self.num_class = num_class

    def forward(self,
                meshes: Optional[Meshes] = None,
                vertices: Optional[torch.Tensor] = None,
                faces: Optional[torch.Tensor] = None,
                K: Optional[torch.Tensor] = None,
                R: Optional[torch.Tensor] = None,
                T: Optional[torch.Tensor] = None,
                cameras: Optional[MMCamerasBase] = None,
                images: Optional[torch.Tensor] = None,
                indexes: Optional[Iterable[int]] = None,
                **kwargs):
        """Render segmentation map.

        Args:
            meshes (Optional[Meshes], optional): meshes to be rendered.
                Require the textures type is `TexturesClosest`.
                The color indicates the class index of the triangle.
                Defaults to None.
            vertices (Optional[torch.Tensor], optional): vertices to be
                rendered. Should be passed together with faces.
                Defaults to None.
            faces (Optional[torch.Tensor], optional): faces of the meshes,
                should be passed together with the vertices.
                Defaults to None.
            K (Optional[torch.Tensor], optional): Camera intrinsic matrixs.
                Defaults to None.
            R (Optional[torch.Tensor], optional): Camera rotation matrixs.
                Defaults to None.
            T (Optional[torch.Tensor], optional): Camera tranlastion matrixs.
                Defaults to None.
            images (Optional[torch.Tensor], optional): background images.
                Defaults to None.
            indexes (Optional[Iterable[int]], optional): indexes for images.
                Defaults to None.

        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """

        cameras = self._init_cameras(
            K=K, R=R, T=T) if cameras is None else cameras
        meshes = self._prepare_meshes(meshes, vertices, faces)
        self._update_resolution(cameras, **kwargs)
        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        segmentation_map = self.shader(
            fragments=fragments, meshes=meshes, cameras=cameras)

        if self.output_path is not None:
            rgba = self.tensor2rgba(segmentation_map)
            if self.output_path is not None:
                self.write_images(rgba, images, indexes)

        return segmentation_map

    def tensor2rgba(self, tensor: torch.Tensor):
        valid_masks = (tensor[..., :] > 0) * 1.0
        color = torch.Tensor(get_different_colors(self.num_class))
        color = torch.cat([torch.zeros(1, 3), color]).to(self.device)
        B, H, W, _ = tensor.shape
        rgbs = color[tensor.view(-1)].view(B, H, W, 3) * valid_masks
        rgbs = self._normalize(
            rgbs.float(), origin_value_range=(0, 255), out_value_range=(0, 1))
        rgba = torch.cat([rgbs, valid_masks], -1)
        return rgba

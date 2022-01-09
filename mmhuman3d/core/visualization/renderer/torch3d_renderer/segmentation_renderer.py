from typing import Iterable, List, Optional, Tuple, Union

import torch
from pytorch3d.structures import Meshes

from mmhuman3d.utils import get_different_colors
from .base_renderer import MeshBaseRenderer
from .builder import RENDERER

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@RENDERER.register_module(name=[
    'seg', 'segmentation', 'segmentation_renderer', 'Segmentation',
    'SegmentationRenderer'
])
class SegmentationRenderer(MeshBaseRenderer):
    """Render segmentation map into a segmentation index tensor."""

    def __init__(self,
                 resolution: Tuple[int, int],
                 device: Union[torch.device, str] = 'cpu',
                 output_path: Optional[str] = None,
                 return_type: Optional[List] = None,
                 out_img_format: str = '%06d.png',
                 projection: Literal['weakperspective', 'fovperspective',
                                     'orthographics', 'perspective',
                                     'fovorthographics'] = 'weakperspective',
                 in_ndc: bool = True,
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
            return_type (Optional[Literal[, optional): the type of tensor to be
                returned. 'tensor' denotes return the determined tensor. E.g.,
                return silhouette tensor of (B, H, W) for SilhouetteRenderer.
                'rgba' denotes the colorful RGBA tensor to be written.
                Will return a colorful segmentation image for 'rgba' and a
                segmentation map for 'tensor' (could be used as segmnentation
                GT).
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

        Returns:
            None
        """
        super().__init__(
            resolution=resolution,
            device=device,
            output_path=output_path,
            obj_path=None,
            return_type=return_type,
            out_img_format=out_img_format,
            projection=projection,
            in_ndc=in_ndc,
            **kwargs)

    def set_render_params(self, **kwargs):
        super().set_render_params(**kwargs)
        self.shader_type = 'nolight'
        self.num_class = kwargs.get('num_class', 1)

    def forward(self,
                meshes: Optional[Meshes] = None,
                K: Optional[torch.Tensor] = None,
                R: Optional[torch.Tensor] = None,
                T: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                indexes: Optional[Iterable[int]] = None,
                **kwargs):
        """Render segmentation map.

        Args:
            meshes (Optional[Meshes], optional): meshes to be rendered.
                Require the textures type is `TexturesClosest`.
                The color indicates the class index of the triangle.
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
        # It is recommended that you use `TexturesClosest` to exclude
        # inappropriate interpolation among the faces, to make sure the
        # segmentation map is sharp.
        cameras = self.init_cameras(K=K, R=R, T=T)
        renderer = self.init_renderer(cameras, None)

        rendered_images = renderer(meshes)

        segmentation_map = rendered_images[..., 0].long()

        if self.output_path is not None or 'rgba' in self.return_type:
            valid_masks = (rendered_images[..., 3:] > 0) * 1.0
            color = torch.Tensor(get_different_colors(self.num_class))
            color = torch.cat([torch.zeros(1, 3), color]).to(self.device)
            rgbs = color[segmentation_map] * valid_masks
            if self.output_path is not None:
                self.write_images(rgbs, valid_masks, images, indexes)

        results = {}
        if 'tensor' in self.return_type:
            results.update(tensor=segmentation_map)
        if 'rgba' in self.return_type:
            results.update(rgba=torch.cat([rgbs, valid_masks], -1))
        return results

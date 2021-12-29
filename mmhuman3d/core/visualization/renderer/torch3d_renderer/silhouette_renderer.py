from typing import Iterable, List, Optional, Tuple, Union

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
        return_type: Optional[List] = None,
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
            return_type (Optional[Literal[, optional): the type of tensor to be
                returned. 'tensor' denotes return the determined tensor. E.g.,
                return silhouette tensor of (B, H, W) for SilhouetteRenderer.
                'rgba' denotes the colorful RGBA tensor to be written.
                Will return a 3 channel mask for 'tensor' and 4 channel for
                'rgba'.
                Defaults to None.
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
        self.shader_type = 'silhouette'

    def forward(self,
                meshes: Optional[Meshes] = None,
                vertices: Optional[torch.Tensor] = None,
                faces: Optional[torch.Tensor] = None,
                K: Optional[torch.Tensor] = None,
                R: Optional[torch.Tensor] = None,
                T: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                indexes: Iterable[str] = None,
                **kwargs):
        """The params are the same as MeshBaseRenderer."""
        meshes = self.prepare_meshes(meshes, vertices, faces)
        cameras = self.init_cameras(K=K, R=R, T=T)
        renderer = self.init_renderer(cameras, None)

        rendered_images = renderer(meshes)
        silhouette_map = rendered_images[..., 3:]
        valid_masks = (silhouette_map > 0) * 1.0
        if self.output_path is not None or 'rgba' in self.return_type:
            rgbs = silhouette_map.repeat(1, 1, 1, 3)
            if self.output_path is not None:
                self.write_images(rgbs, valid_masks, images, indexes)

        results = {}
        if 'tensor' in self.return_type:
            results.update(tensor=silhouette_map)
        if 'rgba' in self.return_type:
            results.update(rgba=valid_masks.repeat(1, 1, 1, 4))

        return results

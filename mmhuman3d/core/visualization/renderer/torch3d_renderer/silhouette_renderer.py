from typing import Iterable, Optional, Tuple, Union

import torch
from pytorch3d.structures import Meshes

from mmhuman3d.core.cameras import MMCamerasBase
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
    shader_type = 'SilhouetteShader'

    def __init__(
        self,
        resolution: Tuple[int, int] = None,
        device: Union[torch.device, str] = 'cpu',
        output_path: Optional[str] = None,
        out_img_format: str = '%06d.png',
        projection: Literal['weakperspective', 'fovperspective',
                            'orthographics', 'perspective',
                            'fovorthographics'] = 'weakperspective',
        in_ndc: bool = True,
        differentiable: bool = True,
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
            projection (str, optional):
                Projection type of camera.
                Defaults to 'weakperspetive'.
            in_ndc (bool, optional): Whether defined in NDC.
                Defaults to True.
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
            out_img_format=out_img_format,
            projection=projection,
            in_ndc=in_ndc,
            differentiable=differentiable,
            **kwargs)

    def forward(self,
                meshes: Optional[Meshes] = None,
                vertices: Optional[torch.Tensor] = None,
                faces: Optional[torch.Tensor] = None,
                K: Optional[torch.Tensor] = None,
                R: Optional[torch.Tensor] = None,
                T: Optional[torch.Tensor] = None,
                cameras: Optional[MMCamerasBase] = None,
                images: Optional[torch.Tensor] = None,
                indexes: Iterable[str] = None,
                **kwargs):
        """The params are the same as MeshBaseRenderer."""
        meshes = self._prepare_meshes(meshes, vertices, faces)
        cameras = self._init_cameras(
            K=K, R=R, T=T) if cameras is None else cameras
        self._update_resolution(cameras, **kwargs)
        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        silhouette_map = self.shader(
            fragments=fragments, meshes=meshes, cameras=cameras)

        if self.output_path is not None:
            rgba = self.tensor2rgba(silhouette_map)
            self.write_images(rgba, images, indexes)

        return silhouette_map

    def tensor2rgba(self, tensor: torch.Tensor):
        silhouette = tensor[..., 3:]
        rgbs = silhouette.repeat(1, 1, 1, 3)
        valid_masks = (silhouette > 0) * 1.0
        rgbs = self._normalize(rgbs, out_value_range=(0, 1))
        return torch.cat([rgbs, valid_masks], -1)

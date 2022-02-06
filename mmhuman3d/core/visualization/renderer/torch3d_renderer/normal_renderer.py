from typing import Iterable, List, Optional, Union

import torch
from pytorch3d.structures import Meshes

from mmhuman3d.core.cameras import NewAttributeCameras
from .base_renderer import MeshBaseRenderer
from .builder import RENDERER

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@RENDERER.register_module(
    name=['Normal', 'normal', 'normal_renderer', 'NormalRenderer'])
class NormalRenderer(MeshBaseRenderer):
    """Render depth map with the help of camera system."""

    def __init__(
        self,
        resolution: Iterable[int] = None,
        device: Union[torch.device, str] = 'cpu',
        output_path: Optional[str] = None,
        out_img_format: str = '%06d.png',
        projection: Literal['weakperspective', 'fovperspective',
                            'orthographics', 'perspective',
                            'fovorthographics'] = 'weakperspective',
        in_ndc: bool = True,
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
            projection (Literal[, optional): Projection type of the cameras.
                Defaults to 'weakperspective'.
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
            out_img_format=out_img_format,
            projection=projection,
            in_ndc=in_ndc,
            **kwargs)

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if self.rasterizer.cameras is not None:
            self.rasterizer.cameras = self.rasterizer.cameras.to(device)
        return self

    def forward(self,
                meshes: Optional[Meshes] = None,
                vertices: Optional[torch.Tensor] = None,
                faces: Optional[torch.Tensor] = None,
                K: Optional[torch.Tensor] = None,
                R: Optional[torch.Tensor] = None,
                T: Optional[torch.Tensor] = None,
                cameras: Optional[NewAttributeCameras] = None,
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
            K (Optional[torch.Tensor], optional): Camera intrinsic matrixs.
                Defaults to None.
            R (Optional[torch.Tensor], optional): Camera rotation matrixs.
                Defaults to None.
            T (Optional[torch.Tensor], optional): Camera tranlastion matrixs.
                Defaults to None.
            images (Optional[torch.Tensor], optional): background images.
                Defaults to None.
            indexes (Optional[Iterable[int]], optional): indexes for the
                images.
                Defaults to None.

        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """
        self._update_resolution(**kwargs)
        meshes = self._prepare_meshes(meshes, vertices, faces)

        cameras = self._init_cameras(
            K=K, R=R, T=T) if cameras is None else cameras

        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        normal_map = self.shader(
            fragments=fragments, meshes=meshes, cameras=cameras)

        if self.output_path is not None:
            rgba = self.tensor2rgba(normal_map)
            self.write_images(rgba, images, indexes)

        return normal_map

    def tensor2rgba(self, tensor: torch.Tensor):
        rgbs, valid_masks = tensor[..., :3], (tensor[..., 3:] > 0) * 1.0
        rgbs = (rgbs + 1) / 2
        return torch.cat([rgbs, valid_masks], -1)

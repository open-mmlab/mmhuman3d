from typing import Iterable, Optional, Tuple, Union

import torch
from pytorch3d.structures import Meshes

from mmhuman3d.core.cameras import MMCamerasBase
from .base_renderer import BaseRenderer
from .builder import RENDERER, build_shader

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@RENDERER.register_module(
    name=['Depth', 'depth', 'depth_renderer', 'DepthRenderer'])
class DepthRenderer(BaseRenderer):
    """Render depth map with the help of camera system."""
    shader_type = 'DepthShader'

    def __init__(
        self,
        resolution: Tuple[int, int] = None,
        device: Union[torch.device, str] = 'cpu',
        output_path: Optional[str] = None,
        out_img_format: str = '%06d.png',
        in_ndc: bool = True,
        projection: Literal['weakperspective', 'fovperspective',
                            'orthographics', 'perspective',
                            'fovorthographics'] = 'weakperspective',
        depth_max: Union[int, float, torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Renderer for depth map of meshes.

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
            in_ndc (bool, optional): Whether defined in NDC.
                Defaults to True.
            projection (Literal[, optional): Projection type of the cameras.
                Defaults to 'weakperspective'.

            depth_max (Union[int, float, torch.Tensor], optional):
                The max value for normalize depth range. Defaults to None.

        Returns:
            None
        """
        super().__init__(
            resolution=resolution,
            device=device,
            output_path=output_path,
            out_img_format=out_img_format,
            in_ndc=in_ndc,
            projection=projection,
            **kwargs)
        self.depth_max = depth_max

    def _init_renderer(self,
                       rasterizer=None,
                       shader=None,
                       materials=None,
                       lights=None,
                       blend_params=None,
                       **kwargs):
        shader = build_shader(dict(
            type='DepthShader')) if shader is None else shader
        return super()._init_renderer(rasterizer, shader, materials, lights,
                                      blend_params, **kwargs)

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
        """Render depth map.

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

        cameras = self._init_cameras(
            K=K, R=R, T=T) if cameras is None else cameras
        meshes = self._prepare_meshes(meshes, vertices, faces)
        self._update_resolution(cameras, **kwargs)
        vertices = meshes.verts_padded()

        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        depth_map = self.shader(
            fragments=fragments, meshes=meshes, cameras=cameras)

        if self.output_path is not None:
            rgba = self.tensor2rgba(depth_map)
            if self.output_path is not None:
                self.write_images(rgba, images, indexes)

        return depth_map

    def tensor2rgba(self, tensor: torch.Tensor):
        rgbs, valid_masks = tensor.repeat(1, 1, 1, 3), (tensor > 0) * 1.0
        depth_max = self.depth_max if self.depth_max is not None else rgbs.max(
        )
        rgbs = self._normalize(
            rgbs, origin_value_range=(0, depth_max), out_value_range=(0, 1))
        return torch.cat([rgbs, valid_masks], -1)

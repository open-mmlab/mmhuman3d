import os.path as osp
from typing import Iterable, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
from skimage import exposure

from .base_renderer import MeshBaseRenderer
from .builder import RENDERER

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@RENDERER.register_module(
    name=['Depth', 'depth', 'depth_renderer', 'DepthRenderer'])
class DepthRenderer(MeshBaseRenderer):
    """Render depth map with the help of camera system."""

    def __init__(
        self,
        resolution: Tuple[int, int],
        device: Union[torch.device, str] = 'cpu',
        output_path: Optional[str] = None,
        return_tensor: bool = True,
        out_img_format: str = '%06d.png',
        projection: Literal['weakperspective', 'fovperspective',
                            'orthographics', 'perspective',
                            'fovorthographics'] = 'weakperspective',
        in_ndc: bool = True,
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
            return_tensor (bool, optional):
                Boolean of whether return the rendered tensor.
                Defaults to False.
            out_img_format (str, optional): name format for temp images.
                Defaults to '%06d.png'.
            projection (Literal[, optional): projection type of camera.
                Defaults to 'weakperspective'.
            in_ndc (bool, optional): cameras whether defined in NDC.
                Defaults to True.

        Returns:
            None
        """
        super().__init__(
            resolution=resolution,
            device=device,
            output_path=output_path,
            obj_path=None,
            return_tensor=return_tensor,
            out_img_format=out_img_format,
            projection=projection,
            in_ndc=in_ndc,
            **kwargs)

    def forward(self,
                meshes: Optional[Meshes] = None,
                vertices: Optional[torch.Tensor] = None,
                faces: Optional[torch.Tensor] = None,
                K: Optional[torch.Tensor] = None,
                R: Optional[torch.Tensor] = None,
                T: Optional[torch.Tensor] = None,
                indexs: Optional[Iterable[int]] = None,
                **kwargs):
        """Render depth map.

        The params are the same as MeshBaseRenderer.
        """
        cameras = self.init_cameras(K=K, R=R, T=T)
        meshes = self.prepare_meshes(meshes, vertices, faces)
        vertices = meshes.verts_padded()
        verts_depth = cameras.compute_depth_of_points(vertices)
        verts_depth_rgb = verts_depth.repeat(1, 1, 3)

        self.norm_scale = torch.max(verts_depth).clone(
        ) if self.norm_scale is None else self.norm_scale
        verts_depth_rgb /= self.norm_scale
        meshes.textures = TexturesVertex(verts_features=verts_depth_rgb)
        renderer = self.init_renderer(cameras, self.lights)

        rendered_images = renderer(meshes)
        rgbs, _ = rendered_images[
            ..., :3], (rendered_images[..., 3:] > 0) * 1.0
        if self.output_path is not None:
            output_images = rgbs.detach().cpu().numpy()
            p2, p98 = np.percentile(output_images, (2, 98))
            img_rescale = exposure.rescale_intensity(
                output_images, in_range=(p2, p98))

            # Adaptive Equalization
            output_images = (
                exposure.equalize_adapthist(img_rescale, clip_limit=0.03) *
                255).astype(np.uint8)

            for idx, real_idx in enumerate(indexs):
                folder = self.temp_path if self.temp_path is not None else\
                    self.output_path
                cv2.imwrite(
                    osp.join(folder, self.out_img_format % real_idx),
                    output_images[idx])

        if self.return_tensor:
            return rendered_images
        else:
            return None

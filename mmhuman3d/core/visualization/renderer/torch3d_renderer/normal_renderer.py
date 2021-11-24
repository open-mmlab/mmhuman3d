import os.path as osp
import warnings
from typing import Iterable, Optional, Union

import cv2
import numpy as np
import torch
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes

from .base_renderer import MeshBaseRenderer

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class NormalRenderer(MeshBaseRenderer):
    """Render depth map with the help of camera system."""

    def __init__(
        self,
        resolution: Iterable[int] = [1024, 1024],
        device: Union[torch.device, str] = 'cpu',
        output_path: Optional[str] = None,
        return_tensor: bool = True,
        img_format: str = '%06d.png',
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
            return_tensor (bool, optional):
                Boolean of whether return the rendered tensor.
                Defaults to False.
            img_format (str, optional): name format for temp images.
                Defaults to '%06d.png'.
            projection (Literal[, optional): projection type of camera.
                Defaults to 'weakperspective'.
            in_ndc (bool, optional): cameras whether defined in NDC.
                Defaults to True.
        """
        super().__init__(
            resolution=resolution,
            device=device,
            output_path=output_path,
            obj_path=None,
            return_tensor=return_tensor,
            img_format=img_format,
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
                indexs: Optional[Iterable[int]] = None):
        """Render normal map.

        The params are the same as MeshBaseRenderer.
        """
        cameras = self.init_cameras(K=K, R=R, T=T)
        if meshes is None:
            assert (vertices is not None) and (faces is not None),\
                'No mesh data input.'

            meshes = Meshes(
                verts=vertices.to(self.device),
                faces=faces.to(self.device),
            )
        else:
            if (vertices is not None) or (faces is not None):
                warnings.warn('Redundant input, will only use meshes.')
            meshes = meshes.to(self.device)
        verts_normals = cameras.compute_normal_of_meshes(meshes)
        verts_depth_rgb = verts_normals.clone()
        meshes.textures = TexturesVertex(verts_features=verts_depth_rgb)
        renderer = self.init_renderer(cameras, self.lights)
        rendered_images = renderer(meshes)
        rgbs, valid_mask = rendered_images[
            ..., :3], (rendered_images[..., 3:] > 0) * 1.0
        if self.output_path is not None:
            scene = rgbs * valid_mask
            R, G, B = torch.unbind(scene, -1)
            scene = torch.cat(
                [R.unsqueeze(-1),
                 G.unsqueeze(-1),
                 B.unsqueeze(-1)], -1)
            scene = (scene + 1) / 2
            output_images = (scene - scene.min()) / (scene.max() - scene.min())
            output_images = (output_images.detach().cpu().numpy() *
                             255).astype(np.uint8)

            for idx, real_idx in enumerate(indexs):
                folder = self.temp_path if self.temp_path is not None else\
                    self.output_path
                cv2.imwrite(
                    osp.join(folder, self.img_format % real_idx),
                    output_images[idx])
        if self.return_tensor:
            return rendered_images
        else:
            return None

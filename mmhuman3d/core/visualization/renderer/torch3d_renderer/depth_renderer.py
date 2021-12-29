from typing import Iterable, List, Optional, Tuple, Union

import torch
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes

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
        return_type: Optional[List] = None,
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
            return_type (List, optional): the type of tensor to be
                returned. 'tensor' denotes return the determined tensor. E.g.,
                return silhouette tensor of (B, H, W) for SilhouetteRenderer.
                'rgba' denotes the colorful RGBA tensor to be written.
                Will be same for MeshBaseRenderer.
                Will return a depth_map for 'tensor' and a normalize map for
                'rgba'.
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
            return_type=return_type,
            out_img_format=out_img_format,
            projection=projection,
            in_ndc=in_ndc,
            **kwargs)

    def set_render_params(self, **kwargs):
        super().set_render_params(**kwargs)
        self.shader_type = 'nolight'

    def forward(self,
                meshes: Optional[Meshes] = None,
                vertices: Optional[torch.Tensor] = None,
                faces: Optional[torch.Tensor] = None,
                K: Optional[torch.Tensor] = None,
                R: Optional[torch.Tensor] = None,
                T: Optional[torch.Tensor] = None,
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
        cameras = self.init_cameras(K=K, R=R, T=T)
        meshes = self.prepare_meshes(meshes, vertices, faces)
        vertices = meshes.verts_padded()
        verts_depth = cameras.compute_depth_of_points(vertices)
        verts_depth_rgb = verts_depth.repeat(1, 1, 3)

        meshes.textures = TexturesVertex(verts_features=verts_depth_rgb)
        renderer = self.init_renderer(cameras, self.lights)

        depth_map = renderer(meshes)

        if self.output_path is not None or 'rgba' in self.return_type:
            rgbs, valid_mask = depth_map[
                ..., :3], (depth_map[..., 3:] > 0) * 1.0
            rgbs = rgbs / rgbs.max()
            if self.output_path is not None:
                self.write_images(rgbs, valid_mask, images, indexes)

        results = {}
        if 'tensor' in self.return_type:
            results.update(tensor=depth_map[..., 0])
        if 'rgba' in self.return_type:
            results.update(rgba=torch.cat([rgbs, valid_mask], -1))

        return results

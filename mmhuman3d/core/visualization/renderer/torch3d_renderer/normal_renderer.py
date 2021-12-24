from typing import Iterable, Optional, Union

import torch
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes

from .base_renderer import MeshBaseRenderer
from .builder import RENDERER

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@RENDERER.register_module(name=['Normal', 'normal', 'NormalRenderer'])
class NormalRenderer(MeshBaseRenderer):
    """Render depth map with the help of camera system."""

    def __init__(
        self,
        resolution: Iterable[int] = [1024, 1024],
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
                images: Optional[torch.Tensor] = None,
                indexs: Optional[Iterable[int]] = None,
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
            indexs (Optional[Iterable[int]], optional): indexs for the images.
                Defaults to None.

        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """
        meshes = self.prepare_meshes(meshes, vertices, faces)

        cameras = self.init_cameras(K=K, R=R, T=T)
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
            rendered_images = (scene - scene.min()) / (
                scene.max() - scene.min())
            rendered_images = torch.cat([rendered_images, valid_mask], -1)
            self.write_images(rendered_images, images, indexs)

        if self.return_tensor:
            return rendered_images
        else:
            return None

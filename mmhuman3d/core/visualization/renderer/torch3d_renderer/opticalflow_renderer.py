from typing import Optional, Union

import torch
import torch.nn as nn
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes
from pytorch3d.structures.utils import padded_to_packed

from .builder import RENDERER


class OpticalFlowShader(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @staticmethod
    def gen_mesh_grid(H, W, N: int = 1):
        h_grid = torch.linspace(-1, 1, H).view(-1, 1).repeat(1, W)
        v_grid = torch.linspace(-1, 1, W).repeat(H, 1)
        mesh_grid = torch.cat((v_grid.unsqueeze(2), h_grid.unsqueeze(2)),
                              dim=2)
        mesh_grid = mesh_grid.unsqueeze(0).repeat(N, 1, 1, 1)
        mesh_grid = torch.cat([mesh_grid, torch.zeros(N, H, W, 1)], -1)
        return mesh_grid  # (N, H, W, 2)

    def to(self, device):
        return self

    def forward(self, fragments: Fragments, meshes: Meshes,
                verts_scene_flow: torch.Tensor, **kwargs) -> torch.Tensor:

        faces = meshes.faces_packed()  # (F, 3)
        verts_scene_flow = padded_to_packed(verts_scene_flow)
        faces_flow = verts_scene_flow[faces]
        pixel_flow = interpolate_face_attributes(
            pix_to_face=fragments.pix_to_face,
            barycentric_coords=fragments.bary_coords,
            face_attributes=faces_flow)
        N, H, W, _, _ = pixel_flow.shape
        mesh_grid = self.gen_mesh_grid(N=N, H=H, W=W)
        pixel_flow = pixel_flow.squeeze(-2) + mesh_grid.to(pixel_flow.device)
        return pixel_flow


@RENDERER.register_module(name=['opticalflow', 'optical_flow', 'OpticalFlow'])
class OpticalFlowRenderer(nn.Module):

    def __init__(self, rasterizer, **kwargs):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = OpticalFlowShader()

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)

    def forward(
        self,
        meshes_source: Optional[Meshes] = None,
        meshes_target: Optional[Meshes] = None,
        cameras=None,
        cameras_source=None,
        cameras_target=None,
        **kwargs,
    ) -> Union[torch.Tensor, None]:
        """Render Meshes.

        Args:
            meshes (Optional[Meshes], optional): meshes to be rendered.
                Defaults to None.
            K (Optional[torch.Tensor], optional): Camera intrinsic matrixs.
                Defaults to None.
            R (Optional[torch.Tensor], optional): Camera rotation matrixs.
                Defaults to None.
            T (Optional[torch.Tensor], optional): Camera tranlastion matrixs.
                Defaults to None.
            indexes (Optional[Iterable[int]], optional): indexes for the
                images.
                Defaults to None.
        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """
        assert len(meshes_source) == len(meshes_target)

        if cameras_source is None:
            cameras_source = cameras
        if cameras_target is None:
            cameras_target = cameras

        fragments_source = self.rasterizer(meshes_source,
                                           **{'cameras': cameras_source})
        fragments_target = self.rasterizer(meshes_target,
                                           **{'cameras': cameras_target})

        if cameras_source is None or cameras_target is None:
            msg = 'Cameras must be specified either at initialization \
                or in the forward pass of OpticalFLowShader'

            raise ValueError(msg)

        verts_source_ndc = cameras_source.transform_points_ndc(
            meshes_source.verts_padded())
        verts_target_ndc = cameras_target.transform_points_ndc(
            meshes_target.verts_padded())

        verts_scene_flow = verts_target_ndc - verts_source_ndc
        pixel_scene_flow = self.shader(
            fragments=fragments_target,
            meshes=meshes_target,
            verts_scene_flow=verts_scene_flow,
            **kwargs)

        mask_target = (fragments_target.pix_to_face >= 0).long()

        visible_face_idx_source = fragments_source.pix_to_face.unique()[1:]
        visible_face_idx_target = fragments_target.pix_to_face.unique()[1:]
        face_visibility_packed_source = torch.zeros(
            meshes_source.faces_packed().shape[0]).long().to(
                meshes_source.device)
        face_visibility_packed_source[visible_face_idx_source] = 1

        face_visibility_packed_target = torch.zeros(
            meshes_target.faces_packed().shape[0]).long().to(
                meshes_target.device)
        face_visibility_packed_target[visible_face_idx_target] = 1

        face_visibility_packed = face_visibility_packed_source * \
            face_visibility_packed_target
        shape = fragments_target.pix_to_face.shape
        visiblity_mask = face_visibility_packed[
            fragments_target.pix_to_face.view(-1)].view(shape) * mask_target

        pixel_scene_flow = torch.cat([pixel_scene_flow, visiblity_mask], -1)
        return pixel_scene_flow

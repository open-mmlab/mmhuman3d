import warnings
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io.obj_io import load_objs_as_meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh import TexturesUV
from pytorch3d.renderer.mesh.rasterizer import (
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.utils import padded_to_packed

from mmhuman3d.core.cameras.cameras import (
    FoVOrthographicCameras,
    MMCamerasBase,
)
from mmhuman3d.utils.path_utils import check_path_suffix
from .utils import array2tensor, rgb2bgr


class UVRenderer(nn.Module):
    """Renderer for SMPL(x) UV map."""

    def __init__(
        self,
        resolution: Tuple[int] = 1024,
        model_type: Optional[str] = 'smpl',
        uv_param_path: Optional[str] = None,
        obj_path: Optional[str] = None,
        device: Union[torch.device, str] = 'cpu',
        threshold_size: int = 512,
        # TODO: Solved the sample bug when the resolution is too small.
        # set threshold_size is just a temporary solution.

        # TODO: add smplx_uv.npz and eval the warping & sampling of smplx
        # model.
    ):
        super().__init__()
        self.threshold_size = threshold_size
        num_verts = {'smpl': 6890, 'smplx': 10475}
        self.NUM_VERTS = num_verts[model_type]
        self.device = device
        self.resolution = (resolution, resolution) if isinstance(
            resolution, int) else resolution
        self.uv_param_path = uv_param_path
        self.obj_path = obj_path
        if uv_param_path is not None:
            check_path_suffix(uv_param_path, allowed_suffix=['npz'])
            param_dict = dict(np.load(uv_param_path))

            verts_uv = torch.Tensor(param_dict['verts_uv'])
            verts_u, verts_v = torch.unbind(verts_uv, -1)
            verts_v_ = 1 - verts_u.unsqueeze(-1)
            verts_u_ = verts_v.unsqueeze(-1)
            self.verts_uv = torch.cat([verts_u_, verts_v_], -1).to(self.device)
            self.faces_uv = torch.LongTensor(param_dict['faces_uv']).to(
                self.device)

            self.NUM_VT = self.verts_uv.shape[0]

            self.faces_tensor = torch.LongTensor(param_dict['faces'].astype(
                np.int64)).to(self.device)
            self.num_faces = self.faces_uv.shape[0]
        elif obj_path is not None:
            check_path_suffix(obj_path, allowed_suffix=['obj'])
            mesh_template = load_objs_as_meshes([obj_path])
            self.faces_uv = mesh_template.textures.faces_uvs_padded()[0].to(
                self.device)
            self.verts_uv = mesh_template.textures.verts_uvs_padded()[0].to(
                self.device)
            self.NUM_VT = self.verts_uv.shape[0]
            self.faces_tensor = mesh_template.faces_padded()[0].to(self.device)
            self.num_faces = self.faces_uv.shape[0]
        self.update_fragments()
        self.update_face_uv_pixel()

        self = self.to(self.device)

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        for k in dir(self):
            if isinstance(getattr(self, k), (torch.Tensor)):
                setattr(self, k, getattr(self, k).to(device))
        return self

    def update_fragments(self):
        """Update pix_to_face, bary_coords."""
        rasterizer = MeshRasterizer(
            cameras=FoVOrthographicCameras(
                min_x=1, max_x=0, max_y=1, min_y=0, device=self.device),
            raster_settings=RasterizationSettings(
                blur_radius=0,
                image_size=self.resolution,
                faces_per_pixel=1,
                perspective_correct=False,
            )).to(self.device)
        verts_uv = torch.cat([
            self.verts_uv[None],
            torch.ones(1, self.NUM_VT, 1).to(self.device)
        ], -1)

        fragments = rasterizer(
            Meshes(verts=verts_uv, faces=self.faces_uv[None]))
        self.pix_to_face = fragments.pix_to_face[0, ..., 0]
        self.bary_coords = fragments.bary_coords[0, ..., 0, :]
        self.mask = (self.pix_to_face >= 0).long()

    def update_face_uv_pixel(self):
        """Move the pixels lie on the edges inside the mask, then refine the
        rest points by searching the nearest pixel in the faces it should be
        in."""
        H, W = self.resolution
        device = self.device
        cameras = FoVOrthographicCameras(
            min_x=1, max_x=0, max_y=1, min_y=0, device=self.device)
        verts_uv = torch.cat([
            self.verts_uv[None],
            torch.ones(1, self.NUM_VT, 1).to(self.device)
        ], -1)

        verts_uv_pixel = cameras.transform_points_screen(
            verts_uv, image_size=self.resolution).round().long()[0, ..., :2]
        verts_uv_pixel[..., 0] = torch.clip(
            verts_uv_pixel[..., 0], min=0, max=W - 1)
        verts_uv_pixel[..., 1] = torch.clip(
            verts_uv_pixel[..., 1], min=0, max=H - 1)
        verts_uv_pixel = verts_uv_pixel.long()
        mask = self.mask

        wrong_indexes = torch.where(
            mask[verts_uv_pixel.view(-1, 2)[:, 1],
                 verts_uv_pixel.view(-1, 2)[:, 0]] == 0)[0]
        for wrong_index in wrong_indexes:
            proposed_faces = torch.where(self.faces_uv == wrong_index)[0]
            vert_xy = verts_uv_pixel[wrong_index]
            faces_xy = []
            for face_id in proposed_faces:
                x = torch.where(self.pix_to_face == face_id)[1]
                y = torch.where(self.pix_to_face == face_id)[0]
                if x.shape[0] > 0:
                    face_xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], -1)
                    faces_xy.append(face_xy)
            if len(faces_xy) > 0:
                faces_xy = torch.cat(faces_xy, 0)
                min_arg = torch.argmin(
                    torch.sqrt(((faces_xy - vert_xy) *
                                (faces_xy - vert_xy)).sum(-1).float()))

                verts_uv_pixel[wrong_index] = faces_xy[min_arg]

        up_bound = ((mask[:-1] - mask[1:]) < 0).long()
        bottom_bound = ((mask[1:] - mask[:-1]) < 0).long()
        left_bound = ((mask[:, :-1] - mask[:, 1:]) < 0).long()
        right_bound = ((mask[:, 1:] - mask[:, :-1]) < 0).long()

        left_bound = torch.cat(
            [left_bound, torch.zeros(H, 1).to(device)], 1).unsqueeze(-1)
        right_bound = torch.cat([torch.zeros(H, 1).to(device), right_bound],
                                1).unsqueeze(-1)
        up_bound = torch.cat([up_bound, torch.zeros(1, W).to(device)],
                             0).unsqueeze(-1)
        bottom_bound = torch.cat([torch.zeros(1, W).to(device), bottom_bound],
                                 0).unsqueeze(-1)

        leftup_corner_ = ((mask[:-1, :-1] - mask[1:, 1:]) < 0).long()
        rightup_corner_ = ((mask[:-1, 1:] - mask[1:, :-1]) < 0).long()
        leftbottom_corner_ = ((mask[1:, :-1] - mask[:-1, 1:]) < 0).long()
        rightbottom_corner_ = ((mask[1:, 1:] - mask[:-1, :-1]) < 0).long()

        leftup_corner = torch.zeros_like(mask).long()
        leftup_corner[:-1, :-1] = leftup_corner_
        leftup_corner = leftup_corner.unsqueeze(-1)

        rightup_corner = torch.zeros_like(mask).long()
        rightup_corner[:-1, 1:] = rightup_corner_
        rightup_corner = rightup_corner.unsqueeze(-1)

        leftbottom_corner = torch.zeros_like(mask).long()
        leftbottom_corner[1:, :-1] = leftbottom_corner_
        leftbottom_corner = leftbottom_corner.unsqueeze(-1)

        rightbottom_corner = torch.zeros_like(mask).long()
        rightbottom_corner[1:, 1:] = rightbottom_corner_
        rightbottom_corner = rightbottom_corner.unsqueeze(-1)

        stride_uv_mask = torch.cat([
            right_bound * -1 + left_bound * 1 + rightbottom_corner * -1 +
            leftbottom_corner * 1 + rightup_corner * -1 + leftup_corner * 1,
            up_bound * 1 + bottom_bound * -1 + rightbottom_corner * -1 +
            leftbottom_corner * -1 + rightup_corner * 1 + leftup_corner * 1
        ], -1).long()

        verts_uv_pixel = verts_uv_pixel + stride_uv_mask[
            verts_uv_pixel.view(-1, 2)[:, 1],
            verts_uv_pixel.view(-1, 2)[:, 0]].view(self.NUM_VT, 2)

        face_uv_pixel = verts_uv_pixel[self.faces_uv]

        face_uv_pixel = face_uv_pixel.long()
        self.face_uv_pixel = face_uv_pixel

    def forward(self,
                verts_attr: Optional[torch.Tensor],
                resolution: Optional[Iterable[int]] = None) -> torch.Tensor:
        """Interpolate the vertex attributes to a map.

        Args:
            verts_attr (Optional[torch.Tensor]): shape should be (N, V, C),
                required.
            resolution (Optional[Iterable[int]], optional): resolution to
                override self.resolution. If None, will use self.resolution.
                Defaults to None.

        Returns:
            torch.Tensor: interpolated maps of (N, H, W, C)
        """
        if verts_attr.ndim == 2:
            verts_attr = verts_attr[None]
        if resolution is not None and resolution != self.resolution:
            self.resolution = resolution
            self.update_fragments()
            self.update_face_uv_pixel()

        bary_coords = self.bary_coords
        pix_to_face = self.pix_to_face

        N, V, C = verts_attr.shape
        assert V == self.NUM_VERTS
        verts_attr = verts_attr.view(N * V, C).to(self.device)
        offset_idx = torch.arange(0, N).long() * (self.NUM_VERTS - 1)
        faces_packed = self.faces_tensor[None].repeat(
            N, 1, 1) + offset_idx.view(-1, 1, 1).to(self.device)
        faces_packed = faces_packed.view(-1, 3)
        face_attr = verts_attr[faces_packed]
        assert face_attr.shape == (N * self.num_faces, 3, C)
        pix_to_face = self.pix_to_face.unsqueeze(0).repeat(N, 1,
                                                           1).unsqueeze(-1)
        bary_coords = self.bary_coords[None].repeat(N, 1, 1, 1).unsqueeze(-2)
        maps_padded = interpolate_face_attributes(
            pix_to_face=pix_to_face.to(self.device),
            barycentric_coords=bary_coords.to(self.device),
            face_attributes=face_attr.to(self.device),
        ).squeeze(-2)
        return maps_padded

    def forward_normal_map(self,
                           meshes: Meshes = None,
                           vertices: torch.Tensor = None,
                           resolution: Optional[Iterable[int]] = None,
                           cameras: MMCamerasBase = None) -> torch.Tensor:
        """Interpolate verts normals to a normal map.

        Args:
            meshes (Meshes): input smpl mesh.
                Will override vertices if both not None.
                Defaults to None.
            vertices (torch.Tensor, optional):
                smpl vertices. Defaults to None.
            resolution (Optional[Iterable[int]], optional): resolution to
                override self.resolution. If None, will use self.resolution.
                Defaults to None.
            cameras (MMCamerasBase, optional):
                cameras to see the mesh.
                Defaults to None.
        Returns:
            torch.Tensor: Normal map of shape (N, H, W, 3)
        """
        if meshes is not None:
            verts_normals = meshes.verts_normals_padded()
        elif meshes is None and vertices is not None:
            meshes = Meshes(
                verts=vertices,
                faces=self.faces_tensor[None].repeat(vertices.shape[0], 1, 1))
            verts_normals = meshes.verts_normals_padded()
        else:
            raise ValueError('No valid input.')
        verts_normals = meshes.verts_normals_padded()
        if cameras:
            verts_normals = cameras.get_world_to_view_transform(
            ).transform_normals(verts_normals)
        normal_map = self.forward(
            verts_attr=verts_normals, resolution=resolution)
        return normal_map

    def forward_uvd_map(self,
                        meshes: Meshes = None,
                        vertices: torch.Tensor = None,
                        resolution: Optional[Iterable[int]] = None,
                        cameras: MMCamerasBase = None) -> torch.Tensor:
        """Interpolate the verts xyz value to a uvd map.

        Args:
            meshes (Meshes): input smpl mesh.
                Defaults to None.
            vertices (torch.Tensor, optional):
                smpl vertices. Will override meshes if both not None.
                Defaults to None.
            resolution (Optional[Iterable[int]], optional): resolution to
                override self.resolution. If None, will use self.resolution.
                Defaults to None.
            cameras (MMCamerasBase, optional):
                cameras to see the mesh.
                Defaults to None.

        Returns:
            torch.Tensor: UVD map of shape (N, H, W, 3)
        """
        if vertices is not None:
            verts_uvd = vertices
        elif vertices is None and meshes is not None:
            verts_uvd = meshes.verts_padded()
        else:
            raise ValueError('No valid input.')
        if cameras:
            verts_uvd = cameras.get_world_to_view_transform(
            ).transform_normals(verts_uvd)
        uvd_map = self.forward(verts_attr=verts_uvd, resolution=resolution)
        return uvd_map

    def vertex_resample(
        self,
        maps_padded: torch.Tensor,
        h_flip: bool = False,
    ) -> torch.Tensor:
        """Resample the vertex attributes from a map.

        Args:
            maps_padded (torch.Tensor): shape should be (N, H, W, C). Required.
            h_flip (bool, optional): whether flip horizontally.
                Defaults to False.

        Returns:
            torch.Tensor: resampled vertex attributes. Shape will be (N, V, C)
        """
        if maps_padded.ndim == 3:
            maps_padded = maps_padded[None]

        if h_flip:
            maps_padded = torch.flip(maps_padded, dims=[2])
        N, H, W, C = maps_padded.shape

        if H < self.threshold_size or W < self.threshold_size:
            maps_padded = F.interpolate(
                maps_padded.permute(0, 3, 1, 2),
                size=(self.threshold_size, self.threshold_size),
                mode='bicubic',
                align_corners=False).permute(0, 2, 3, 1)
            H, W = self.threshold_size, self.threshold_size
        if (H, W) != self.resolution:
            self.resolution = (H, W)
            self.update_fragments()
            self.update_face_uv_pixel()
        offset_idx = torch.arange(0, N).long() * (self.NUM_VERTS - 1)
        faces_packed = self.faces_tensor[None].repeat(
            N, 1, 1) + offset_idx.view(-1, 1, 1).to(self.device)
        faces_packed = faces_packed.view(-1, 3)

        verts_feature_packed = torch.zeros(N * self.NUM_VERTS,
                                           C).to(self.device)

        face_uv_pixel = self.face_uv_pixel.view(-1, 2)
        verts_feature_packed[
            faces_packed] = maps_padded[:, face_uv_pixel[:, 1],
                                        face_uv_pixel[:, 0]].view(
                                            N * self.num_faces, 3, C)
        verts_feature_padded = verts_feature_packed.view(N, self.NUM_VERTS, C)

        return verts_feature_padded

    def wrap_normal(
        self,
        meshes: Meshes,
        normal: torch.Tensor = None,
        normal_map: torch.Tensor = None,
    ) -> Meshes:
        """Warp a normal map or vertex normal to the input meshes.

        Args:
            meshes (Meshes): the input meshes.
            normal (torch.Tensor, optional): vertex normal. Shape should be
                (N, V, 3).
                Defaults to None.
            normal_map (torch.Tensor, optional):
                normal map. Defaults to None.

        Returns:
            Meshes: returned meshes.
        """
        if normal_map is not None and normal is None:
            normal = self.vertex_resample(normal_map)
        elif normal_map is not None and normal is not None:
            normal_map = None
        elif normal_map is None and normal is None:
            warnings.warn('Redundant input, will only take displacement.')
        batch_size = len(meshes)
        if normal.ndim == 2:
            normal = normal[None]
        assert normal.shape[1:] == (self.NUM_VERTS, 3)
        assert normal.shape[0] in [batch_size, 1]

        if normal.shape[0] == 1:
            normal = normal.repeat(batch_size, 1, 1)
        meshes = meshes.clone()

        meshes._set_verts_normals(normal)
        return meshes

    def wrap_displacement(
        self,
        meshes: Meshes,
        displacement: torch.Tensor = None,
        displacement_map: torch.Tensor = None,
    ) -> Meshes:
        """Offset a vertex displacement or displacement_map to the input
        meshes.

        Args:
            meshes (Meshes): the input meshes.
            displacement (torch.Tensor, optional): vertex displacement.
                shape should be (N, V, 3).
                Defaults to None.
            displacement_map (torch.Tensor, optional): displacement_map,
                shape should be (N, H, W, 3).
                Defaults to None.

        Returns:
            Meshes: returned meshes.
        """
        if displacement_map is not None and displacement is None:
            displacement = self.vertex_resample(displacement_map)
        elif displacement_map is not None and displacement is not None:
            displacement_map = None
            warnings.warn('Redundant input, will only take displacement.')
        elif displacement_map is None and displacement is None:
            raise ValueError('No valid input.')
        batch_size = len(meshes)
        if displacement.ndim == 2:
            displacement = displacement[None]
        assert displacement.shape[1] == self.NUM_VERTS
        assert displacement.shape[0] in [batch_size, 1]

        if displacement.shape[0] == 1:
            displacement = displacement.repeat(batch_size, 1, 1)
        C = displacement.shape[-1]
        if C == 1:
            displacement = meshes.verts_normals_padded() * displacement

        displacement = padded_to_packed(displacement)

        meshes = meshes.to(self.device)
        meshes = meshes.offset_verts(displacement)
        return meshes

    def wrap_texture(self,
                     texture_map: torch.Tensor,
                     resolution: Optional[Iterable[int]] = None,
                     mode: Optional[str] = 'bicubic',
                     is_bgr: bool = True) -> Meshes:
        """Wrap a texture map to the input meshes.

        Args:
            texture_map (torch.Tensor): the texture map to be wrapped.
                Shape should be (N, H, W, 3)
            resolution (Optional[Iterable[int]], optional): resolution to
                override self.resolution. If None, will use self.resolution.
                Defaults to None.
            mode (Optional[str], optional): interpolate mode.
                Should be in ['nearest', 'bilinear', 'trilinear', 'bicubic',
                'area'].
                Defaults to 'bicubic'.
            is_bgr (bool, optional): Whether the color channel is BGR.
                Defaults to True.

        Returns:
            Meshes: returned meshes.
        """

        assert texture_map.shape[-1] == 3
        if texture_map.ndim == 3:
            texture_map_padded = texture_map[None]
        elif texture_map.ndim == 4:
            texture_map_padded = texture_map
        else:
            raise ValueError(f'Wrong texture_map shape: {texture_map.shape}.')
        N, H, W, _ = texture_map_padded.shape

        resolution = resolution if resolution is not None else (H, W)

        if resolution != (H, W):
            texture_map_padded = F.interpolate(
                texture_map_padded.view(0, 3, 1, 2), resolution,
                mode=mode).view(0, 2, 3, 1)
        assert texture_map_padded.shape[0] in [N, 1]

        if isinstance(texture_map_padded, np.ndarray):
            texture_map_padded = array2tensor(texture_map_padded)
            is_bgr = True
        if is_bgr:
            texture_map_padded = rgb2bgr(texture_map_padded)

        if texture_map_padded.shape[0] == 1:
            texture_map_padded = texture_map_padded.repeat(N, 1, 1, 1)

        faces_uvs = self.faces_uv[None].repeat(N, 1, 1)
        verts_uvs = self.verts_uv[None].repeat(N, 1, 1)
        textures = TexturesUV(
            faces_uvs=faces_uvs, verts_uvs=verts_uvs, maps=texture_map_padded)
        return textures

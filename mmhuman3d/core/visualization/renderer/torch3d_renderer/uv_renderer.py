from collections import namedtuple
import pickle
import warnings
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io.obj_io import load_objs_as_meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh import TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.structures.utils import padded_to_packed
from pytorch3d.renderer.mesh.rasterizer import (MeshRasterizer,
                                                RasterizationSettings)
from mmhuman3d.core.cameras.cameras import (NewAttributeCameras,
                                            FoVOrthographicCameras)
from mmhuman3d.models.body_models import SMPL, SMPLX
from mmhuman3d.utils.path_utils import check_path_suffix


class UVRenderer(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """

    def __init__(self,
                 param_path: Optional[str] = None,
                 obj_path: Optional[str] = None,
                 model_type: Optional[str] = 'smpl',
                 device: Union[torch.device, str] = 'cpu',
                 resolution: Iterable[int] = (256, 256)):
        """[summary]

        Args:
            param_path (Optional[str], optional): [description].
                Defaults to None.
            obj_path (Optional[str], optional): [description].
                Defaults to None.
            model_type (Optional[str], optional): [description].
                Defaults to 'smpl'.
            device (Union[torch.device, str], optional): [description].
                Defaults to 'cpu'.
            resolution (Iterable[int], optional): [description].
                Defaults to (256, 256).
        """
        super().__init__()
        body_model_class = {'smpl': SMPL, 'smplx': SMPLX}
        self.NUM_VERTS = body_model_class[model_type].NUM_VERTS
        self.device = device
        self.resolution = resolution
        if param_path is not None:
            check_path_suffix(param_path, allowed_suffix=['pkl', 'pickle'])
            with open(param_path, 'rb') as f:
                param_dict = pickle.load(f)
            self.bary_coords = torch.Tensor(param_dict['bary_weights']).to(
                self.device)
            verts_uv = torch.Tensor(param_dict['texcoords']).to(self.device)
            verts_u, verts_v = torch.unbind(verts_uv, -1)
            verts_v_ = 1 - verts_u.unsqueeze(-1)
            verts_u_ = verts_v.unsqueeze(-1)
            self.verts_uv = torch.cat([verts_u_, verts_v_], -1).to(self.device)
            self.faces_uv = torch.LongTensor(param_dict['vt_faces']).to(
                self.device)
            self.vt_to_v = param_dict['vt_to_v']
            self.v_to_vt = param_dict['vt_to_v']
            self.NUM_VT = self.verts_uv.shape[0]
            self.pix_to_face = torch.LongTensor(param_dict['face_id']).to(
                self.device)
            self.face_tensor = torch.LongTensor(param_dict['faces'].astype(
                np.int64)).to(self.device)
            self.num_faces = self.faces_uv.shape[0]

            self.vt_to_v_index = torch.LongTensor(
                [self.vt_to_v[i] for i in range(self.NUM_VT)]).to(self.device)
        elif obj_path is not None:
            check_path_suffix(obj_path, allowed_suffix=['obj'])
            mesh_template = load_objs_as_meshes([obj_path])
            self.faces_uv = mesh_template.textures.faces_uvs_padded()[0]
            self.verts_uv = mesh_template.textures.verts_uvs_padded()[0]
            self.num_faces = self.faces_uv.shape[0]

    def raster_fragments(
            self,
            resolution: Optional[Iterable[int]] = None) -> Tuple[torch.Tensor]:
        rasterizer = MeshRasterizer(
            cameras=FoVOrthographicCameras(
                min_x=1, max_x=0, max_y=1, min_y=0, device=self.device),
            raster_settings=RasterizationSettings(
                blur_radius=0,
                image_size=resolution,
                faces_per_pixel=1,
                perspective_correct=False,
            )).to(self.device)
        verts_uv = torch.cat([
            self.verts_uv[None],
            torch.ones(1, self.NUM_VT, 1).to(self.device)
        ], -1)

        fragments = rasterizer(
            Meshes(verts=verts_uv, faces=self.faces_uv[None]))
        pix_to_face = fragments.pix_to_face[0, ..., 0]
        bary_coords = fragments.bary_coords[0, ..., 0, :]
        mask = (pix_to_face >= 0).long()
        return pix_to_face, bary_coords, mask

    def forward(self,
                verts_attr: Optional[torch.Tensor],
                resolution: Optional[Iterable[int]] = None) -> torch.Tensor:
        """[summary]

        Args:
            verts_attr (Optional[torch.Tensor]): [description]

        Returns:
            [type]: [description]
        """
        if verts_attr.ndim == 2:
            verts_attr = verts_attr[None]
        if resolution is not None and resolution != self.resolution:
            pix_to_face, bary_coords, _ = self.raster_fragments(
                resolution=resolution)
        else:
            bary_coords = self.bary_coords
            pix_to_face = self.pix_to_face
        N, V, D = verts_attr.shape
        assert V == self.NUM_VERTS
        verts_attr = verts_attr.view(N * V, D).to(self.device)
        offset_idx = torch.arange(0, N).long() * (self.NUM_VERTS - 1)
        faces_packed = self.face_tensor[None].repeat(
            N, 1, 1) + offset_idx.view(-1, 1, 1).to(self.device)
        faces_packed = faces_packed.view(-1, 3)
        face_attr = verts_attr[faces_packed]
        assert face_attr.shape == (N * self.num_faces, 3, D)
        pix_to_face = self.pix_to_face.unsqueeze(0).repeat(N, 1,
                                                           1).unsqueeze(-1)
        bary_coords = self.bary_coords[None].repeat(N, 1, 1, 1).unsqueeze(-2)
        maps_padded = interpolate_face_attributes(
            pix_to_face=pix_to_face.to(self.device),
            barycentric_coords=bary_coords.to(self.device),
            face_attributes=face_attr.to(self.device),
        ).squeeze(-2)
        return maps_padded

    def forward_normal_map(
            self,
            meshes: Meshes,
            resolution: Optional[Iterable[int]] = None,
            cameras: NewAttributeCameras = None) -> torch.Tensor:
        verts_normals = meshes.verts_normals_padded()
        if cameras:
            verts_normals = cameras.get_world_to_view_transform(
            ).transform_normals(verts_normals)
        normal_map = self.forward(
            verts_attr=verts_normals, resolution=resolution)
        return normal_map

    def forward_uvd_map(self,
                        meshes: Meshes,
                        resolution: Optional[Iterable[int]] = None,
                        cameras: NewAttributeCameras = None) -> torch.Tensor:
        verts_uvd = meshes.verts_padded()
        if cameras:
            verts_uvd = cameras.get_world_to_view_transform(
            ).transform_normals(verts_uvd)
        uvd_map = self.forward(verts_attr=verts_uvd, resolution=resolution)
        return uvd_map

    def resample(self,
                 maps_padded: torch.Tensor,
                 h_flip: bool = False,
                 **kwargs) -> torch.Tensor:
        """[summary]

        Args:
            maps_padded (torch.Tensor): shape should be (N, H, W, C) or
                (H, W, C)
            h_flip (bool, optional): whether flip the map horizontally.
                Defaults to False.
        """
        if maps_padded.ndim == 3:
            maps_padded = maps_padded[None].to(self.device)
        if h_flip:
            maps_padded = torch.flip(maps_padded, dims=[2])
        N, H, W, C = maps_padded.shape

        triangle_uv = self.verts_uv[self.faces_uv] * torch.LongTensor(
            [W, H]).view(1, 1, 2).to(self.device)
        triangle_uv[..., 0] = torch.clip(triangle_uv[..., 0], 0, W - 1)
        triangle_uv[..., 1] = torch.clip(triangle_uv[..., 1], 0, H - 1)
        triangle_uv = triangle_uv.long().view(-1, 2)

        offset_idx = torch.arange(0, N).long() * (self.NUM_VERTS - 1)
        faces_packed = self.face_tensor[None].repeat(
            N, 1, 1) + offset_idx.view(-1, 1, 1).to(self.device)
        faces_packed = faces_packed.view(-1, 3)

        verts_feature = torch.zeros(N * self.NUM_VERTS, 3).to(self.device)

        verts_feature[faces_packed.view(
            -1,
            3)] = maps_padded[:, H - triangle_uv[:, 0],
                              triangle_uv[:, 1]].view(N * self.num_faces, 3, C)
        verts_feature = verts_feature.view(N, self.NUM_VERTS, 3)

        return verts_feature

    def warp_normal_map(
        self,
        meshes: Meshes,
        normal: torch.Tensor = None,
        normal_map: torch.Tensor = None,
    ) -> Meshes:
        if normal_map is not None and normal is None:
            normal = self.resample(normal_map)
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

    def warp_displacement(
        self,
        meshes: Meshes,
        displacement: torch.Tensor = None,
        displacement_map: torch.Tensor = None,
    ) -> Meshes:

        if displacement_map is not None and displacement is None:
            displacement = self.resample(displacement_map)
        elif displacement_map is not None and displacement is not None:
            displacement_map = None
        elif displacement_map is None and displacement is None:
            warnings.warn('Redundant input, will only take displacement.')
        batch_size = len(meshes)
        if displacement.ndim == 2:
            displacement = displacement[None]
        assert displacement.shape[1:] == (self.NUM_VERTS, 3)
        assert displacement.shape[0] in [batch_size, 1]

        if displacement.shape[0] == 1:
            displacement = displacement.repeat(batch_size, 1, 1)
        displacement = padded_to_packed(displacement)
        meshes = meshes.offset_verts(displacement)
        return meshes

    def warp_texture_map(self,
                         meshes: Meshes,
                         texture_map: torch.Tensor,
                         resolution: Optional[Iterable[int]] = None,
                         mode: Optional[str] = 'bicubic',
                         is_bgr: bool = True) -> Meshes:
        """[summary]

        Args:
            meshes (Meshes): [description]
            texture_map (torch.Tensor): [description]
            resolution (Optional[Iterable[int]], optional): [description].
                Defaults to None.
            mode (Optional[str], optional): [description].
                Defaults to 'bicubic'.
            is_bgr (bool, optional): [description].
                Defaults to True.

        Returns:
            [type]: [description]
        """
        texture_map = self.normalize(
            texture_map, min_value=0, max_value=1, dtype=torch.float32)
        resolution = resolution if resolution is not None else self.resolution

        batch_size = len(meshes)
        if texture_map.ndim == 3:
            texture_map = texture_map[None]
        _, H, W, _ = texture_map.shape
        if resolution != (H, W):
            texture_map = F.interpolate(texture_map, resolution, mode=mode)
        assert texture_map.shape[0] in [batch_size, 1]

        if isinstance(texture_map, np.ndarray):
            texture_map = self.array2tensor(texture_map)
            is_bgr = True
        if is_bgr:
            texture_map = self.rgb2bgr(texture_map)

        if texture_map.shape[0] == 1:
            texture_map = texture_map.repeat(batch_size, 1, 1, 1)

        faces_uvs = self.faces_uv[None].repeat(batch_size, 1, 1)
        verts_uvs = self.verts_uv[None].repeat(batch_size, 1, 1)
        textures = TexturesUV(
            faces_uvs=faces_uvs, verts_uvs=verts_uvs, maps=texture_map)
        meshes.textures = textures
        return meshes

    @staticmethod
    def rgb2bgr(rgbs) -> Union[torch.Tensor, np.ndarray]:
        """[summary]"""
        if isinstance(rgbs, torch.Tensor):
            bgrs = torch.cat(
                [rgbs[..., 0, None], rgbs[..., 1, None], rgbs[..., 2, None]],
                -1)
        elif isinstance(rgbs, np.ndarray):
            bgrs = np.concatenate(
                [rgbs[..., 0, None], rgbs[..., 1, None], rgbs[..., 2, None]],
                -1)
        return bgrs

    @staticmethod
    def normalize(value, min_value, max_value,
                  dtype) -> Union[torch.Tensor, np.ndarray]:
        """[summary]

        Args:
            value ([type]): [description]
            min_value ([type]): [description]
            max_value ([type]): [description]
            dtype ([type]): [description]

        Returns:
            [type]: [description]
        """
        value = (value - value.min()) / (value.max() - value.min() + 1e-9) * (
            max_value - min_value) + min_value
        if isinstance(value, torch.Tensor):
            return value.type(dtype)
        elif isinstance(value, np.ndarray):
            return value.astype(dtype)

    def tensor2array(self, image) -> np.ndarray:
        """"""
        image = self.normalize(
            image, min_value=0, max_value=255, dtype=np.uint8)
        return image

    def array2tensor(self, image) -> torch.Tensor:
        """"""
        image = self.normalize(
            image, min_value=0, max_value=1, dtype=torch.float32)
        return image

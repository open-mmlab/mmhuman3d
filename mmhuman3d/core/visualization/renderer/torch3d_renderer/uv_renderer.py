import pickle
from typing import Iterable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io.obj_io import load_objs_as_meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh import TexturesUV
from pytorch3d.structures import Meshes

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
                Defaults toNone.
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
            bary_coords = torch.Tensor(param_dict['bary_weights'])
            verts_uv = torch.Tensor(param_dict['texcoords'])
            verts_u, verts_v = torch.unbind(verts_uv, -1)
            verts_v_ = 1 - verts_u.unsqueeze(-1)
            verts_u_ = verts_v.unsqueeze(-1)
            self.verts_uv = torch.cat([verts_u_, verts_v_], -1)
            self.faces_uv = torch.Tensor(param_dict['vt_faces'])
            self.num_faces = self.faces_uv.shape[0]
            self.vt_to_v = param_dict['vt_to_v']
            self.v_to_vt = param_dict['vt_to_v']
            self.NUM_VT = self.verts_uv.shape[0]
            face_id = torch.Tensor(param_dict['face_id'])
            if resolution != (256, 256):
                self.bary_coords = F.interpolate(
                    bary_coords[None], resolution, mode='bicubic').clamp(
                        min=0, max=1)[0]
                self.pix_to_face = F.interpolate(
                    face_id[None], resolution, mode='bicubic').long()[0]
            self.vt_to_v_index = torch.Tensor(
                [self.vt_to_v[i] for i in range(self.NUM_VT)])
        elif obj_path is not None:
            check_path_suffix(obj_path, allowed_suffix=['obj'])
            mesh_template = load_objs_as_meshes([obj_path])
            self.faces_uv = mesh_template.textures.faces_uvs_padded()[0]
            self.verts_uv = mesh_template.textures.verts_uvs_padded()[0]
            self.num_faces = self.faces_uv.shape[0]

    def forward(self, verts_attr: Optional[torch.Tensor]):
        """[summary]

        Args:
            verts_attr (Optional[torch.Tensor]): [description]

        Returns:
            [type]: [description]
        """
        if verts_attr.ndim == 2:
            verts_attr = verts_attr[None]
        batch_size, num_verts, D = verts_attr
        assert num_verts == self.NUM_VERTS
        verts_attr = verts_attr.view(batch_size * num_verts, D)
        vt_attr = verts_attr[self.vt_to_v_index.repeat(batch_size)]
        face_attr = vt_attr[self.faces_uv.repeat(batch_size, 1)]
        face_attr = face_attr.view(batch_size, self.num_faces, 3, D)
        pix_to_face = self.pix_to_face.unsqueeze(0).repeat(batch_size, 1,
                                                           1).unsqueeze(-1)
        bary_coords = self.bary_coords[None].repeat(batch_size, 1, 1,
                                                    1).unsqueeze(-2)
        return interpolate_face_attributes(
            pix_to_face=pix_to_face.to(self.device),
            barycentric_coords=bary_coords.to(self.device),
            face_attributes=face_attr.to(self.device),
        )

    def resample_from_uv():
        pass

    def warp_normal_map(self):
        pass

    def warp_displacement(self, meshes, displacementmap):
        pass

    def warp_texture_map(self,
                         meshes: Meshes,
                         texture_map: torch.Tensor,
                         resolution: Optional[Iterable[int]] = None,
                         mode: Optional[str] = 'bicubic',
                         is_bgr: bool = True):
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
    def rgb2bgr(rgbs):
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
    def normalize(value, min_value, max_value, dtype):
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

    def tensor2array(self, image):
        """"""
        image = self.normalize(
            image, min_value=0, max_value=255, dtype=np.uint8)
        return image

    def array2tensor(self, image):
        """"""
        image = self.normalize(
            image, min_value=0, max_value=1, dtype=torch.float32)
        return image

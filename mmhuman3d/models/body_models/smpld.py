import os
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesUV
from .smpl import SMPL
from ..builder import BODY_MODELS
import pickle
import torch.nn as nn
import warnings
from mmhuman3d.utils.path_utils import check_input_path

osp = os.path


@BODY_MODELS.register_module(name=['smpld', 'smpl_d', 'smpl+D', 'SmPLD'])
class SMPLD(SMPL):

    def __init__(self,
                 uv_param_path=None,
                 displacement=None,
                 texture_image=None,
                 texture_res: int = None,
                 create_texture: bool = True,
                 create_displacement: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if uv_param_path is not None:
            self.uv_param_path = uv_param_path
            check_input_path(
                uv_param_path,
                allowed_suffix=['pkl', 'pickle'],
                tag='uv parameter file',
                path_type='file')
            with open(uv_param_path, 'rb') as f:
                param_dict = pickle.load(f)
            verts_uv = torch.FloatTensor(param_dict['texcoords'])
            verts_u, verts_v = torch.unbind(verts_uv, -1)
            verts_v_ = 1 - verts_u.unsqueeze(-1)
            verts_u_ = verts_v.unsqueeze(-1)
            verts_uv = torch.cat([verts_u_, verts_v_], -1)
            faces_uv = torch.LongTensor(param_dict['vt_faces'])
            self.register_buffer('verts_uv', verts_uv)
            self.register_buffer('faces_uv', faces_uv)
            assert isinstance(texture_res, int)
        else:
            if create_texture is True:
                warnings.warn(
                    'No uv paramter provided, could not create texture.')
            create_texture = False

        if create_displacement:
            if displacement is None:
                default_displacement = torch.zeros([self.NUM_VERTS, 1],
                                                   dtype=torch.float32)
            else:
                if torch.is_tensor(displacement):
                    default_displacement = displacement.clone().detach()
                else:
                    default_displacement = torch.tensor(
                        displacement, dtype=torch.float32)
            self.register_parameter(
                'displacement',
                nn.Parameter(default_displacement, requires_grad=True))

        if create_texture:
            if texture_image is None:
                default_texture = torch.zeros([texture_res, texture_res, 3],
                                              dtype=torch.float32)
            else:
                if torch.is_tensor(texture_image):
                    default_texture = texture_image.clone().detach()
                else:
                    default_texture = torch.tensor(
                        texture_image, dtype=torch.float32)
            self.register_parameter(
                'texture_image',
                nn.Parameter(default_texture, requires_grad=True))

        _v_template = self.v_template.clone()
        self.register_buffer('_v_template', _v_template)

        mesh_template = Meshes(
            verts=self._v_template[None], faces=self.faces_tensor[None])
        v_normals_template = mesh_template.verts_normals_padded()
        self.register_buffer('v_normals_template', v_normals_template)

    def forward(self, return_mesh=False, return_texture=False, **kwargs):
        device = kwargs.get('body_pose').device
        displacement = kwargs.get('displacement', self.displacement)

        if displacement.ndim == 2:
            displacement = displacement.unsqueeze(0)

        if displacement.shape[-1] == 1:
            displacement = self.v_normals_template * displacement

        self.v_template = self._v_template[None] + displacement

        smpl_output = super().forward(**kwargs)

        if return_mesh:
            verts = smpl_output['vertices']
            batch_size = verts.shape[0]

            if return_texture:
                texture_image = kwargs.get('texture_image', self.texture_image)
                if texture_image.ndim == 3:
                    texture_image = texture_image.unsqueeze(0)
                textures = TexturesUV(
                    maps=texture_image.repeat(batch_size, 1, 1, 1),
                    faces_uvs=self.faces_uv[None].repeat(batch_size, 1, 1),
                    verts_uvs=self.verts_uv[None].repeat(batch_size, 1,
                                                         1)).to(device)
            else:
                textures = None
            meshes = Meshes(
                verts=verts,
                faces=self.faces_tensor[None].repeat(batch_size, 1,
                                                     1).to(device),
                textures=textures).to(device)
            smpl_output['meshes'] = meshes
        return smpl_output

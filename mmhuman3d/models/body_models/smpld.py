import os
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesUV
from .smpl import SMPL
from ..builder import BODY_MODELS
import pickle
from mmhuman3d.utils.path_utils import check_input_path

osp = os.path


@BODY_MODELS.register_module(name=['smpld', 'smpl_d', 'smpl+D', 'SmPLD'])
class SMPLD(SMPL):

    def __init__(self,
                 uv_param_path=None,
                 displacement=None,
                 texture_image=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

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

        displacement = torch.zeros(self.__class__.NUM_VERTS,
                                   3) if displacement is None else displacement

        _v_template = self.v_template.clone()
        # The vertices of the template model
        self.register_buffer('_v_template', _v_template)
        self.texture_image = texture_image
        self.displacement = displacement
        mesh_template = Meshes(
            verts=self._v_template, faces=self.face_tensor[None])
        v_normals_template = mesh_template.verts_normals_padded()
        self.register_buffer('v_normals_template', v_normals_template)

    def forward(self, return_mesh=False, return_texture=False, **kwargs):
        device = kwargs.get('body_pose').device
        displacement = kwargs.get('displacement', self.displacement)
        texture_image = kwargs.get('texture_image', self.texture_image)
        if displacement.shape[-1] == 1:
            displacement = self.v_normals_template * displacement[None]

        self.v_template = self._v_template + displacement

        smpl_output = super().forward(**kwargs)

        if return_mesh:
            verts = smpl_output['vertices']
            batch_size = verts.shape[0]

            if return_texture and isinstance(texture_image, torch.Tensor):
                textures = TexturesUV(
                    maps=texture_image[None].repeat(batch_size, 1, 1, 1),
                    faces_uvs=self.faces_uv[None].repeat(batch_size, 1, 1),
                    verts_uvs=self.verts_uv[None].repeat(batch_size, 1,
                                                         1)).to(device)
            else:
                textures = None
            meshes = Meshes(
                verts=verts,
                faces=self.face_tensor[None].repeat(batch_size, 1,
                                                    1).to(device),
                textures=textures).to(device)
            smpl_output['meshes'] = meshes
        return smpl_output

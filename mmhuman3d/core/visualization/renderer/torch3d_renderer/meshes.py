from typing import Iterable, List, Union

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes

from mmhuman3d.models.body_models import SMPL, SMPLX
from .builder import build_textures


class ParametricMeshes(Meshes):
    """Mesh structure for parametric body models, E.g., smpl, smplx, mano,
    flame."""

    def __init__(self,
                 verts=None,
                 faces=None,
                 textures=None,
                 body_model: Union[nn.Module, dict] = None,
                 vertex_color: Union[Iterable[float], torch.Tensor,
                                     np.ndarray] = ((1, 1, 1), ),
                 use_nearest: bool = False,
                 model_type: str = 'smpl',
                 texture_image: Union[torch.Tensor, None] = None,
                 *,
                 verts_normals=None,
                 **pose_params) -> None:
        self.body_model = body_model

        model_type = body_model._get_name().lower(
        ) if body_model is not None else model_type
        self.model_type = model_type

        # More model class to be added
        self.model_class = {'smpl': SMPL, 'smplx': SMPLX}[model_type]

        if verts is None:
            verts = body_model(**pose_params)['vertices']
        elif isinstance(verts, np.ndarray):
            verts = torch.Tensor(np.ndarray)

        assert verts.shape[-2] // self.model_class.NUM_VERTS == 0
        num_individual = int(verts.shape[-2] // self.model_class.NUM_VERTS)
        self.num_individual = num_individual
        verts = verts.view(-1, self.model_class.NUM_VERTS, 3)
        device = verts.device

        N, V, _ = verts.shape
        if faces is None:
            self.face_individual = body_model.faces_tensor[None].to(device)
            faces = self.get_faces_padded(N, num_individual)
        else:
            self.face_individual = faces[0:1, :self.model_class.NUM_FACES].to(
                device)
        assert faces.shape == (N, self.model_class.NUM_FACES * num_individual,
                               3)

        if textures is None:
            if texture_image is None:
                if isinstance(vertex_color, (tuple, list)):
                    vertex_color = torch.Tensor(vertex_color).view(1, 1,
                                                                   3).repeat(
                                                                       N, V, 1)
                elif isinstance(vertex_color, (torch.Tensor, np.ndarray)):
                    vertex_color = torch.Tensor(vertex_color)
                    if vertex_color.numel() == 3:
                        vertex_color = vertex_color.view(1, 1,
                                                         3).repeat(N, V, 1)
                    else:
                        if vertex_color.shape[0] == 1:
                            vertex_color = vertex_color.repeat(N, 1, 1)
                        if vertex_color.shape[1] == 1:
                            vertex_color = vertex_color.repeat(1, V, 1)
                if use_nearest:
                    textures = build_textures(
                        dict(type='nearest',
                             verts_features=vertex_color)).to(device)
                else:
                    textures = build_textures(
                        dict(type='vertex',
                             verts_features=vertex_color)).to(device)
            else:
                assert body_model.uv_renderer is not None
                textures = body_model.uv_renderer.wrap_texture(texture_image)

        super().__init__(
            verts=verts,
            faces=faces,
            textures=textures,
            verts_normals=verts_normals,
        )

    def get_faces_padded(self, num_batch, num_individual):
        faces = self.face_individual.repeat(num_batch, num_individual, 1)
        faces_offset = torch.arange(num_individual).view(
            num_individual,
            1).repeat(1, self.model_class.NUM_FACES).view(1, -1, 1)
        faces += faces_offset
        return faces

    # @staticmethod
    # def join_meshes_as_scene():

    # @staticmethod
    # def join_batch_meshes_as_scene():
    #     pass

    def __getitem__(self, batch_index, individual_index):
        mesh_selected_by_batch = super().__getitem__(batch_index)
        if isinstance(individual_index, int):
            person_index = [individual_index]
        vertex_index = [
            torch.arange(self.model_class.NUM_VERTS) * idx
            for idx in person_index
        ]
        vertex_index = torch.cat(vertex_index).to(self.device)
        verts_padded = mesh_selected_by_batch.verts_padded()[:, vertex_index]
        faces_padded = self.get_faces_padded(
            len(verts_padded), len(person_index))
        # TODO: textures
        return self.__class__(verts=verts_padded, faces=faces_padded)

    def wrap_texture(self, texture_image):
        assert self.body_model.uv_renderer is not None
        for person_index in range(self.num_individual):
            self[:,
                 person_index].textures = self.body_model.uv_renderer.\
                     wrap_texture(texture_image)

    def split(self, ) -> List[Meshes]:
        return [self[:, index] for index in range(self.num_individual)]

    @property
    def shape(self, ):
        return (len(self), self.num_individual)

    @classmethod
    def join_meshes_as_batch(cls):
        cls.__init__()

    @classmethod
    def join_meshes_as_scene():
        pass

    @classmethod
    def join_batch_meshes_as_scene():
        pass

from typing import Iterable, List, Union

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes, list_to_padded, padded_to_list

from mmhuman3d.models.body_models import SMPL, SMPLX
from mmhuman3d.utils.mesh_utils import \
    join_meshes_as_batch as _join_meshes_as_batch
from mmhuman3d.utils.mesh_utils import \
    join_meshes_as_scene as _join_meshes_as_scene
from .textures import TexturesNearest


class ParametricMeshes(Meshes):
    # More model class to be added
    model_classes = {'smpl': SMPL, 'smplx': SMPLX}
    """Mesh structure for parametric body models, E.g., smpl, smplx, mano,
    flame."""

    def __init__(self,
                 verts=None,
                 faces=None,
                 textures=None,
                 meshes: Meshes = None,
                 body_model: Union[nn.Module, dict] = None,
                 vertex_color: Union[Iterable[float], torch.Tensor,
                                     np.ndarray] = ((1, 1, 1), ),
                 use_nearest: bool = False,
                 model_type: str = 'smpl',
                 texture_images: Union[torch.Tensor, List[torch.Tensor],
                                       None] = None,
                 *,
                 verts_normals=None,
                 **pose_params) -> None:
        if isinstance(meshes, Meshes):
            verts = meshes.verts_padded()
            faces = meshes.faces_padded()
            textures = meshes.textures

        self.body_model = body_model

        model_type = body_model._get_name().lower(
        ) if body_model is not None else model_type
        self.model_type = model_type

        self.model_class = self.model_classes[model_type]

        if verts is None:
            verts = body_model(**pose_params)['vertices']
        elif isinstance(verts, np.ndarray):
            verts = torch.Tensor(np.ndarray)
        elif isinstance(verts, list):
            verts = list_to_padded(verts)

        num_individual = int(verts.shape[-2] // self.model_class.NUM_VERTS)
        self.num_individual = num_individual
        verts = verts.view(-1,
                           self.model_class.NUM_VERTS * self.num_individual, 3)
        device = verts.device
        N, V, _ = verts.shape
        assert V % self.model_class.NUM_VERTS == 0

        if isinstance(faces, list):
            self.face_individual = faces[0][:self.model_class.NUM_FACES].to(
                device)
            faces = self.get_faces_padded(N, self.num_individual)

        if faces is None:
            self.face_individual = body_model.faces_tensor[None].to(device)
            faces = self.get_faces_padded(N, num_individual)
        else:
            self.face_individual = faces[0:1, :self.model_class.NUM_FACES].to(
                device)

        assert faces.shape == (N, self.model_class.NUM_FACES * num_individual,
                               3)

        if textures is None:
            if texture_images is None:
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
                    textures = TexturesNearest(
                        verts_features=vertex_color).to(device)
                else:
                    textures = TexturesVertex(
                        verts_features=vertex_color).to(device)
            else:
                if isinstance(texture_images, torch.Tensor):
                    if texture_images.ndim == 3:
                        texture_images = texture_images[None]
                    if texture_images.shape[0] == 1:
                        texture_images = texture_images.repeat(
                            self.num_individual, 1, 1, 1)
                elif isinstance(texture_images, Iterable[torch.Tensor]):

                    assert len(texture_images) in (
                        1, self.num_individual
                    ), 'Number of texture images should be'
                    '1 or same as num individuals.'
                    texture_images = list_to_padded(texture_images)
                assert body_model.uv_renderer is not None

                textures = body_model.uv_renderer.wrap_texture(texture_images)
                textures = textures.join_scene()
                textures = textures.extend(N)

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
            1).repeat(1, self.model_class.NUM_FACES).view(1, -1,
                                                          1).to(faces.device)
        faces += faces_offset
        return faces

    def __getitem__(self, index):
        if isinstance(index, tuple):
            batch_index, individual_index = index
        else:
            batch_index, individual_index = index, None
        mesh_selected_by_batch = super().__getitem__(batch_index)
        if individual_index is None:
            return self.__class__(
                meshes=mesh_selected_by_batch,
                model_type=self.model_type,
                body_model=self.body_model)

        if isinstance(individual_index, int):
            individual_index = [individual_index]
        if isinstance(individual_index, (tuple, list, slice)):
            individual_index = torch.arange(
                self.num_individual)[individual_index]

        if (individual_index > self.num_individual).any():
            raise (IndexError, 'list index out of range')
        vertex_index = [
            torch.arange(self.model_class.NUM_VERTS) +
            idx * self.model_class.NUM_VERTS for idx in individual_index
        ]
        vertex_index = torch.cat(vertex_index).to(self.device).long()

        face_index = [
            torch.arange(self.model_class.NUM_FACES) +
            idx * self.model_class.NUM_FACES for idx in individual_index
        ]
        face_index = torch.cat(face_index).to(self.device).long()

        verts_padded = mesh_selected_by_batch.verts_padded()[:, vertex_index]
        faces_padded = self.get_faces_padded(
            len(verts_padded), len(individual_index))
        # TODO: eval textures
        textures_batch = mesh_selected_by_batch.textures

        if isinstance(textures_batch, TexturesUV):
            maps = textures_batch.maps_padded()
            width_individual = maps.shape[-2] // self.num_individual
            maps_index = [
                torch.arange(width_individual) * idx
                for idx in individual_index
            ]
            maps_index = torch.cat(maps_index).to(self.device)
            verts_uvs_padded = textures_batch.verts_uvs_padded(
            )[:, :len(face_index)]
            faces_uvs_padded = textures_batch.faces_uvs_padded(
            )[:, :len(face_index)]
            maps_padded = maps[:, :, maps_index]
            textures = TexturesUV(
                faces_uvs=faces_uvs_padded,
                verts_uvs=verts_uvs_padded,
                maps=maps_padded)
        elif isinstance(textures_batch, (TexturesVertex, TexturesNearest)):
            verts_features_padded = textures_batch.verts_features_padded(
            )[:, vertex_index]
            textures = textures_batch.__class__(verts_features_padded)
        return self.__class__(
            verts=verts_padded,
            faces=faces_padded,
            textures=textures,
            model_type=self.model_type,
            body_model=self.body_model)

    @property
    def shape(self, ):
        return (len(self), self.num_individual)


def join_meshes_as_batch(meshes: List[ParametricMeshes],
                         include_textures: bool = True):

    if isinstance(meshes, ParametricMeshes):
        raise ValueError('Wrong first argument to join_meshes_as_batch.')
    first = meshes[0]
    assert all(mesh.model_type == first.model_type for mesh in meshes), \
        'model_type should all be the same'

    meshes = _join_meshes_as_batch(meshes, include_textures=include_textures)
    return ParametricMeshes(
        model_type=getattr(first, 'model_type', None),
        body_model=getattr(first, 'body_model', None),
        meshes=meshes)


def join_meshes_as_scene(meshes: Union[ParametricMeshes,
                                       List[ParametricMeshes]],
                         include_textures: bool = True):

    first = meshes[0]
    assert all(mesh.model_type == first.model_type for mesh in meshes), \
        'model_type should all be the same'

    meshes = _join_meshes_as_scene(meshes, include_textures=include_textures)
    return ParametricMeshes(
        model_type=getattr(first, 'model_type', None),
        body_model=getattr(first, 'body_model', None),
        meshes=meshes)


def join_batch_mesh_as_scene(
    meshes: List[ParametricMeshes],
    include_textures: bool = True,
):
    first = meshes[0]
    assert all(mesh.model_type == first.model_type for mesh in meshes), \
        'model_type should all be the same'
    assert all(mesh.num_individual == first.num_individual for mesh in meshes)
    assert all(len(mesh) == len(first) for mesh in meshes)
    for mesh in meshes:
        mesh._verts_list = padded_to_list(mesh.verts_padded(),
                                          mesh.num_verts_per_mesh().tolist())
    num_scene_size = len(meshes)
    num_batch_size = len(meshes[0])

    meshes_all = []
    for j in range(num_batch_size):
        meshes_batch = []
        for i in range(num_scene_size):
            meshes_batch.append(meshes[i][j])
        meshes_all.append(join_meshes_as_scene(meshes_batch, include_textures))
    meshes_final = join_meshes_as_batch(meshes_all, include_textures)

    return meshes_final

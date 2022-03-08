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
from .utils import align_input_to_padded


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
                 N_individual_overdide: int = None,
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

        self.model_type = body_model._get_name().lower(
        ) if body_model is not None else model_type

        self.model_class = self.model_classes[self.model_type]

        use_list = False
        # formart verts as verts_padded: (N, V, 3)
        if verts is None:
            assert self.body_model is not None
            verts = self.body_model(**pose_params)['vertices']

        elif isinstance(verts, np.ndarray):
            verts = torch.Tensor(np.ndarray)
        elif isinstance(verts, list):
            verts = list_to_padded(verts)
            use_list = True
        if N_individual_overdide is not None:
            verts = verts.view(
                -1, self.model_class.NUM_VERTS * N_individual_overdide, 3)

        self._N_individual = int(verts.shape[-2] // self.model_class.NUM_VERTS)

        verts = verts.view(-1, self.model_class.NUM_VERTS * self._N_individual,
                           3)
        device = verts.device
        N, V, _ = verts.shape
        assert V % self.model_class.NUM_VERTS == 0

        # formart faces as faces_padded: (N, F, 3)
        if isinstance(faces, list):
            self.face_individual = faces[0][:self.model_class.NUM_FACES].to(
                device)
            faces = self.get_faces_padded(N, self._N_individual)
        elif faces is None:
            self.face_individual = body_model.faces_tensor[None].to(device)
            faces = self.get_faces_padded(N, self._N_individual)
        elif isinstance(faces, torch.Tensor):
            self.face_individual = faces[:1, :self.model_class.NUM_FACES].to(
                device)
            faces = self.get_faces_padded(N, self._N_individual)
        else:
            raise ValueError(f'Wrong type of faces: {type(faces)}.')

        assert faces.shape == (N,
                               self.model_class.NUM_FACES * self._N_individual,
                               3)
        F = faces.shape[1]
        if textures is None:
            if texture_images is None:
                if isinstance(vertex_color, (tuple, list)):
                    vertex_color = torch.Tensor(vertex_color).view(1, 1,
                                                                   3).repeat(
                                                                       N, V, 1)
                elif isinstance(vertex_color, (torch.Tensor, np.ndarray)):
                    vertex_color = torch.Tensor(vertex_color) if isinstance(
                        vertex_color, np.ndarray) else vertex_color
                    if vertex_color.numel() == 3:
                        vertex_color = vertex_color.view(1, 1,
                                                         3).repeat(N, V, 1)
                    elif vertex_color.shape[-2] == 1:
                        vertex_color = vertex_color.repeat_interleave(V, -2)
                vertex_color = align_input_to_padded(
                    vertex_color, ndim=3, batch_size=N)
                assert vertex_color.shape == verts.shape
                if use_nearest:
                    textures = TexturesNearest(
                        verts_features=vertex_color).to(device)
                else:
                    textures = TexturesVertex(
                        verts_features=vertex_color).to(device)
            else:

                texture_images = align_input_to_padded(
                    texture_images, ndim=4, batch_size=N)

                assert self.body_model.uv_renderer is not None

                textures = self.body_model.uv_renderer.wrap_texture(
                    texture_images).to(device)
                textures = textures.join_scene()
                textures = textures.extend(N)
        num_verts_per_mesh = [V for _ in range(N)]
        num_faces_per_mesh = [F for _ in range(N)]

        if use_list:
            verts = padded_to_list(verts, num_verts_per_mesh)
            faces = padded_to_list(faces, num_faces_per_mesh)
        super().__init__(
            verts=verts,
            faces=faces,
            textures=textures,
            verts_normals=verts_normals,
        )

    def get_faces_padded(self, N_batch, N_individual):
        faces = self.face_individual.repeat(N_batch, N_individual, 1)
        faces_offset = torch.arange(N_individual).view(N_individual, 1).repeat(
            1, self.model_class.NUM_FACES).view(1, -1, 1).to(faces.device)
        faces += faces_offset
        return faces

    def _compute_list(self):
        self._faces_list = self.faces_list()
        self._verts_list = self.verts_list()

    def extend(self, N_batch: int, N_scene: int = 1):
        if N_batch != 1:
            meshes = join_meshes_as_batch([self for _ in range(N_batch)])
        if N_scene != 1:
            meshes = join_batch_meshes_as_scene([self for _ in range(N_scene)])
        return meshes

    def clone(self):
        verts_list = self.verts_list()
        faces_list = self.faces_list()
        new_verts_list = [v.clone() for v in verts_list]
        new_faces_list = [f.clone() for f in faces_list]
        other = self.__class__(
            verts=new_verts_list,
            faces=new_faces_list,
            model_type=self.model_type,
            body_model=self.body_model)
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())

        # Textures is not a tensor but has a clone method
        if self.textures is not None:
            other.textures = self.textures.clone()
        return other

    def detach(self):
        verts_list = self.verts_list()
        faces_list = self.faces_list()
        new_verts_list = [v.detach() for v in verts_list]
        new_faces_list = [f.detach() for f in faces_list]
        other = self.__class__(
            verts=new_verts_list,
            faces=new_faces_list,
            model_type=self.model_type,
            body_model=self.body_model)

        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.detach())

        # Textures is not a tensor but has a detach method
        if self.textures is not None:
            other.textures = self.textures.detach()
        return other

    def update_padded(self, new_verts_padded):

        def check_shapes(x, size):
            if x.shape[0] != size[0]:
                raise ValueError(
                    'new values must have the same batch dimension.')
            if x.shape[1] != size[1]:
                raise ValueError(
                    'new values must have the same number of points.')
            if x.shape[2] != size[2]:
                raise ValueError('new values must have the same dimension.')

        check_shapes(new_verts_padded, [self._N, self._V, 3])

        new = self.__class__(
            verts=new_verts_padded,
            faces=self.faces_padded(),
            model_type=self.model_type,
            body_model=self.body_model)

        if new._N != self._N or new._V != self._V or new._F != self._F:
            raise ValueError('Inconsistent sizes after construction.')

        # overwrite the equisized flag
        new.equisized = self.equisized

        # overwrite textures if any
        new.textures = self.textures

        # copy auxiliary tensors
        copy_tensors = ['_num_verts_per_mesh', '_num_faces_per_mesh', 'valid']

        for k in copy_tensors:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(new, k, v)  # shallow copy

        # shallow copy of faces_list if any, st new.faces_list()
        # does not re-compute from _faces_padded
        new._faces_list = self._faces_list

        # update verts/faces packed if they are computed in self
        if self._verts_packed is not None:
            copy_tensors = [
                '_faces_packed',
                '_verts_packed_to_mesh_idx',
                '_faces_packed_to_mesh_idx',
                '_mesh_to_verts_packed_first_idx',
                '_mesh_to_faces_packed_first_idx',
            ]
            for k in copy_tensors:
                v = getattr(self, k)
                assert torch.is_tensor(v)
                setattr(new, k, v)  # shallow copy
            # update verts_packed
            pad_to_packed = self.verts_padded_to_packed_idx()
            new_verts_packed = new_verts_padded.reshape(-1,
                                                        3)[pad_to_packed, :]
            new._verts_packed = new_verts_packed
            new._verts_padded_to_packed_idx = pad_to_packed

        # update edges packed if they are computed in self
        if self._edges_packed is not None:
            copy_tensors = [
                '_edges_packed',
                '_edges_packed_to_mesh_idx',
                '_mesh_to_edges_packed_first_idx',
                '_faces_packed_to_edges_packed',
                '_num_edges_per_mesh',
            ]
            for k in copy_tensors:
                v = getattr(self, k)
                assert torch.is_tensor(v)
                setattr(new, k, v)  # shallow copy

        # update laplacian if it is compute in self
        if self._laplacian_packed is not None:
            new._laplacian_packed = self._laplacian_packed

        assert new._verts_list is None
        assert new._verts_normals_packed is None
        assert new._faces_normals_packed is None
        assert new._faces_areas_packed is None

        return new

    def __getitem__(self, index):
        if isinstance(index, tuple):
            batch_index, individual_index = index
        else:
            batch_index, individual_index = index, None

        if isinstance(batch_index, int):
            batch_index = [batch_index]
        elif isinstance(batch_index, (tuple, list, slice)):
            batch_index = torch.arange(self._N)[batch_index]
        batch_index = torch.tensor(batch_index).to(self.device)
        batch_index = batch_index.long() if not (
            batch_index.dtype is torch.long) else batch_index
        if (batch_index > self._N).any():
            raise (IndexError, 'list index out of range')

        if individual_index is None:
            return self.__class__(
                verts=self.verts_padded()[batch_index],
                faces=self.faces_padded()[batch_index],
                textures=self.textures[batch_index]
                if self.textures is not None else None,
                model_type=self.model_type,
                body_model=self.body_model)

        if isinstance(individual_index, int):
            individual_index = [individual_index]
        elif isinstance(individual_index, (tuple, list, slice)):
            individual_index = torch.arange(
                self._N_individual)[individual_index]
        individual_index = torch.tensor(individual_index)
        if (individual_index > self._N_individual).any():
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

        verts_padded = self.verts_padded()[batch_index][:, vertex_index]
        faces_padded = self.get_faces_padded(
            len(verts_padded), len(individual_index))
        # TODO: eval textures
        textures_batch = self.textures[batch_index]

        if isinstance(textures_batch, TexturesUV):
            maps = textures_batch.maps_padded()
            width_individual = maps.shape[-2] // self._N_individual
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
        meshes = self.__class__(
            verts=verts_padded,
            faces=faces_padded,
            textures=textures,
            model_type=self.model_type,
            body_model=self.body_model)
        return meshes

    @property
    def shape(self, ):
        return (len(self), self._N_individual)


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


def join_batch_meshes_as_scene(
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

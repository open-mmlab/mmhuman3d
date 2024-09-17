from typing import Iterable, List, Union

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import TexturesUV, TexturesVertex
from pytorch3d.renderer.mesh.textures import TexturesBase
from pytorch3d.structures import Meshes, list_to_padded, padded_to_list

from mmhuman3d.models.body_models.builder import SMPL, SMPLX, STAR
from mmhuman3d.utils.mesh_utils import \
    join_meshes_as_batch as _join_meshes_as_batch
from .builder import build_renderer
from .textures.textures import TexturesNearest
from .utils import align_input_to_padded


class ParametricMeshes(Meshes):
    """Mesh structure for parametric body models, E.g., smpl, smplx, mano,
    flame.

    There are 3 ways to initialize the verts:
        1): Pass the verts directly as verts_padded (N, V, 3) or verts_list
        (list of (N, 3)).
        2): Pass body_model and pose_params.
        3): Pass meshes. Could be Meshes or ParametricMeshes.
            Will use the verts from the meshes.
    There are 3 ways to initialize the faces:
        1): Pass the faces directly as faces_padded (N, F, 3) or faces_list
        (list of (F, 3)).
        2): Pass body_model and will use body_model.faces_tensor.
        3): Pass meshes. Could be Meshes or ParametricMeshes.
        Will use the faces from the meshes.
    There are 4 ways to initialize the textures.
        1): Pass the textures directly.
        2): Pass the texture_images of shape (H, W, 3) for single person or
        (_N_individual, H, W, 3) for multi-person. `body_model` should be
        passed and should has `uv_renderer`.
        3): Pass the vertex_color of shape (3) or (V, 3) or (N, V, 3).
        4): Pass meshes. Could be Meshes or ParametricMeshes.
        Will use the textures directly from the meshes.
    """
    # TODO: More model class to be added (FLAME, MANO)
    MODEL_CLASSES = {'smpl': SMPL, 'smplx': SMPLX, 'star': STAR}

    def __init__(self,
                 verts: Union[List[torch.Tensor], torch.Tensor] = None,
                 faces: Union[List[torch.Tensor], torch.Tensor] = None,
                 textures: TexturesBase = None,
                 meshes: Meshes = None,
                 body_model: Union[nn.Module, dict] = None,
                 uv_renderer: Union[nn.Module, dict] = None,
                 vertex_color: Union[Iterable[float], torch.Tensor,
                                     np.ndarray] = ((1, 1, 1), ),
                 use_nearest: bool = False,
                 texture_images: Union[torch.Tensor, List[torch.Tensor],
                                       None] = None,
                 model_type: str = 'smpl',
                 N_individual_override: int = None,
                 *,
                 verts_normals: torch.Tensor = None,
                 **pose_params) -> None:

        if isinstance(meshes, Meshes):
            verts = meshes.verts_padded()
            faces = meshes.faces_padded()
            textures = meshes.textures

        self.model_type = body_model._get_name().lower(
        ) if body_model is not None else model_type

        self.model_class = self.MODEL_CLASSES[self.model_type]

        use_list = False

        # formart verts as verts_padded: (N, V, 3)
        if verts is None:
            assert body_model is not None
            verts = body_model(**pose_params)['vertices']
        elif isinstance(verts, list):
            verts = list_to_padded(verts)
            use_list = True
        # specify number of individuals
        if N_individual_override is not None:
            verts = verts.view(
                -1, self.model_class.NUM_VERTS * N_individual_override, 3)

        # the information of _N_individual should be revealed in verts's shape
        self._N_individual = int(verts.shape[-2] // self.model_class.NUM_VERTS)

        assert verts.shape[1] % self.model_class.NUM_VERTS == 0
        verts = verts.view(-1, self.model_class.NUM_VERTS * self._N_individual,
                           3)
        device = verts.device
        N, V, _ = verts.shape

        # formart faces as faces_padded: (N, F, 3)
        if isinstance(faces, list):
            faces = list_to_padded(faces)
            self.face_individual = faces[0][:self.model_class.NUM_FACES].to(
                device)
        elif faces is None:
            assert body_model is not None
            self.face_individual = body_model.faces_tensor[None].to(device)
            faces = self.get_faces_padded(N, self._N_individual)
        elif isinstance(faces, torch.Tensor):
            faces = align_input_to_padded(faces, ndim=3, batch_size=N)
            self.face_individual = faces[:1, :self.model_class.NUM_FACES].to(
                device)
        else:
            raise ValueError(f'Wrong type of faces: {type(faces)}.')

        assert faces.shape == (N,
                               self.model_class.NUM_FACES * self._N_individual,
                               3)
        F = faces.shape[1]
        if textures is None:
            if texture_images is None:
                # input vertex_color should be
                #   (3), (1, 3), (1, 1, 3). all the same color
                #   (V, 3), (1, V, 3), each vertex has a single color
                #   (N, V, 3), each batch each vertex has a single color
                if isinstance(vertex_color, (tuple, list)):
                    vertex_color = torch.Tensor(vertex_color)
                elif isinstance(vertex_color, np.ndarray):
                    vertex_color = torch.from_numpy(vertex_color)
                if vertex_color.numel() == 3:
                    vertex_color = vertex_color.view(1, 3).repeat(V, 1)
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
                    texture_images, ndim=4, batch_size=N).to(device)

                assert uv_renderer is not None
                if isinstance(uv_renderer, dict):
                    uv_renderer = build_renderer(uv_renderer)
                uv_renderer = uv_renderer.to(device)
                textures = uv_renderer.wrap_texture(texture_images).to(device)
                if self._N_individual > 1:
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
        faces = faces + faces_offset * self.model_class.NUM_VERTS
        return faces

    def _compute_list(self):
        self._faces_list = self.faces_list()
        self._verts_list = self.verts_list()

    def extend(self, N_batch: int, N_scene: int = 1):
        if N_batch == 1:
            meshes_batch = self
        else:
            meshes_batch = join_meshes_as_batch([self for _ in range(N_batch)])

        if N_scene == 1:
            meshes = meshes_batch
        else:
            meshes = join_batch_meshes_as_scene(
                [meshes_batch for _ in range(N_scene)])
        return meshes

    def clone(self):
        """Modified from pytorch3d and add `model_type` in
        __class__.__init__."""
        verts_list = self.verts_list()
        faces_list = self.faces_list()
        new_verts_list = [v.clone() for v in verts_list]
        new_faces_list = [f.clone() for f in faces_list]
        other = self.__class__(
            verts=new_verts_list,
            faces=new_faces_list,
            model_type=self.model_type)
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())

        # Textures is not a tensor but has a clone method
        if self.textures is not None:
            other.textures = self.textures.clone()
        return other

    def detach(self):
        """Modified from pytorch3d and add `model_type` in
        __class__.__init__."""
        verts_list = self.verts_list()
        faces_list = self.faces_list()
        new_verts_list = [v.detach() for v in verts_list]
        new_faces_list = [f.detach() for f in faces_list]
        other = self.__class__(
            verts=new_verts_list,
            faces=new_faces_list,
            model_type=self.model_type)

        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.detach())

        # Textures is not a tensor but has a detach method
        if self.textures is not None:
            other.textures = self.textures.detach()
        return other

    def update_padded(self, new_verts_padded: torch.Tensor):
        """Modified from pytorch3d and add `model_type` in
        __class__.__init__."""

        def check_shapes(x, size):
            if x.shape[0] != size[0]:
                raise ValueError('new values must have the same batch size.')
            if x.shape[1] != size[1]:
                raise ValueError(
                    'new values must have the same number of points.')
            if x.shape[2] != size[2]:
                raise ValueError('new values must have the same dimension.')

        check_shapes(new_verts_padded, [self._N, self._V, 3])

        new = self.__class__(
            verts=new_verts_padded,
            faces=self.faces_padded(),
            model_type=self.model_type)

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

    def __getitem__(self, index: Union[tuple, int, list, slice, torch.Tensor]):
        """Slice the meshes by the batch dim like pytorch3d Meshes. And slice
        by scene dim due to the topology of the parametric meshes.

        Args:
            index (Union[tuple, int, list, slice, torch.Tensor]): indexes, if
            pass only one augment, will ignore the scene dim.
        """
        if isinstance(index, tuple):
            batch_index, individual_index = index
        else:
            batch_index, individual_index = index, None

        if isinstance(batch_index, int):
            batch_index = [batch_index]
        elif isinstance(batch_index, (tuple, list, slice)):
            batch_index = torch.arange(self._N)[batch_index]
        batch_index = torch.tensor(batch_index) if not isinstance(
            batch_index, torch.Tensor) else batch_index
        batch_index = batch_index.to(self.device, dtype=torch.long)

        if (batch_index >= self._N).any():
            raise IndexError('list index out of range')

        if individual_index is None:
            return self.__class__(
                verts=self.verts_padded()[batch_index],
                faces=self.faces_padded()[batch_index],
                textures=self.textures[batch_index]
                if self.textures is not None else None,
                model_type=self.model_type)

        if isinstance(individual_index, int):
            individual_index = [individual_index]
        elif isinstance(individual_index, (tuple, list, slice)):
            individual_index = torch.arange(
                self._N_individual)[individual_index]
        individual_index = torch.tensor(individual_index) if not isinstance(
            individual_index, torch.Tensor) else individual_index
        if (individual_index > self._N_individual).any():
            raise IndexError('list index out of range')
        vertex_index = [
            torch.arange(self.model_class.NUM_VERTS) +
            idx * self.model_class.NUM_VERTS for idx in individual_index
        ]
        vertex_index = torch.cat(vertex_index).to(self.device).long()

        new_face_num = self.model_class.NUM_FACES * len(individual_index)

        verts_padded = self.verts_padded()[batch_index][:, vertex_index]
        faces_padded = self.get_faces_padded(
            len(verts_padded), len(individual_index))

        textures_batch = self.textures[batch_index]

        if isinstance(textures_batch, TexturesUV):
            # TODO: there is still some problem with `TexturesUV`
            # slice and need to fix the function `join_meshes_as_scene`.
            # It is recommended that we re-inplement the `TexturesUV`
            # as `ParametricTexturesUV`, mainly for the `__getitem__`
            # and `join_scene` functions.

            # textures_batch.get('unique_map_index ')

            # This version only consider the maps tensor as different id.
            maps = textures_batch.maps_padded()
            width_individual = maps.shape[-2] // self._N_individual
            maps_index = [
                torch.arange(width_individual * idx,
                             width_individual * (idx + 1))
                for idx in individual_index
            ]
            maps_index = torch.cat(maps_index).to(self.device)
            verts_uvs_padded = textures_batch.verts_uvs_padded(
            )[:, :len(vertex_index)] * torch.Tensor([
                self._N_individual / len(individual_index), 1
            ]).view(1, 1, 2).to(self.device)
            faces_uvs_padded = textures_batch.faces_uvs_padded(
            )[:, :new_face_num]
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
            model_type=self.model_type)
        return meshes

    @property
    def shape(self, ):
        return (len(self), self._N_individual)


def join_meshes_as_batch(meshes: List[ParametricMeshes],
                         include_textures: bool = True) -> ParametricMeshes:
    """Join the meshes along the batch dim.

    Args:
        meshes (Union[ParametricMeshes, List[ParametricMeshes, Meshes,
            List[Meshes]]]): Meshes object that contains a batch of meshes,
            or a list of Meshes objects.
        include_textures (bool, optional): whether to try to join the textures.
            Defaults to True.

    Returns:
        ParametricMeshes: the joined ParametricMeshes.
    """
    if isinstance(meshes, ParametricMeshes):
        raise ValueError('Wrong first argument to join_meshes_as_batch.')
    first = meshes[0]

    assert all(mesh.model_type == first.model_type
               for mesh in meshes), 'model_type should all be the same.'

    meshes = _join_meshes_as_batch(meshes, include_textures=include_textures)
    return ParametricMeshes(model_type=first.model_type, meshes=meshes)


def join_meshes_as_scene(meshes: Union[ParametricMeshes,
                                       List[ParametricMeshes]],
                         include_textures: bool = True) -> ParametricMeshes:
    """Join the meshes along the scene dim.

    Args:
        meshes (Union[ParametricMeshes, List[ParametricMeshes]]):
            ParametricMeshes object that contains a batch of meshes,
            or a list of ParametricMeshes objects.
        include_textures (bool, optional): whether to try to join the textures.
            Defaults to True.

    Returns:
        ParametricMeshes: the joined ParametricMeshes.
    """
    first = meshes[0]
    assert all(mesh.model_type == first.model_type
               for mesh in meshes), 'model_type should all be the same.'

    if isinstance(meshes, List):
        meshes = join_meshes_as_batch(
            meshes, include_textures=include_textures)

    if len(meshes) == 1:
        return meshes
    verts = meshes.verts_packed()  # (sum(V_n), 3)
    # Offset automatically done by faces_packed
    faces = meshes.faces_packed()  # (sum(F_n), 3)
    textures = None

    if include_textures and meshes.textures is not None:
        textures = meshes.textures.join_scene()

    mesh = ParametricMeshes(
        verts=verts.unsqueeze(0),
        faces=faces.unsqueeze(0),
        textures=textures,
        model_type=first.model_type)

    return mesh


def join_batch_meshes_as_scene(
        meshes: List[ParametricMeshes],
        include_textures: bool = True) -> ParametricMeshes:
    """Join `meshes` as a scene each batch. For ParametricMeshes. The Meshes
    must share the same batch size, and topology could be different. They must
    all be on the same device. If `include_textures` is true, the textures
    should be the same type, all be None is not accepted. If `include_textures`
    is False, textures are ignored. The return meshes will have no textures.

    Args:
        meshes (List[ParametricMeshes]): Meshes object that contains a list of
            Meshes objects.
        include_textures (bool, optional): whether to try to join the textures.
            Defaults to True.


    Returns:
        New Meshes which has join different Meshes by each batch.
    """
    first = meshes[0]

    assert all(mesh.model_type == first.model_type
               for mesh in meshes), 'model_type should all be the same.'

    assert all(len(mesh) == len(first) for mesh in meshes)
    if not all(mesh.shape[1] == first.shape[1] for mesh in meshes):
        meshes_temp = []
        for mesh_scene in meshes:
            meshes_temp.extend([
                mesh_scene[:, individual_index]
                for individual_index in range(mesh_scene._N_individual)
            ])
        meshes = meshes_temp
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

import warnings
from typing import List, Optional, Union

import torch
from pytorch3d.io import IO
from pytorch3d.io import load_objs_as_meshes as _load_objs_as_meshes
from pytorch3d.io import save_obj
from pytorch3d.renderer import TexturesUV, TexturesVertex
from pytorch3d.structures import (
    Meshes,
    Pointclouds,
    join_meshes_as_batch,
    join_meshes_as_scene,
    padded_to_list,
)

from .path_utils import prepare_output_path


def join_batch_meshes_as_scene(
    meshes: List[Meshes],
    include_textures: bool = True,
) -> Meshes:
    """Join `meshes` as a scene each batch. Only for Pytorch3D `meshes`. The
    Meshes must share the same batch size, and topology could be different.
    They must all be on the same device. If `include_textures` is true, the
    textures should be the same type, all be None is not accepted. If
    `include_textures` is False, textures are ignored. The return meshes will
    have no textures.

    Args:
        meshes (List[Meshes]): A `list` of `Meshes` with the same batches.
            Required.
        include_textures: (bool) whether to try to join the textures.

    Returns:
        New Meshes which has join different Meshes by each batch.
    """
    for mesh in meshes:
        mesh._verts_list = padded_to_list(mesh.verts_padded(),
                                          mesh.num_verts_per_mesh().tolist())
    num_scene_size = len(meshes)
    num_batch_size = len(meshes[0])
    for i in range(num_scene_size):
        assert len(
            meshes[i]
        ) == num_batch_size, 'Please make sure that the Meshes all have'
        'the same batch size.'
    meshes_all = []
    for j in range(num_batch_size):
        meshes_batch = []
        for i in range(num_scene_size):
            meshes_batch.append(meshes[i][j])
        meshes_all.append(join_meshes_as_scene(meshes_batch, include_textures))
    meshes_final = join_meshes_as_batch(meshes_all, include_textures)
    return meshes_final


def mesh_to_pointcloud_vc(
    meshes: Meshes,
    include_textures: bool = True,
    alpha: float = 1.0,
) -> Pointclouds:
    """Convert PyTorch3D vertex color `Meshes` to `PointClouds`.

    Args:
        meshes (Meshes): input meshes.
        include_textures (bool, optional): Whether include colors.
            Require the texture of input meshes is vertex color.
            Defaults to True.
        alpha (float, optional): transparency.
            Defaults to 1.0.

    Returns:
        Pointclouds: output pointclouds.
    """
    assert isinstance(
        meshes.textures,
        TexturesVertex), 'textures of input meshes should be `TexturesVertex`'
    vertices = meshes.verts_padded()
    if include_textures:
        verts_rgb = meshes.textures.verts_features_padded()
        verts_rgba = torch.cat(
            [verts_rgb,
             torch.ones_like(verts_rgb)[..., 0:1] * alpha], dim=-1)
    else:
        verts_rgba = None
    pointclouds = Pointclouds(points=vertices, features=verts_rgba)
    return pointclouds


def texture_uv2vc(meshes: Meshes) -> Meshes:
    """Convert a Pytorch3D meshes's textures from TexturesUV to TexturesVertex.

    Args:
        meshes (Meshes): input Meshes.

    Returns:
        Meshes: converted Meshes.
    """
    assert isinstance(meshes.textures, TexturesUV)
    device = meshes.device
    vert_uv = meshes.textures.verts_uvs_padded()
    batch_size = vert_uv.shape[0]
    verts_features = []
    num_verts = meshes.verts_padded().shape[1]
    for index in range(batch_size):
        face_uv = vert_uv[index][meshes.textures.faces_uvs_padded()
                                 [index].view(-1)]

        img = meshes.textures._maps_padded[index]
        width, height, _ = img.shape

        face_uv = face_uv * torch.Tensor([width - 1, height - 1
                                          ]).long().to(device)

        face_uv[:, 0] = torch.clip(face_uv[:, 0], 0, width - 1)
        face_uv[:, 1] = torch.clip(face_uv[:, 1], 0, height - 1)
        face_uv = face_uv.long()
        faces = meshes.faces_padded()
        verts_rgb = torch.zeros(1, num_verts, 3).to(device)
        verts_rgb[:, faces.view(-1)] = img[height - 1 - face_uv[:, 1],
                                           face_uv[:, 0]]
        verts_features.append(verts_rgb)
    verts_features = torch.cat(verts_features)

    meshes = meshes.clone()
    meshes.textures = TexturesVertex(verts_features)
    return meshes


def load_objs_as_meshes(files: List[str],
                        device: Optional[Union[torch.device, str]] = None,
                        load_textures: bool = True,
                        **kwargs) -> Meshes:
    if not isinstance(files, list):
        files = [files]
    return _load_objs_as_meshes(
        files=files, device=device, load_textures=load_textures, **kwargs)


def load_plys_as_meshes(
    files: List[str],
    device: Optional[Union[torch.device, str]] = None,
    load_textures: bool = True,
) -> Meshes:
    writer = IO()
    meshes = []
    if not isinstance(files, list):
        files = [files]
    for idx in range(len(files)):
        assert files[idx].endswith('.ply'), 'Please input .ply files.'
        mesh = writer.load_mesh(
            path=files[idx], include_textures=load_textures, device=device)
        meshes.append(mesh)
    meshes = join_meshes_as_batch(meshes, include_textures=load_textures)
    return meshes


def save_meshes_as_plys(files: List[str],
                        meshes: Meshes = None,
                        verts: torch.Tensor = None,
                        faces: torch.Tensor = None,
                        verts_rgb: torch.Tensor = None) -> None:
    """Save meshes as .ply files. Mainly for vertex color meshes.

    Args:
        files (List[str]): Output .ply file list.
        meshes (Meshes, optional): higher priority than
            (verts & faces & verts_rgb). Defaults to None.
        verts (torch.Tensor, optional): lower priority than meshes.
            Defaults to None.
        faces (torch.Tensor, optional): lower priority than meshes.
            Defaults to None.
        verts_rgb (torch.Tensor, optional): lower priority than meshes.
            Defaults to None.
    """
    if meshes is None:
        assert verts is not None and faces is not None, 'Not mesh input.'
        meshes = Meshes(
            verts=verts,
            faces=faces,
            textures=TexturesVertex(
                verts_features=verts_rgb) if verts_rgb is not None else None)
    else:
        if verts is not None or faces is not None or verts_rgb is not None:
            warnings.warn('Redundant input, will use meshes only.')
    assert files is not None
    if not isinstance(files, list):
        files = [files]
    assert len(files) >= len(meshes), 'Not enough output files.'
    writer = IO()
    for idx in range(len(meshes)):
        assert files[idx].endswith('.ply'), 'Please save as .ply files.'
        writer.save_mesh(
            meshes[idx], files[idx], colors_as_uint8=True, binary=False)


def save_meshes_as_objs(files: List[str], meshes: Meshes = None) -> None:
    """Save meshes as .obj files. Pytorch3D will not save vertex color for.

    .obj, please use `save_meshes_as_plys`.

    Args:
        files (List[str]): Output .obj file list.
        meshes (Meshes, optional):
            Defaults to None.
    """
    if not isinstance(files, list):
        files = [files]

    assert len(files) >= len(meshes), 'Not enough output files.'

    for idx in range(len(meshes)):
        prepare_output_path(
            files[idx], allowed_suffix=['.obj'],
            path_type='file'), 'Please save as .obj files.'
        if isinstance(meshes.textures, TexturesUV):
            verts_uvs = meshes.textures.verts_uvs_padded()[idx]
            faces_uvs = meshes.textures.faces_uvs_padded()[idx]
            texture_map = meshes.textures.maps_padded()[idx]
        else:
            verts_uvs = None
            faces_uvs = None
            texture_map = None
        save_obj(
            f=files[idx],
            verts=meshes.verts_padded()[idx],
            faces=meshes.faces_padded()[idx],
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            texture_map=texture_map)

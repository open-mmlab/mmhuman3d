import warnings
from typing import List

import torch
from pytorch3d.io import IO, save_obj
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import (
    Meshes,
    Pointclouds,
    join_meshes_as_batch,
    join_meshes_as_scene,
    padded_to_list,
)

from mmhuman3d.utils.path_utils import prepare_output_path


def join_batch_meshes_as_scene(
    meshes: List[Meshes],
    include_textures: bool = True,
) -> Meshes:
    """Join meshes as a scene each batch. Only for pytorch3d meshes. The Meshes
    must share the same batch size, and arbitrary topology. They must all be on
    the same device. If include_textures is true, they must all be compatible,
    either all or none having textures, and all the Textures objects being the
    same type. If include_textures is False, textures are ignored. If not,
    ValueError would be raised in join_meshes_as_batch and
    join_meshes_as_scene.

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
    """Convert pytorch3d `Meshes` to `PointClouds`.

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


def save_meshes_as_plys(meshes: Meshes = None,
                        verts: torch.Tensor = None,
                        faces: torch.Tensor = None,
                        verts_rgb: torch.Tensor = None,
                        paths: List[str] = []) -> None:
    """Save meshes as .ply files. Mainly for vertex color meshes.

    Args:
        meshes (Meshes, optional): higher priority than
            (verts & faces & verts_rgb). Defaults to None.
        verts (torch.Tensor, optional): lower priority than meshes.
            Defaults to None.
        faces (torch.Tensor, optional): lower priority than meshes.
            Defaults to None.
        verts_rgb (torch.Tensor, optional): lower priority than meshes.
            Defaults to None.
        paths (List[str], optional): Output .ply file list.
            Defaults to [].
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

    assert len(paths) >= len(meshes), 'Not enough output paths.'
    writer = IO()
    if not isinstance(paths, list):
        paths = [paths]
    for idx in range(len(meshes)):
        assert paths[idx].endswith('.ply'), 'Please save as .ply files.'
        writer.save_mesh(
            meshes[idx], paths[idx], colors_as_uint8=True, binary=False)


def save_meshes_as_objs(meshes: Meshes = None, paths: List[str] = []) -> None:
    """Save meshes as .obj files. Mainly for uv texture meshes.

    Args:
        meshes (Meshes, optional):
            Defaults to None.
        paths (List[str], optional): Output .obj file list.
            Defaults to [].
    """

    assert len(paths) >= len(meshes), 'Not enough output paths.'
    assert not isinstance(meshes.textures, TexturesVertex), 'For vertex '
    'color mesh please use save_meshes_as_plys.'
    if not isinstance(paths, list):
        paths = [paths]
    for idx in range(len(meshes)):
        prepare_output_path(
            paths[idx], allowed_suffix=['.obj'],
            path_type=['file']), 'Please save as .obj files.'
        save_obj(
            f=paths[idx],
            verts=meshes.verts_padded()[idx],
            faces=meshes.faces_padded()[idx],
            verts_uvs=meshes.textures.verts_uvs_padded()[idx],
            faces_uvs=meshes.textures.faces_uvs_padded()[idx],
            texture_map=meshes.textures.maps_padded()[idx])

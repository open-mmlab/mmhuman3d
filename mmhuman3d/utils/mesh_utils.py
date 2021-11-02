from typing import List, Optional, Union

import torch
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import (
    Meshes,
    Pointclouds,
    join_meshes_as_batch,
    join_meshes_as_scene,
    padded_to_list,
)
from pytorch3d.utils.ico_sphere import ico_sphere


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
):
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


def point_cloud_mesh(
    pointclouds: Pointclouds,
    vertices: Optional[torch.Tensor] = None,
    verts_rgb: Optional[torch.Tensor] = None,
    level: int = 0,
    radius: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
):
    assert pointclouds is not None or vertices is not None
    if pointclouds is not None:
        vertices = pointclouds.points_padded()
        verts_rgb = pointclouds.features_padded()
    else:
        if verts_rgb is None:
            verts_rgb = torch.ones_like(vertices)
    sphere_mesh = ico_sphere(level=level, device=device)
    ico_verts = sphere_mesh.verts_padded()
    ico_faces = sphere_mesh.faces_padded()
    num_vert_ico = ico_verts.shape[-2]
    num_batch, num_points = vertices.shape[:-1]  # num_batch, num_points, 3
    if level < 0:
        raise ValueError('level must be >= 0.')
    if level == 0:
        verts_ico = torch.tensor(
            ico_verts, dtype=torch.float32, device=device)[None]
        faces_ico = torch.tensor(
            ico_faces, dtype=torch.int64, device=device)[None]
    verts = verts_ico.repeat(num_batch, num_points, 1,
                             1)  # num_batch, num_points, num_ico, 3
    faces = faces_ico.repeat(num_batch, num_points, 1,
                             1)  # num_batch, num_points, num_face_ico, 3
    faces_offset = torch.range(0, num_points - 1) * num_vert_ico
    faces_offset = faces_offset.view(1, num_points, 1, 1)
    verts *= radius
    verts += vertices.unsqueeze(-2)
    faces += faces_offset
    mesh_list = []
    for idx in range(num_batch):
        mesh_list.append(
            join_meshes_as_scene(
                Meshes(
                    verts=verts[idx],
                    faces=faces[idx],
                    textures=TexturesVertex(verts_features=verts_rgb[idx]))))
    meshes = join_meshes_as_batch(mesh_list)
    return meshes

from typing import List

from pytorch3d.structures import (
    Meshes,
    join_meshes_as_batch,
    join_meshes_as_scene,
    padded_to_list,
)


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

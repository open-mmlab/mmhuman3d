import pytest
from pytorch3d.utils import torus

from mmhuman3d.utils.mesh_utils import (
    mesh_to_pointcloud_vc,
    save_meshes_as_plys,
)


def test_save_meshes():
    Torus = torus(r=10, R=20, sides=100, rings=100)
    # wrong paths
    with pytest.raises(AssertionError):
        save_meshes_as_plys(meshes=Torus)

    # No input
    with pytest.raises(AssertionError):
        save_meshes_as_plys()

    # File suffix wrong
    with pytest.raises(AssertionError):
        save_meshes_as_plys(Torus, paths=['1.obj'])

    save_meshes_as_plys(Torus, paths=['1.ply'])
    save_meshes_as_plys(
        verts=Torus.verts_padded(), faces=Torus.faces_padded(), paths='1.ply')
    save_meshes_as_plys(
        Torus,
        verts=Torus.verts_padded(),
        faces=Torus.faces_packed(),
        paths='1.ply')


def test_mesh2pointcloud():
    Torus = torus(r=10, R=20, sides=100, rings=100)
    Torus.textures = None
    with pytest.raises(AssertionError):
        mesh_to_pointcloud_vc(Torus)

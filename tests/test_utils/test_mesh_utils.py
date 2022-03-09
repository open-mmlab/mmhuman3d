import pytest
import torch
from pytorch3d.utils import torus

from mmhuman3d.core.visualization.renderer.torch3d_renderer.meshes import (
    ParametricMeshes,
    join_batch_meshes_as_scene,
)
from mmhuman3d.models.builder import build_body_model
from mmhuman3d.utils.mesh_utils import (
    mesh_to_pointcloud_vc,
    save_meshes_as_objs,
    save_meshes_as_plys,
)


def test_parametric_meshes_ops():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    body_model_config = {
        'type': 'smpl',
        'use_pca': False,
        'use_face_contour': True,
        'model_path': 'data/body_models/smpl'
    }
    body_model = build_body_model(body_model_config).to(device_name)
    full_pose = torch.Tensor(1, 72).to(device_name)
    pose_dict = body_model.tensor2dict(full_pose)
    meshes = ParametricMeshes(body_model=body_model, **pose_dict)
    meshes2 = meshes.extend(2)
    assert meshes2.shape == (2, 1)
    assert meshes2.verts_padded().shape == (2, meshes2.model_class.NUM_VERTS,
                                            3)
    meshes3 = meshes.extend(3, 4)
    assert meshes3.verts_padded().shape == (3,
                                            meshes2.model_class.NUM_VERTS * 4,
                                            3)
    meshes4 = meshes3.to(device_name)
    assert meshes4[:2, :3].shape == (2, 3)

    meshes5 = join_batch_meshes_as_scene(
        [meshes.extend(2, 3), meshes.extend(2, 2)])
    assert meshes5.shape == (2, 5)
    assert all(meshes.model_type == mesh
               for mesh in [meshes2, meshes3, meshes4, meshes5])


def test_save_meshes():
    Torus = torus(r=10, R=20, sides=100, rings=100)
    # wrong files
    with pytest.raises(AssertionError):
        save_meshes_as_plys(meshes=Torus)

    # No input
    with pytest.raises(AssertionError):
        save_meshes_as_plys()

    # File suffix wrong
    with pytest.raises(AssertionError):
        save_meshes_as_plys(Torus, files=['1.obj'])

    save_meshes_as_plys(Torus, files=['1.ply'])
    save_meshes_as_plys(
        verts=Torus.verts_padded(), faces=Torus.faces_padded(), files='1.ply')
    save_meshes_as_plys(
        Torus,
        verts=Torus.verts_padded(),
        faces=Torus.faces_packed(),
        files='1.ply')
    save_meshes_as_objs(Torus, files='1.obj')


def test_mesh2pointcloud():
    Torus = torus(r=10, R=20, sides=100, rings=100)
    Torus.textures = None
    with pytest.raises(AssertionError):
        mesh_to_pointcloud_vc(Torus)

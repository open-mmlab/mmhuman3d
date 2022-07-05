import os

import pytest
import torch
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.utils import torus

from mmhuman3d.core.renderer.torch3d_renderer.builder import build_renderer
from mmhuman3d.core.renderer.torch3d_renderer.meshes import (
    ParametricMeshes,
    join_batch_meshes_as_scene,
)
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.utils.mesh_utils import \
    join_batch_meshes_as_scene as join_batch_meshes_as_scene_
from mmhuman3d.utils.mesh_utils import (
    load_plys_as_meshes,
    mesh_to_pointcloud_vc,
    save_meshes_as_objs,
    save_meshes_as_plys,
    texture_uv2vc,
)

output_dir = 'tests/data/mesh_utils_output'
os.makedirs(output_dir, exist_ok=True)


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
    assert meshes.clone()
    assert meshes.detach()
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
    assert all(mesh.model_type == mesh.model_type
               for mesh in [meshes2, meshes3, meshes4, meshes5])
    assert all(
        mesh.textures._verts_features_padded.shape == mesh.verts_padded().shape
        for mesh in [meshes2, meshes3, meshes4, meshes5])


def test_parametric_meshes_ops_uv():
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
    uv_renderer = build_renderer(
        dict(
            type='uv',
            device=device_name,
            uv_param_path='data/body_models/smpl/smpl_uv.npz'))
    meshes = ParametricMeshes(
        body_model=body_model,
        uv_renderer=uv_renderer,
        texture_images=torch.ones(100, 100, 3),
        **pose_dict)
    assert isinstance(texture_uv2vc(meshes).textures, TexturesVertex)
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
    assert all(mesh.model_type == mesh.model_type
               for mesh in [meshes2, meshes3, meshes4, meshes5])
    # TODO: This assert is linked to ParametricMeshes line 382 and 497.
    # It is recommended that we can assure the shape of _maps_padded
    # is known due to the unique maps.
    # So this assert could be modified but we must know the what the texture
    # map of individual j in batch i is like.
    # assert all(mesh.textures._maps_padded.shape == (mesh.shape[0], 512,
    #                                                 512 * mesh._N_individual,
    #                                                 3)
    #            for mesh in [meshes2, meshes3, meshes4, meshes5])


def test_save_meshes():
    Torus = torus(r=10, R=20, sides=100, rings=100)
    # no files arg
    with pytest.raises(TypeError):
        save_meshes_as_plys(meshes=Torus)

    # No input
    with pytest.raises(AssertionError):
        save_meshes_as_plys(files=None)

    # File suffix wrong
    with pytest.raises(AssertionError):
        save_meshes_as_plys(files=[os.path.join(output_dir, 'test.obj')])

    save_meshes_as_plys(
        files=[os.path.join(output_dir, 'test.ply')], meshes=Torus)
    mesh = load_plys_as_meshes(files=[os.path.join(output_dir, 'test.ply')])
    assert mesh.verts_padded().shape == Torus.verts_padded().shape
    save_meshes_as_plys(
        files=os.path.join(output_dir, 'test.ply'),
        verts=Torus.verts_padded(),
        faces=Torus.faces_padded())
    save_meshes_as_plys(
        files=os.path.join(output_dir, 'test.ply'),
        meshes=Torus,
        verts=Torus.verts_padded(),
        faces=Torus.faces_packed(),
    )
    save_meshes_as_objs(
        files=os.path.join(output_dir, 'test.obj'), meshes=Torus)
    Torus2 = Torus.extend(2)
    Torus2_2 = join_batch_meshes_as_scene_([Torus2, Torus2])
    assert Torus2_2.verts_padded().shape == (2,
                                             Torus.verts_padded().shape[1] * 2,
                                             3)


def test_mesh2pointcloud():
    Torus = torus(r=10, R=20, sides=100, rings=100)
    Torus.textures = None
    with pytest.raises(AssertionError):
        mesh_to_pointcloud_vc(Torus)

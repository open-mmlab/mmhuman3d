import torch
from pytorch3d.structures import Meshes

from mmhuman3d.core.renderer.torch3d_renderer.builder import UVRenderer
from mmhuman3d.models.body_models.builder import build_body_model

if torch.cuda.is_available():
    device_name = 'cuda:0'
else:
    device_name = 'cpu'

device = torch.device(device_name)
uv_param_path = 'data/body_models/smpl/smpl_uv.npz'
uv_renderer = UVRenderer(
    resolution=(512, 512),
    model_type='smpl',
    uv_param_path=uv_param_path,
    device=device)

model_path = 'data/body_models/smpl'
body_model = build_body_model(dict(type='smpl',
                                   model_path=model_path)).to(device)


def test_uv_resample():
    pose_dict = body_model.tensor2dict(
        torch.zeros(1, (body_model.NUM_BODY_JOINTS + 1) * 3).to(device))
    smpl_output = body_model(**pose_dict)
    verts = smpl_output['vertices'].view(1, body_model.NUM_VERTS, 3)
    mesh = Meshes(
        verts=verts,
        faces=body_model.faces_tensor.to(device).view(1, body_model.NUM_FACES,
                                                      3))

    displacement_map = torch.ones(1, 600, 600, 3).to(device)
    normal_map = torch.ones(1, 600, 600, 3).to(device)
    texture_map = torch.ones(1, 600, 600, 3).to(device)
    mesh1 = uv_renderer.wrap_displacement(
        mesh, displacement_map=displacement_map)
    assert torch.isclose(
        mesh.verts_padded(), mesh1.verts_padded() - 1, atol=1e-3).all()
    mesh2 = uv_renderer.wrap_normal(mesh, normal_map=normal_map)
    assert (mesh2.verts_normals_padded() == 1).all()
    mesh3 = mesh2.clone()
    mesh3.textures = uv_renderer.wrap_texture(texture_map=texture_map)
    assert mesh3.textures.maps_padded().shape == (1, 600, 600, 3)

    normal_map_small = torch.ones(1, 200, 200, 3).to(device)
    mesh4 = uv_renderer.wrap_normal(mesh, normal_map=normal_map_small)
    assert (mesh4.verts_normals_padded() == 1).all()


def test_uv_forward():
    verts_attr = torch.zeros(2, 6890, 3)
    attr_map = uv_renderer(verts_attr, resolution=(600, 600))
    assert attr_map.shape == (2, 600, 600, 3)

    pose_dict = body_model.tensor2dict(
        torch.zeros(1, (body_model.NUM_BODY_JOINTS + 1) * 3).to(device))
    smpl_output = body_model(**pose_dict)
    verts = smpl_output['vertices'].view(1, body_model.NUM_VERTS, 3)
    normal_map = uv_renderer.forward_normal_map(
        vertices=verts, resolution=(512, 512))
    assert normal_map.shape == (1, 512, 512, 3)

    uvd_map = uv_renderer.forward_uvd_map(
        vertices=verts, resolution=(512, 512))
    assert uvd_map.shape == (1, 512, 512, 3)

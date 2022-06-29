import numpy as np
import pytest
import torch
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.utils import ico_sphere

from mmhuman3d.core.cameras import compute_orbit_cameras
from mmhuman3d.core.cameras.builder import build_cameras
from mmhuman3d.core.renderer.mpr_renderer.camera import Pinhole2D
from mmhuman3d.core.renderer.torch3d_renderer import render_runner
from mmhuman3d.core.renderer.torch3d_renderer.builder import build_renderer
from mmhuman3d.models.body_models.builder import build_body_model


def test_render_runner():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'

    device = torch.device(device_name)
    meshes = ico_sphere(3, device)

    meshes.textures = TexturesVertex(
        verts_features=torch.ones_like(meshes.verts_padded()).to(device))
    K, R, T = compute_orbit_cameras(orbit_speed=1.0, batch_size=2)
    resolution = 128
    cameras = build_cameras(
        dict(type='fovperspective', K=K, R=R, T=T, resolution=resolution))
    renderer = build_renderer(
        dict(
            type='mesh',
            resolution=resolution,
            shader=dict(type='soft_phong'),
            lights=dict(type='ambient')))
    tensor = render_runner.render(
        meshes=meshes.extend(2),
        cameras=cameras,
        renderer=renderer,
        device=device,
        return_tensor=True,
        batch_size=2,
        output_path='/tmp/demo.mp4')
    assert tensor.shape == (2, 128, 128, 4)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_realtime_render_cuda():
    from mmhuman3d.core.renderer.mpr_renderer.smpl_realrender import VisualizerMeshSMPL  # noqa: E501

    vertices = torch.ones([6890, 3]).to(device='cuda')
    body_model = build_body_model(
        dict(
            type='SMPL',
            gender='neutral',
            num_betas=10,
            model_path='data/body_models/smpl'))
    renderer = VisualizerMeshSMPL(
        body_models=body_model, resolution=[224, 224], device='cuda')

    res = renderer(vertices)
    assert res.shape == (224, 224, 3)


def test_realtime_render():
    # test camera
    pinhole2d = Pinhole2D(fx=5000., fy=5000., cx=112, cy=112, w=1024, h=1024)
    K = pinhole2d.get_K()
    assert K == np.array([[5000., 0, 112], [0, 5000., 112], [0, 0, 1]])

    verts = torch.ones([6890, 3])
    verts_ndc = pinhole2d.project_ndc(verts)
    assert verts_ndc.shape == (6890, 3)

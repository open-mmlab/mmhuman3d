from unittest.mock import patch

import numpy as np
import pytest
import torch
from surrogate import surrogate

from mmhuman3d.core.renderer.mpr_renderer.camera import Pinhole2D
from mmhuman3d.models.body_models.builder import build_body_model


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


def mock_estimate_normals_cuda(*args):
    return torch.zeros([1024, 1024, 3]), torch.zeros([1024, 1024, 3])


def mock_project_mesh_cuda(*args):
    return torch.zeros([1024, 1024, 1])


@surrogate(
    'mmhuman3d.core.renderer.mpr_renderer.cuda.rasterizer.estimate_normals')
@patch('mmhuman3d.core.renderer.mpr_renderer.cuda.rasterizer.estimate_normals',
       mock_estimate_normals_cuda)
@surrogate('mmhuman3d.core.renderer.mpr_renderer.cuda.rasterizer.project_mesh')
@patch('mmhuman3d.core.renderer.mpr_renderer.cuda.rasterizer.project_mesh',
       mock_project_mesh_cuda)
def test_realtime_render():
    # test estimate normals
    from mmhuman3d.core.renderer.mpr_renderer.rasterizer import \
        estimate_normals
    vertices = torch.zeros([6890, 3])
    vertices_filter = torch.ones(6890)
    faces = torch.zeros([13776, 3])
    pinhole2d = Pinhole2D(fx=5000., fy=5000., cx=112, cy=112, w=1024, h=1024)

    coords, normals = estimate_normals(vertices, faces, pinhole2d,
                                       vertices_filter)
    assert coords.shape == (1024, 1024, 3)

    # requires CUDA support
    with pytest.raises(AssertionError):
        coords, normals = estimate_normals(vertices, faces, pinhole2d)

    # test project mesh
    from mmhuman3d.core.renderer.mpr_renderer.rasterizer import \
        project_mesh

    z_buff = project_mesh(vertices, faces, vertices, pinhole2d,
                          vertices_filter)
    assert z_buff.shape == (1024, 1024, 1)

    # requires CUDA support
    with pytest.raises(AssertionError):
        z_buff = project_mesh(vertices, faces, vertices, pinhole2d)

    # test camera
    pinhole2d = Pinhole2D(fx=5000., fy=5000., cx=112, cy=112, w=1024, h=1024)
    K = pinhole2d.get_K()
    assert (K == np.array([[5000., 0, 112], [0, 5000., 112], [0, 0, 1]])).all()

    verts = torch.ones([6890, 3])
    verts_ndc = pinhole2d.project_ndc(verts)
    assert verts_ndc.shape == (6890, 3)

    # test utils
    from mmhuman3d.core.renderer.mpr_renderer.utils import (vis_z_buffer,
                                                            vis_normals)
    z_ones = torch.ones([1024, 1024, 1])
    z_zeros = torch.zeros([1024, 1024, 1])
    coords = torch.zeros([1024, 1024, 3])
    normals = torch.zeros([1024, 1024, 3])

    res = vis_z_buffer(z_zeros)
    assert res.shape == (1024, 1024)

    res = vis_z_buffer(z_ones)
    assert res.shape == (1024, 1024)

    res = vis_normals(coords, normals)
    assert res.shape == (1024, 1024)


def mock_estimate_normals(vertices, faces, pinhole):
    return torch.zeros([1024, 1024, 3]), torch.zeros([1024, 1024, 3])


@patch('mmhuman3d.core.renderer.mpr_renderer.rasterizer.estimate_normals',
       mock_estimate_normals)
@surrogate(
    'mmhuman3d.core.renderer.mpr_renderer.cuda.rasterizer.estimate_normals')
@patch('mmhuman3d.core.renderer.mpr_renderer.cuda.rasterizer.estimate_normals',
       mock_estimate_normals_cuda)
@surrogate('mmhuman3d.core.renderer.mpr_renderer.cuda.rasterizer.project_mesh')
@patch('mmhuman3d.core.renderer.mpr_renderer.cuda.rasterizer.project_mesh',
       mock_project_mesh_cuda)
def test_smpl_realtime_render():
    from mmhuman3d.core.renderer.mpr_renderer.smpl_realrender import VisualizerMeshSMPL  # noqa: E501
    vertices = torch.ones([6890, 3]).to(device='cpu')
    body_model = build_body_model(
        dict(
            type='SMPL',
            gender='neutral',
            num_betas=10,
            model_path='data/body_models/smpl'))
    renderer = VisualizerMeshSMPL(
        body_models=body_model, resolution=[1024, 1024], device='cpu')

    res = renderer(vertices)
    assert res.shape == (1024, 1024, 3)

    bg = np.zeros([1024, 1024, 3])
    res = renderer(vertices, bg=bg)
    assert res.shape == (1024, 1024, 3)

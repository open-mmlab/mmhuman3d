import torch
from minimal_pytorch_rasterizer.cuda.rasterizer import estimate_normals as estimate_normals_cuda
from minimal_pytorch_rasterizer.cuda.rasterizer import project_mesh as project_mesh_cuda
from minimal_pytorch_rasterizer import assert_utils


def estimate_normals(vertices, faces, pinhole, vertices_filter=None):
    if vertices_filter is None:
        assert_utils.is_cuda_tensor(vertices)
        assert_utils.check_shape_len(vertices, 2)
        n = vertices.shape[0]
        vertices_filter = torch.ones((n), dtype=torch.uint8, device=vertices.device)
    vertices = vertices.contiguous()
    vertices_ndc = pinhole.project_ndc(vertices)
    coords, normals = estimate_normals_cuda(
        vertices_ndc, faces, vertices, vertices_filter,
        pinhole.h, pinhole.w
    )
    return coords, normals


def project_mesh(vertices, faces, vertice_values, pinhole, vertices_filter=None):
    if vertices_filter is None:
        assert_utils.is_cuda_tensor(vertices)
        assert_utils.check_shape_len(vertices, 2)
        n = vertices.shape[0]
        vertices_filter = torch.ones((n), dtype=torch.uint8, device=vertices.device)
    vertices = vertices.contiguous()
    vertices_ndc = pinhole.project_ndc(vertices)
    return project_mesh_cuda(
        vertices_ndc, faces, vertice_values, vertices_filter,
        pinhole.h, pinhole.w
    )

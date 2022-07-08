#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> estimate_normals_cuda(
    const torch::Tensor& vertices_ndc,
    const torch::Tensor& faces,
    const torch::Tensor& vertices,
    const torch::Tensor& vertices_filter,
    int h, int w
);


torch::Tensor project_mesh_cuda(
    const torch::Tensor& vertices_ndc,
    const torch::Tensor& faces,
    const torch::Tensor& vertice_values,
    const torch::Tensor& vertices_filter,
    int h, int w
);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void check_equal_dtype(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(
        a.dtype() == b.dtype(),
        "expected equal dtype, got ", a.dtype(), " != ", b.dtype()
    );
}

void check_equal_gpuid(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(
        a.device().index() == b.device().index(),
        "expected equal gpu id, got ", a.device().index(), " != ", b.device().index()
    );
}

std::vector<torch::Tensor> estimate_normals(
    const torch::Tensor& vertices_ndc,
    const torch::Tensor& faces,
    const torch::Tensor& vertices,
    const torch::Tensor& vertices_filter,
    int h, int w
) {
    TORCH_CHECK(h > 0, "h expected to be > 0");
    TORCH_CHECK(w > 0, "w expected to be > 0");
    CHECK_INPUT(vertices_ndc);
    CHECK_INPUT(faces);
    CHECK_INPUT(vertices_filter);
    return estimate_normals_cuda(
        vertices_ndc, faces, vertices, vertices_filter,
        h, w
    );
}

torch::Tensor project_mesh(
    const torch::Tensor& vertices_ndc,
    const torch::Tensor& faces,
    const torch::Tensor& vertice_values,
    const torch::Tensor& vertices_filter,
    int h, int w
) {
    TORCH_CHECK(h > 0, "h expected to be > 0");
    TORCH_CHECK(w > 0, "w expected to be > 0");
    CHECK_INPUT(vertices_ndc);
    CHECK_INPUT(faces);
    CHECK_INPUT(vertice_values);
    CHECK_INPUT(vertices_filter);
    return project_mesh_cuda(
        vertices_ndc, faces, vertice_values, vertices_filter,
        h, w
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("estimate_normals", &estimate_normals, "estimate_normals (CUDA)");
    m.def("project_mesh", &project_mesh, "project_mesh (CUDA)");
}

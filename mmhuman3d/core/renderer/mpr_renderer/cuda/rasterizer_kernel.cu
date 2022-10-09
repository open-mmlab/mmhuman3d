/*

There are 2 ways to rasterize triangles that came to mind:
1) iterate over all pixels (they define CUDA grid), for selected pixel feed all triangles to 1 CUDA block
2) iterate over all triangels (they define CUDA grid), for selected triangle feed pixels that are bounded by selected triangle to 1 CUDA block

2nd way is implemented here
*/


#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define BLOCK_SIZE 512
#define BLOCK_SIZE_2D_X 32
#define BLOCK_SIZE_2D_Y 16
#define BLOCK_SIZE_3D_X 32
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 4

// vertices coords:
// vertices[:, 0]: x
// vertices[:, 1]: y
// vertices[:, 2]: z

// 2d tensor axis:
// 0: yi
// 1: xi

// 3d tensor axis:
// 0: zi
// 1: yi
// 2: xi

template <typename scalar_t>
__device__ __forceinline__ scalar_t atomicMinFloat(scalar_t * addr, scalar_t value) {
        scalar_t old;
        old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
             __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
        return old;
}

__device__ double atomicMin_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// kernel utils

template <typename scalar_t>
__device__ int lower_bound(const scalar_t* values, const scalar_t value, const int N) {
    int left = 0;
    int right = N;
    int mid;
    while (right - left > 1) {
        mid = (left + right) / 2;
        if (values[mid] < value) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return right;
}

// kernels

template <typename scalar_t>
__global__ void rasterize_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2> vertices_ndc,
    const torch::PackedTensorAccessor32<int32_t,2> faces,
    const torch::PackedTensorAccessor32<uint8_t,1> vertices_filter,
    torch::PackedTensorAccessor32<scalar_t,2> depth,
    scalar_t* global_face_ndc_inv,
    int* global_is_bad_face
) {
    const int face_indx = blockIdx.x;
    const int H = depth.size(0);
    const int W = depth.size(1);

    scalar_t min_x, max_x, min_y, max_y;
    scalar_t denom;

    __shared__ int vertices_per_thread_x, vertices_per_thread_y;
    __shared__ int ai, bi, ci;
    __shared__ bool is_bad_face;
    __shared__ int min_xi, max_xi, min_yi, max_yi;
    __shared__ scalar_t face_ndc[9];
    __shared__ scalar_t face_ndc_inv[9];
    const scalar_t eps = 1e-5;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        ai = faces[face_indx][0];
        bi = faces[face_indx][1];
        ci = faces[face_indx][2];

        if (vertices_filter[ai] == 0 || vertices_filter[bi] == 0 || vertices_filter[ci] == 0) {
            is_bad_face = true;
            global_is_bad_face[face_indx] = 1;
            return;
        }

        face_ndc[0] = vertices_ndc[ai][0]; face_ndc[1] = vertices_ndc[ai][1]; face_ndc[2] = vertices_ndc[ai][2];
        face_ndc[3] = vertices_ndc[bi][0]; face_ndc[4] = vertices_ndc[bi][1]; face_ndc[5] = vertices_ndc[bi][2];
        face_ndc[6] = vertices_ndc[ci][0]; face_ndc[7] = vertices_ndc[ci][1]; face_ndc[8] = vertices_ndc[ci][2];

        // negative vertex
        is_bad_face = false;
        if (face_ndc[2] < eps || face_ndc[5] < eps || face_ndc[8] < eps) {
            is_bad_face = true;
            global_is_bad_face[face_indx] = 1;
            return;
        }

        face_ndc_inv[0] = face_ndc[4] - face_ndc[7];
        face_ndc_inv[1] = face_ndc[6] - face_ndc[3];
        face_ndc_inv[2] = face_ndc[3] * face_ndc[7] - face_ndc[6] * face_ndc[4];
        face_ndc_inv[3] = face_ndc[7] - face_ndc[1];
        face_ndc_inv[4] = face_ndc[0] - face_ndc[6];
        face_ndc_inv[5] = face_ndc[6] * face_ndc[1] - face_ndc[0] * face_ndc[7];
        face_ndc_inv[6] = face_ndc[1] - face_ndc[4];
        face_ndc_inv[7] = face_ndc[3] - face_ndc[0];
        face_ndc_inv[8] = face_ndc[0] * face_ndc[4] - face_ndc[3] * face_ndc[1];

        denom = (
            face_ndc[6] * (face_ndc[1] - face_ndc[4]) +
            face_ndc[0] * (face_ndc[4] - face_ndc[7]) +
            face_ndc[3] * (face_ndc[7] - face_ndc[1])
        );

//        if (abs(denom) < eps) {
//            is_bad_face = true;
//            global_is_bad_face[face_indx] = 1;
//            return;
//        }

        for (int i = 0; i < 9; ++i) {
            face_ndc_inv[i] /= denom;
        }

        for (int i = 0; i < 9; ++i) {
            global_face_ndc_inv[9 * face_indx + i] = face_ndc_inv[i];
        }

        global_is_bad_face[face_indx] = 0;

        min_x = min(min(face_ndc[0], face_ndc[3]), face_ndc[6]);
        min_x = (min_x + 1) / 2 * W;  // convert from ndc to img coordinates
        min_xi = static_cast<int>(floorf(static_cast<float>(min_x)));
        min_xi = min(max(min_xi, 0), W - 1);
        max_x = max(max(face_ndc[0], face_ndc[3]), face_ndc[6]);
        max_x = (max_x + 1) / 2 * W;
        max_xi = static_cast<int>(ceilf(static_cast<float>(max_x)));
        max_xi = min(max(max_xi, 0), W - 1);

        min_y = min(min(face_ndc[1], face_ndc[4]), face_ndc[7]);
        min_y = (min_y + 1) / 2 * H;
        min_yi = static_cast<int>(floorf(static_cast<float>(min_y)));
        min_yi = min(max(min_yi, 0), H - 1);
        max_y = max(max(face_ndc[1], face_ndc[4]), face_ndc[7]);
        max_y = (max_y + 1) / 2 * H;
        max_yi = static_cast<int>(ceilf(static_cast<float>(max_y)));
        max_yi = min(max(max_yi, 0), H - 1);

        vertices_per_thread_x = (max_xi - min_xi) / blockDim.x + 1;
        vertices_per_thread_y = (max_yi - min_yi) / blockDim.y + 1;
    }
    __syncthreads();
    if (is_bad_face) {
        return;
    }

    const int left = min_xi + vertices_per_thread_x * threadIdx.x;
    const int right = min(left + vertices_per_thread_x, max_xi);

    const int top = min_yi + vertices_per_thread_y * threadIdx.y;
    const int bottom = min(top + vertices_per_thread_y, max_yi);

    scalar_t x, y, face_z, wa, wb, wc, wsum;
    for (int i = top; i <= bottom; i++) {
        for (int j = left; j <= right; j++) {
            x = 2 * ((scalar_t)j + 0.5) / W - 1;
            y = 2 * ((scalar_t)i + 0.5) / H - 1;

            // check pixel is inside the face
            if (((y - face_ndc[1]) * (face_ndc[3] - face_ndc[0]) > (x - face_ndc[0]) * (face_ndc[4] - face_ndc[1])) ||
                ((y - face_ndc[4]) * (face_ndc[6] - face_ndc[3]) > (x - face_ndc[3]) * (face_ndc[7] - face_ndc[4])) ||
                ((y - face_ndc[7]) * (face_ndc[0] - face_ndc[6]) > (x - face_ndc[6]) * (face_ndc[1] - face_ndc[7]))) {
                continue;
            }

            wa = face_ndc_inv[0] * x + face_ndc_inv[1] * y + face_ndc_inv[2];
            wb = face_ndc_inv[3] * x + face_ndc_inv[4] * y + face_ndc_inv[5];
            wc = face_ndc_inv[6] * x + face_ndc_inv[7] * y + face_ndc_inv[8];
            wsum = wa + wb + wc;
            wa /= wsum; wb /= wsum; wc /= wsum;

            wa /= face_ndc[2];
            wb /= face_ndc[5];
            wc /= face_ndc[8];
            wsum = wa + wb + wc;
            wa /= wsum; wb /= wsum; wc /= wsum;

            face_z = wa * face_ndc[2] + wb * face_ndc[5] + wc * face_ndc[8];

            if (sizeof(scalar_t) == sizeof(double)) {
                atomicMin_double((double*)&depth[i][j], (double)face_z);
            } else {
                atomicMinFloat(&depth[i][j], face_z);
            }
        }
    }
}


template <typename scalar_t>
__global__ void interpolate_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2> vertices_ndc,
    const torch::PackedTensorAccessor32<int32_t,2> faces,
    const torch::PackedTensorAccessor32<scalar_t,2> depth,
    const scalar_t* global_face_ndc_inv,
    const int* global_is_bad_face,
    const torch::PackedTensorAccessor32<scalar_t,2> vertice_values,
    torch::PackedTensorAccessor32<scalar_t,3> result
) {
    const int face_indx = blockIdx.x;

    if (global_is_bad_face[face_indx]) {
        return;
    }

    const int H = depth.size(0);
    const int W = depth.size(1);
    const int C = vertice_values.size(1);
    const scalar_t eps = 1e-5;

    scalar_t min_x, max_x, min_y, max_y;
    __shared__ int vertices_per_thread_x, vertices_per_thread_y;
    __shared__ int ai, bi, ci;
    __shared__ scalar_t face_ndc[9];
    __shared__ scalar_t face_ndc_inv[9];
    __shared__ int min_xi, max_xi, min_yi, max_yi;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        ai = faces[face_indx][0];
        bi = faces[face_indx][1];
        ci = faces[face_indx][2];

        face_ndc[0] = vertices_ndc[ai][0]; face_ndc[1] = vertices_ndc[ai][1]; face_ndc[2] = vertices_ndc[ai][2];
        face_ndc[3] = vertices_ndc[bi][0]; face_ndc[4] = vertices_ndc[bi][1]; face_ndc[5] = vertices_ndc[bi][2];
        face_ndc[6] = vertices_ndc[ci][0]; face_ndc[7] = vertices_ndc[ci][1]; face_ndc[8] = vertices_ndc[ci][2];

        for (int i = 0; i < 9; ++i) {
            face_ndc_inv[i] = global_face_ndc_inv[9 * face_indx + i];
        }

        min_x = min(min(face_ndc[0], face_ndc[3]), face_ndc[6]);
        min_x = (min_x + 1) / 2 * W;  // convert from ndc to img coordinates
        min_xi = static_cast<int>(floorf(static_cast<float>(min_x)));
        min_xi = min(max(min_xi, 0), W - 1);
        max_x = max(max(face_ndc[0], face_ndc[3]), face_ndc[6]);
        max_x = (max_x + 1) / 2 * W;
        max_xi = static_cast<int>(ceilf(static_cast<float>(max_x)));
        max_xi = min(max(max_xi, 0), W - 1);

        min_y = min(min(face_ndc[1], face_ndc[4]), face_ndc[7]);
        min_y = (min_y + 1) / 2 * H;
        min_yi = static_cast<int>(floorf(static_cast<float>(min_y)));
        min_yi = min(max(min_yi, 0), H - 1);
        max_y = max(max(face_ndc[1], face_ndc[4]), face_ndc[7]);
        max_y = (max_y + 1) / 2 * H;
        max_yi = static_cast<int>(ceilf(static_cast<float>(max_y)));
        max_yi = min(max(max_yi, 0), H - 1);

        vertices_per_thread_x = (max_xi - min_xi) / blockDim.x + 1;
        vertices_per_thread_y = (max_yi - min_yi) / blockDim.y + 1;
    }
    __syncthreads();

    const int left = min_xi + vertices_per_thread_x * threadIdx.x;
    const int right = min(left + vertices_per_thread_x, max_xi);

    const int top = min_yi + vertices_per_thread_y * threadIdx.y;
    const int bottom = min(top + vertices_per_thread_y, max_yi);

    scalar_t x, y, face_z, wa, wb, wc, wsum;
    for (int i = top; i <= bottom; i++) {
        for (int j = left; j <= right; j++) {
            x = 2 * ((scalar_t)j + 0.5) / W - 1;
            y = 2 * ((scalar_t)i + 0.5) / H - 1;

            // check pixel is inside the face
            if (((y - face_ndc[1]) * (face_ndc[3] - face_ndc[0]) > (x - face_ndc[0]) * (face_ndc[4] - face_ndc[1])) ||
                ((y - face_ndc[4]) * (face_ndc[6] - face_ndc[3]) > (x - face_ndc[3]) * (face_ndc[7] - face_ndc[4])) ||
                ((y - face_ndc[7]) * (face_ndc[0] - face_ndc[6]) > (x - face_ndc[6]) * (face_ndc[1] - face_ndc[7]))) {
                continue;
            }

            wa = face_ndc_inv[0] * x + face_ndc_inv[1] * y + face_ndc_inv[2];
            wb = face_ndc_inv[3] * x + face_ndc_inv[4] * y + face_ndc_inv[5];
            wc = face_ndc_inv[6] * x + face_ndc_inv[7] * y + face_ndc_inv[8];
            wsum = wa + wb + wc;
            wa /= wsum; wb /= wsum; wc /= wsum;

            wa /= face_ndc[2];
            wb /= face_ndc[5];
            wc /= face_ndc[8];
            wsum = wa + wb + wc;
            wa /= wsum; wb /= wsum; wc /= wsum;

            face_z = wa * face_ndc[2] + wb * face_ndc[5] + wc * face_ndc[8];

            if (face_z - eps < depth[i][j]) {
                for (int c = 0; c < C; c++) {
                    result[i][j][c] = wa * vertice_values[ai][c] + wb * vertice_values[bi][c] + wc * vertice_values[ci][c];
                }
            }
        }
    }
}


template <typename scalar_t>
__global__ void estimate_normals_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2> vertices_ndc,
    const torch::PackedTensorAccessor32<int32_t,2> faces,
    const torch::PackedTensorAccessor32<scalar_t,2> depth,
    const scalar_t* global_face_ndc_inv,
    const int* global_is_bad_face,
    const torch::PackedTensorAccessor32<scalar_t,2> vertices,
    torch::PackedTensorAccessor32<scalar_t,3> coords,
    torch::PackedTensorAccessor32<scalar_t,3> normals
) {
    const int face_indx = blockIdx.x;

    if (global_is_bad_face[face_indx]) {
        return;
    }

    const int H = depth.size(0);
    const int W = depth.size(1);
    const scalar_t eps = 1e-5;

    scalar_t min_x, max_x, min_y, max_y;
    scalar_t v1x, v1y, v1z, v2x, v2y, v2z, nlen;
    __shared__ int vertices_per_thread_x, vertices_per_thread_y;
    __shared__ int ai, bi, ci;
    __shared__ scalar_t face[9];
    __shared__ scalar_t face_ndc[9];
    __shared__ scalar_t face_ndc_inv[9];
    __shared__ int min_xi, max_xi, min_yi, max_yi;
    __shared__ scalar_t nx, ny, nz;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        ai = faces[face_indx][0];
        bi = faces[face_indx][1];
        ci = faces[face_indx][2];

        face[0] = vertices[ai][0]; face[1] = vertices[ai][1]; face[2] = vertices[ai][2];
        face[3] = vertices[bi][0]; face[4] = vertices[bi][1]; face[5] = vertices[bi][2];
        face[6] = vertices[ci][0]; face[7] = vertices[ci][1]; face[8] = vertices[ci][2];

        v1x = face[3] - face[0]; v2x = face[6] - face[0];
        v1y = face[4] - face[1]; v2y = face[7] - face[1];
        v1z = face[5] - face[2]; v2z = face[8] - face[2];

        nx = v1y * v2z - v1z * v2y;
        ny = v1z * v2x - v1x * v2z;
        nz = v1x * v2y - v1y * v2x;
        nlen = nx * nx + ny * ny + nz * nz;
        nlen = (scalar_t)sqrt((float)nlen);
        nx /= nlen;
        ny /= nlen;
        nz /= nlen;

        face_ndc[0] = vertices_ndc[ai][0]; face_ndc[1] = vertices_ndc[ai][1]; face_ndc[2] = vertices_ndc[ai][2];
        face_ndc[3] = vertices_ndc[bi][0]; face_ndc[4] = vertices_ndc[bi][1]; face_ndc[5] = vertices_ndc[bi][2];
        face_ndc[6] = vertices_ndc[ci][0]; face_ndc[7] = vertices_ndc[ci][1]; face_ndc[8] = vertices_ndc[ci][2];

        for (int i = 0; i < 9; ++i) {
            face_ndc_inv[i] = global_face_ndc_inv[9 * face_indx + i];
        }

        min_x = min(min(face_ndc[0], face_ndc[3]), face_ndc[6]);
        min_x = (min_x + 1) / 2 * W;  // convert from ndc to img coordinates
        min_xi = static_cast<int>(floorf(static_cast<float>(min_x)));
        min_xi = min(max(min_xi, 0), W - 1);
        max_x = max(max(face_ndc[0], face_ndc[3]), face_ndc[6]);
        max_x = (max_x + 1) / 2 * W;
        max_xi = static_cast<int>(ceilf(static_cast<float>(max_x)));
        max_xi = min(max(max_xi, 0), W - 1);

        min_y = min(min(face_ndc[1], face_ndc[4]), face_ndc[7]);
        min_y = (min_y + 1) / 2 * H;
        min_yi = static_cast<int>(floorf(static_cast<float>(min_y)));
        min_yi = min(max(min_yi, 0), H - 1);
        max_y = max(max(face_ndc[1], face_ndc[4]), face_ndc[7]);
        max_y = (max_y + 1) / 2 * H;
        max_yi = static_cast<int>(ceilf(static_cast<float>(max_y)));
        max_yi = min(max(max_yi, 0), H - 1);

        vertices_per_thread_x = (max_xi - min_xi) / blockDim.x + 1;
        vertices_per_thread_y = (max_yi - min_yi) / blockDim.y + 1;
    }
    __syncthreads();

    const int left = min_xi + vertices_per_thread_x * threadIdx.x;
    const int right = min(left + vertices_per_thread_x, max_xi);

    const int top = min_yi + vertices_per_thread_y * threadIdx.y;
    const int bottom = min(top + vertices_per_thread_y, max_yi);

    scalar_t x, y, face_z, wa, wb, wc, wsum;
    for (int i = top; i <= bottom; i++) {
        for (int j = left; j <= right; j++) {
            x = 2 * ((scalar_t)j + 0.5) / W - 1;
            y = 2 * ((scalar_t)i + 0.5) / H - 1;

            // check pixel is inside the face
            if (((y - face_ndc[1]) * (face_ndc[3] - face_ndc[0]) > (x - face_ndc[0]) * (face_ndc[4] - face_ndc[1])) ||
                ((y - face_ndc[4]) * (face_ndc[6] - face_ndc[3]) > (x - face_ndc[3]) * (face_ndc[7] - face_ndc[4])) ||
                ((y - face_ndc[7]) * (face_ndc[0] - face_ndc[6]) > (x - face_ndc[6]) * (face_ndc[1] - face_ndc[7]))) {
                continue;
            }

            wa = face_ndc_inv[0] * x + face_ndc_inv[1] * y + face_ndc_inv[2];
            wb = face_ndc_inv[3] * x + face_ndc_inv[4] * y + face_ndc_inv[5];
            wc = face_ndc_inv[6] * x + face_ndc_inv[7] * y + face_ndc_inv[8];
            wsum = wa + wb + wc;
            wa /= wsum; wb /= wsum; wc /= wsum;

            wa /= face_ndc[2];
            wb /= face_ndc[5];
            wc /= face_ndc[8];
            wsum = wa + wb + wc;
            wa /= wsum; wb /= wsum; wc /= wsum;

            face_z = wa * face_ndc[2] + wb * face_ndc[5] + wc * face_ndc[8];

            if (face_z - eps < depth[i][j]) {
                coords[i][j][0] = wa * face[0] + wb * face[3] + wc * face[6];
                coords[i][j][1] = wa * face[1] + wb * face[4] + wc * face[7];
                coords[i][j][2] = wa * face[2] + wb * face[5] + wc * face[8];

                normals[i][j][0] = nx;
                normals[i][j][1] = ny;
                normals[i][j][2] = nz;
            }
        }
    }
}

// cpp defined functions

torch::Tensor project_mesh_cuda(
    const torch::Tensor& vertices_ndc,
    const torch::Tensor& faces,
    const torch::Tensor& vertice_values,
    const torch::Tensor& vertices_filter,
    int H, int W
) {
    const int N = vertices_ndc.size(0);
    const int C = vertice_values.size(1);
    const int M = faces.size(0);

    const int gpuid = vertices_ndc.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));
    auto options = torch::dtype(vertices_ndc.scalar_type()).device(torch::kCUDA, gpuid);

    const dim3 dimGrid(M);
    const dim3 dimBlock(4, 4);

    auto depth = torch::ones({H, W}, options) * 1e10;
    auto result = torch::zeros({H, W, C}, options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vertices_ndc.scalar_type(), "project_mesh_cuda_kernel", [&] {
        scalar_t* global_face_ndc_inv;
        cudaMalloc(&global_face_ndc_inv, M * 9 * sizeof(scalar_t));
        int* global_is_bad_face;
        cudaMalloc(&global_is_bad_face, M * sizeof(int));
        rasterize_cuda_kernel<scalar_t><<<dimGrid, dimBlock>>>(
            vertices_ndc.packed_accessor32<scalar_t,2>(),
            faces.packed_accessor32<int32_t,2>(),
            vertices_filter.packed_accessor32<uint8_t,1>(),
            depth.packed_accessor32<scalar_t,2>(),
            global_face_ndc_inv,
            global_is_bad_face
        );
        AT_CUDA_CHECK(cudaGetLastError());

        interpolate_cuda_kernel<scalar_t><<<dimGrid, dimBlock>>>(
            vertices_ndc.packed_accessor32<scalar_t,2>(),
            faces.packed_accessor32<int32_t,2>(),
            depth.packed_accessor32<scalar_t,2>(),
            global_face_ndc_inv,
            global_is_bad_face,
            vertice_values.packed_accessor32<scalar_t,2>(),
            result.packed_accessor32<scalar_t,3>()
        );
        AT_CUDA_CHECK(cudaGetLastError());

        cudaFree(global_face_ndc_inv);
        cudaFree(global_is_bad_face);
        AT_CUDA_CHECK(cudaGetLastError());
    });

    return result;
}


std::vector<torch::Tensor> estimate_normals_cuda(
    const torch::Tensor& vertices_ndc,
    const torch::Tensor& faces,
    const torch::Tensor& vertices,
    const torch::Tensor& vertices_filter,
    int H, int W
) {
    const int N = vertices_ndc.size(0);
    const int M = faces.size(0);

    const int gpuid = vertices_ndc.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));
    auto options = torch::dtype(vertices_ndc.scalar_type()).device(torch::kCUDA, gpuid);

    const dim3 dimGrid(M);
    const dim3 dimBlock(4, 4);

    auto depth = torch::ones({H, W}, options) * 1e10;
    auto coords = torch::zeros({H, W, 3}, options);
    auto normals = torch::zeros({H, W, 3}, options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vertices_ndc.scalar_type(), "project_mesh_cuda_kernel", [&] {
        scalar_t* global_face_ndc_inv;
        cudaMalloc(&global_face_ndc_inv, M * 9 * sizeof(scalar_t));
        int* global_is_bad_face;
        cudaMalloc(&global_is_bad_face, M * sizeof(int));
        rasterize_cuda_kernel<scalar_t><<<dimGrid, dimBlock>>>(
            vertices_ndc.packed_accessor32<scalar_t,2>(),
            faces.packed_accessor32<int32_t,2>(),
            vertices_filter.packed_accessor32<uint8_t,1>(),
            depth.packed_accessor32<scalar_t,2>(),
            global_face_ndc_inv,
            global_is_bad_face
        );
        AT_CUDA_CHECK(cudaGetLastError());

        estimate_normals_cuda_kernel<scalar_t><<<dimGrid, dimBlock>>>(
            vertices_ndc.packed_accessor32<scalar_t,2>(),
            faces.packed_accessor32<int32_t,2>(),
            depth.packed_accessor32<scalar_t,2>(),
            global_face_ndc_inv,
            global_is_bad_face,
            vertices.packed_accessor32<scalar_t,2>(),
            coords.packed_accessor32<scalar_t,3>(),
            normals.packed_accessor32<scalar_t,3>()
        );
        AT_CUDA_CHECK(cudaGetLastError());

        cudaFree(global_face_ndc_inv);
        cudaFree(global_is_bad_face);
        AT_CUDA_CHECK(cudaGetLastError());
    });

    return {coords, normals};
}

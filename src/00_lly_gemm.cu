#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"

template <typename T>
__global__ void sgemm_naive(size_t m, size_t n, size_t k, T alpha, T const* A,
                            size_t lda, T const* B, size_t ldb, T beta, T* C,
                            size_t ldc)
{
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if (x < m && y < n)
    {
        T tmp = 0.0;
        for (int i = 0; i < k; ++i)
        {
            tmp += A[x * k + i] * B[i * n + y];
        }
        // C = α*(A@B)+β*C
        C[x * n + y] = alpha * tmp + beta * C[x * n + y];
    }
}

template <typename T>
void launch_lly_gemm_kernel_v00(size_t m, size_t n, size_t k, T const* alpha,
                                T const* A, size_t lda, T const* B, size_t ldb,
                                T const* beta, T* C, size_t ldc,
                                cudaStream_t stream)
{
    dim3 const block_dim{16U, 16U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};
    sgemm_naive<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda,
                                                        B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_lly_gemm_kernel_v00<float>(
    size_t m, size_t n, size_t k, float const* alpha, float const* A,
    size_t lda, float const* B, size_t ldb, float const* beta, float* C,
    size_t ldc, cudaStream_t stream);
template void launch_lly_gemm_kernel_v00<double>(
    size_t m, size_t n, size_t k, double const* alpha, double const* A,
    size_t lda, double const* B, size_t ldb, double const* beta, double* C,
    size_t ldc, cudaStream_t stream);
template void launch_lly_gemm_kernel_v00<__half>(
    size_t m, size_t n, size_t k, __half const* alpha, __half const* A,
    size_t lda, __half const* B, size_t ldb, __half const* beta, __half* C,
    size_t ldc, cudaStream_t stream);
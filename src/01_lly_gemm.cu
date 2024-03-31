#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"
#include "wmma_lsmem.cuh"

// transpose B
template <typename T>
__global__ void lly_sgemm_v1(size_t m, size_t n, size_t k, T alpha, T const* A,
                             size_t lda, T const* B, size_t ldb, T beta, T* C,
                             size_t ldc)
{
    // A is row-major
    // B is col-major
    // 128 threads [x, y, z] = [32, 2, 2]
    // threadblock mma: 128x128x32
    // warp mma: 64x64x16
    extern __shared__ uint8_t shared_storage[];
    T* SA = reinterpret_cast<T*>(shared_storage); // 128x32
    T* SB =
        reinterpret_cast<T*>(shared_storage + MI * KI * sizeof(T)); // 32x128
    T* SC = reinterpret_cast<T*>(shared_storage);                   // 64x64

    const size_t fa_size = MII / wmmaM; // 4
    const size_t fb_size = NII / wmmaN; // 4
    const size_t fc_size = fa_size * fb_size;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, T,
                           nvcuda::wmma::row_major>
        FragA[fa_size];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, T,
                           nvcuda::wmma::col_major>
        FragB[fb_size];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, T>
        Accum[fc_size];

    for (int mii = 0; mii < fc_size; mii++)
    {
        nvcuda::wmma::fill_fragment(Accum[mii], 0.0);
    }
    for (int ko = 0; ko < k / KI; ko++)
    {
        loadSmemA(SA, A, m, k, ko);
        loadSmemB(SB, B, n, k, ko);
        __syncthreads();
        // step_k = 2
        for (int ki = 0; ki < KI / KII; ki++)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA, ki);
            loadFragB(FragB, SB, ki);
            for (int mii = 0; mii < fa_size; mii++)
            {
                for (int nii = 0; nii < fb_size; nii++)
                {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(Accum[mii * fb_size + nii],
                                           FragA[mii], FragB[nii],
                                           Accum[mii * fb_size + nii]);
                }
            }
        }
    }
    storeAccum(SC, Accum);
    __syncthreads();
    storeSmemC(C, SC, m, n);
}

template <typename T>
void launch_lly_gemm_kernel_v01(size_t m, size_t n, size_t k, T const* alpha,
                                T const* A, size_t lda, T const* B, size_t ldb,
                                T const* beta, T* C, size_t ldc,
                                cudaStream_t stream)
{
    dim3 const block_dim{32u, 2u, 2u};
    dim3 const grid_dim{(unsigned int)(n + 127) / 128u,
                        (unsigned int)(m + 127) / 128u};
    const int smem_size = 128 * 128 * sizeof(T);
    lly_sgemm_v1<T><<<grid_dim, block_dim, smem_size, stream>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_lly_gemm_kernel_v01<__half>(
    size_t m, size_t n, size_t k, __half const* alpha, __half const* A,
    size_t lda, __half const* B, size_t ldb, __half const* beta, __half* C,
    size_t ldc, cudaStream_t stream);
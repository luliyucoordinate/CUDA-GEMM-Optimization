#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"
#include "wmma_lsmem.cuh"

// transpose B
template <typename T>
__global__ void lly_sgemm_v2(size_t M, size_t N, size_t K, T alpha, T const* A,
                             size_t lda, T const* B, size_t ldb, T beta, T* C,
                             size_t ldc)
{
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();

    constexpr size_t stages_count = 4; // Pipeline with four stage
    // Allocate shared storage for a single stage cuda::pipeline:
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block, stages_count>
        shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // A is row-major
    // B is col-major
    // 128 threads [x, y, z] = [32, 2, 2]
    // threadblock mma: 128x128x32
    // warp mma: 64x64x16
    extern __shared__ uint8_t shared_storage[];
    T* SA1 = (T*)shared_storage;
    T* SA2 = SA1 + MI * KI;
    T* SA3 = SA2 + MI * KI;
    T* SA4 = SA3 + MI * KI;
    T* SB1 = SA4 + MI * KI;
    T* SB2 = SB1 + NI * KI;
    T* SB3 = SB2 + NI * KI;
    T* SB4 = SB3 + NI * KI;
    T* SC = (T*)shared_storage;

    const size_t fa_size = MII / wmmaM; // 4
    const size_t fb_size = NII / wmmaN; // 4
    const size_t fc_size = fa_size * fb_size;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, T,
                           nvcuda::wmma::row_major>
        FragA[fa_size];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, T,
                           nvcuda::wmma::col_major>
        FragB[fb_size];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, half>
        Accum[fc_size];

    for (int mii = 0; mii < fc_size; mii++)
    {
        nvcuda::wmma::fill_fragment(Accum[mii], 0.0);
    }

    // prologue
    loadSmemA(SA1, A, M, K, 0);
    loadSmemB(SB1, B, N, K, 0);

    loadSmemA(SA2, A, M, K, 1);
    loadSmemB(SB2, B, N, K, 1);

    loadSmemA(SA3, A, M, K, 2);
    loadSmemB(SB3, B, N, K, 2);

    for (int ko = 0; ko < K / KI; ko += 4)
    {
        __syncthreads();
        if (ko + 3 < K / KI)
        {
            loadSmemA(SA4, A, M, K, ko + 3);
            loadSmemB(SB4, B, N, K, ko + 3);
        }
        for (int ki = 0; ki < KI / KII; ki++)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA1, ki);
            loadFragB(FragB, SB1, ki);
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

        __syncthreads();
        if (ko + 4 < K / KI)
        {
            loadSmemA(SA1, A, M, K, ko + 4);
            loadSmemB(SB1, B, N, K, ko + 4);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA2, ki);
            loadFragB(FragB, SB2, ki);
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

        __syncthreads();
        if (ko + 5 < K / KI)
        {
            loadSmemA(SA2, A, M, K, ko + 5);
            loadSmemB(SB2, B, N, K, ko + 5);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA3, ki);
            loadFragB(FragB, SB3, ki);
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

        __syncthreads();
        if (ko + 6 < K / KI)
        {
            loadSmemA(SA3, A, M, K, ko + 6);
            loadSmemB(SB3, B, N, K, ko + 6);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA4, ki);
            loadFragB(FragB, SB4, ki);
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
    storeSmemC(C, SC, M, N);
}

template <typename T>
void launch_lly_gemm_kernel_v02(size_t m, size_t n, size_t k, T const* alpha,
                                T const* A, size_t lda, T const* B, size_t ldb,
                                T const* beta, T* C, size_t ldc,
                                cudaStream_t stream)
{
    dim3 const block_dim{32u, 2u, 2u};
    dim3 const grid_dim{(unsigned int)(n + 127) / 128u,
                        (unsigned int)(m + 127) / 128u};
    const int smem_size =
        max(128 * 128 * sizeof(T), 4 * 128 * 32 * 2 * sizeof(T));
    if (smem_size >= (48 << 10))
    {
        cudaFuncSetAttribute(lly_sgemm_v2<T>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
    }
    lly_sgemm_v2<T><<<grid_dim, block_dim, smem_size, stream>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_lly_gemm_kernel_v02<half>(size_t m, size_t n, size_t k,
                                               half const* alpha, half const* A,
                                               size_t lda, half const* B,
                                               size_t ldb, half const* beta,
                                               half* C, size_t ldc,
                                               cudaStream_t stream);
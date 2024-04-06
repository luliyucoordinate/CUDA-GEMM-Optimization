#ifndef WMMA_LSMEM
#define WMMA_LSMEM

#include <cuda.h>
#include <mma.h>

const int MI = 128;
const int NI = 128;
const int KI = 32;
const int MII = 64;
const int NII = 64;
const int KII = 16;
const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 16;

template <typename T>
__device__ void loadSmemA(T* smem, const T* A, int M, int K, int ko)
{
    // load 128 * 32
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i)
    {
        int s_index = tid / 32 % 4 * 16 + tid / 16 % 2 * 256 + tid % 16 +
                      i / 4 * 512 + i % 4 * 64;
        int a_index =
            tid % 32 + tid / 32 * K + i * 4 * K + by * 128 * K + ko * KI;
        smem[s_index] = A[a_index];
    }
}

template <typename T>
__device__ void loadSmemB(T* smem, const T* B, int N, int K, int ko)
{
    // load 128 * 32
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i)
    {
        int s_index = tid / 32 % 4 * 16 + tid / 16 % 2 * 256 + tid % 16 +
                      i / 4 * 512 + i % 4 * 64;
        int a_index = (ko * KI + tid % 32) * N + tid / 32 + i * 4 + bx * 128;
        smem[s_index] = B[a_index];
    }
}

template <typename T>
__device__ void loadSmemC(T* smem, T* C, int M, int N)
{
    // load 128 * 128
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i)
    {
        int row = i;
        int col = tid;
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 +
             col % 16] = C[(by * 128 + row) * N + bx * 128 + col];
    }
}

template <typename T>
__device__ void storeSmemC(T* C, T* smem, int M, int N)
{
    // load 128 * 128
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i)
    {
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        (C[(by * 128 + i) * N + bx * 128 + tid]) =
            smem[tid % 16 + tid / 16 * 256 + i % 16 * 16 + i / 16 * 2048];
    }
}

template <typename T>
__device__ void
loadFragA(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, T,
                                 nvcuda::wmma::row_major>* frag,
          T* smem, int ki)
{
    // load 64x16
    int tz = threadIdx.z; // m
    for (int i = 0; i < 4; ++i)
    {
        nvcuda::wmma::load_matrix_sync(
            frag[i], smem + tz * 2048 + i * 512 + ki * 256, 16);
    }
}

template <typename T>
__device__ void
loadFragB(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, T,
                                 nvcuda::wmma::col_major>* frag,
          T* smem, int ki)
{
    // load 64x16
    int ty = threadIdx.y;
    for (int i = 0; i < 4; ++i)
    {
        nvcuda::wmma::load_matrix_sync(
            frag[i], smem + ty * 2048 + i * 512 + ki * 256, 16);
    }
}

template <typename T>
__device__ void storeAccum(T* ptr,
                           nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                                                  wmmaM, wmmaN, wmmaK, T>* frag)
{
    // store 64x64
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 2 + ty;
    for (int i = 0; i < 16; i++)
    {
        nvcuda::wmma::store_matrix_sync(
            ptr + tid % 2 * 1024 + tid / 2 * 8192 + i % 4 * 256 + i / 4 * 2048,
            frag[i], 16, nvcuda::wmma::mem_row_major);
    }
}

#endif
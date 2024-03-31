#ifndef WMMA_LSMEM
#define WMMA_LSMEM

#include <cuda.h>
#include <mma.h>

const int MI = 128;
// const int NI = 128;
const int KI = 32;
const int MII = 64;
const int NII = 64;
const int KII = 16;
const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 16;

__device__ void loadSmemA(half* smem, const half* A, int M, int K, int ko)
{
    // load 128 * 32
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // tid 0
        // (0, 0)  -> 0
        // (4, 0)  -> 4*16 -> 2*32
        // (8, 0)  -> 8*16 -> 4*32
        // (12, 0) -> 12*16 -> 6*32
        // (16, 0) -> 32*16 -> 16*32
        // (20, 0) -> 32*16 + 4*16 -> 18*32
        // (16*4, 0) -> 128*16 -> 64*32
        // tid 8
        // (0, 8) ->
        // tid 16
        // (0, 16) -> 8*32
        // (4, 16) -> 10*32
        // tid 32
        // (1, 0) -> 16
        // tid 64
        // (2, 0) -> 1*32
        // tid 96
        // (3, 0) -> 3*16
        // smem layout: (16, 8, 32)
        // A layout: (8, 2, 16, 16) 
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 +
             col % 16] = A[(by * 128 + row) * K + ko * KI + col];
    }
}

__device__ void loadSmemB(half* smem, const half* B, int N, int K, int ko)
{
    // load 128 * 32
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        // (N, K) (bx*128+row, ko*KI+col) -> (K, N) (ko*KI+col, bx*128+row)
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 +
             col % 16] = B[(bx * 128 + row) + (ko * KI + col) * N];
    }
}

__device__ void loadSmemC(half* smem, half* C, int M, int N)
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

__device__ void storeSmemC(half* C, half* smem, int M, int N)
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
        (C[(by * 128 + row) * N + bx * 128 + col]) =
            smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) +
                 row % 16 * 16 + col % 16];
    }
}

__device__ void
loadFragA(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK,
                                 half, nvcuda::wmma::row_major>* frag,
          half* smem, int ki)
{
    // load 64x16
    int tz = threadIdx.z; // m
    for (int i = 0; i < 4; ++i)
    {
        int row = tz * 64 + i * 16;
        int col = ki * KII;
        nvcuda::wmma::load_matrix_sync(
            frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16),
            16);
    }
}

__device__ void
loadFragB(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK,
                                 half, nvcuda::wmma::col_major>* frag,
          half* smem, int ki)
{
    // load 64x16
    int ty = threadIdx.y;
    for (int i = 0; i < 4; ++i)
    {
        int row = ty * 64 + i * 16;
        int col = ki * KII;
        nvcuda::wmma::load_matrix_sync(
            frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16),
            16);
    }
}

__device__ void
storeAccum(half* ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM,
                                             wmmaN, wmmaK, half>* frag)
{
    // store 64x64
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            int row = tz * 64 + i * 16;
            int col = ty * 64 + j * 16;
            // laoyut: [8, 8, 16, 16]
            nvcuda::wmma::store_matrix_sync(
                ptr + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16),
                frag[i * 4 + j], 16, nvcuda::wmma::mem_row_major);
        }
    }
}

#endif
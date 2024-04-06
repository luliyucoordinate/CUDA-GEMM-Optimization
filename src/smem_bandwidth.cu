#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <mma.h>

const int WARMUP = 100;
const int BLOCK = 256;

// number of LDS instructions to be timed
const int ROUND = 512;

__global__ void load_smem_v1(const half* A, int K)
{
    extern __shared__ half smem[];
    // load 128 * 32
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 +
             col % 16] = A[row * K + col];
    }
}

__global__ void load_smem_v2(const half* A, int K)
{
    extern __shared__ half smem[];
    // load 128 * 32
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 2; i++)
    {
        int row = tid / 2, col = tid % 2;
        for (int j = 0; j < 16; j++)
        {
            // A data layout: (8, 2, 16, 16) -> (2, 8, 8, 2, 16):(8, 8, 2)
            // smem data layout: (16, 8, 32) -> (2, 8, 8, 2, 16):(8, 8, 2)
            // thread layout: (128) -> (8, 8, 2)
            // [:, :, 0] -> [:, 0, :]
            // [:, :, 1] -> [:, 1, :]
            // (128, 32) -> ((8, 8, 2), (2, 16)) -> (2, (8, 8, 2), 16) ->
            // (16, 8, 2, 16)
            // (8, 2) ->(trans) (2, 8) (x, y)->(y, x)
            // (2, 8) ->(layout) (8, 2) t = (y * 8 + z) (t / 2, t % 2)
            int A_index = (row + i * 64) * K + col * 16 + j;
            int ax = A_index / 256, ay = A_index / 32 % 8,
                az = A_index / 16 % 2, am = A_index % 16;
            int t = az * 8 + ay;
            int sy = t / 2, sz = t % 2;
            // T32 (0, 1, 0, 0) -> (0, 0, 1, 0) T16
            // T64 (0, 2, 0, 0) -> (0, 0, 2, 0) T32
            // T16 (0, 0, 1, 0) -> (0, 4, 0, 0) T128
            int A_cindex = ax * 256 + ay * 32 + az * 16 + am;
            int S_index = ax * 256 + sy * 32 + sz * 16 + am;
            smem[S_index] = A[A_cindex];
        }
    }
}

__global__ void load_smem_v3(const half* A, half* B, int K)
{
    extern __shared__ half smem[];
    // load 128 * 32
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;

    for (int i = 0; i < 32; i++)
    {
        int s_index = tid / 16 * 512 + tid % 16 * 16 + i / 16 * 256 + i % 16;
        int a_index = tid * 32 + i;
        smem[s_index] = A[a_index];
        B[s_index] = smem[s_index];
    }
}

__global__ void load_smem_v4(const half* A, half* B, int K)
{
    extern __shared__ half smem[];
    // load 128 * 32
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;

    for (int i = 0; i < 32; i++)
    {
        int s_index = tid / 32 % 4 * 16 + tid / 16 % 2 * 256 + tid % 16 +
                      i / 4 * 512 + i % 4 * 64;
        int a_index = tid + i * 128;
        smem[s_index] = A[a_index];
        B[s_index] = smem[s_index];
    }
}

__global__ void smem_bandwidth_kernel(int* ret, uint32_t* clk_start,
                                      uint32_t* clk_stop)
{
    __shared__ int4 smem[BLOCK + ROUND];

    uint32_t tid = threadIdx.x;

    uint32_t start;
    uint32_t stop;
    uint32_t smem_addr;
    int4 reg = make_int4(tid, tid + 1, tid + 2, tid + 3);

    asm volatile("{.reg .u64 u64addr;\n"
                 " cvta.to.shared.u64 u64addr, %1;\n"
                 " cvt.u32.u64 %0, u64addr;}\n"
                 : "=r"(smem_addr)
                 : "l"(smem + tid));

    asm volatile("bar.sync 0;\n"
                 "mov.u32 %0, %%clock;\n"
                 : "=r"(start)
                 :
                 : "memory");

#pragma unroll
    for (int i = 0; i < ROUND; ++i)
    {
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
                     :
                     : "r"(smem_addr + i * (uint32_t)sizeof(int4)) "r"(reg.x),
                       "r"(reg.y), "r"(reg.z), "r"(reg.w)
                     : "memory");
    }

    asm volatile("bar.sync 0;\n"
                 "mov.u32 %0, %%clock;\n"
                 : "=r"(stop)
                 :
                 : "memory");

    if (threadIdx.x % 32 == 0)
    {
        clk_start[threadIdx.x / 32] = start;
        clk_stop[threadIdx.x / 32] = stop;
    }

    // dummy write back
    int tmp = ((int*)smem)[tid];
    if (tmp < 0)
    {
        *ret = tmp;
    }
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (prop.major == 3)
    {
        // enable 64-bit bank for Kepler GPU
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        if (BLOCK < 256)
        {
            // 2 warps each schedular for Kepler GPU
            printf("thread block size is not enough to utilze all LSU.\n");
        }
    }
    else
    {
        if (BLOCK < 128)
        {
            // 1 warp each schedular for Maxwell+ GPU
            printf("thread block size is not enough to utilze all LSU.\n");
        }
    }

    int* d_ret;
    uint32_t* d_clk_start;
    uint32_t* d_clk_stop;
    cudaMalloc(&d_ret, BLOCK * sizeof(int));
    cudaMalloc(&d_clk_start, BLOCK / 32 * sizeof(uint32_t));
    cudaMalloc(&d_clk_stop, BLOCK / 32 * sizeof(uint32_t));

    // pupulate l0/l1 i-cache
    for (int i = 0; i < WARMUP; ++i)
    {
        smem_bandwidth_kernel<<<1, BLOCK>>>(d_ret, d_clk_start, d_clk_stop);
    }

    // shared memory bandwidth benchmark
    smem_bandwidth_kernel<<<1, BLOCK>>>(d_ret, d_clk_start, d_clk_stop);

    uint32_t h_clk_start[BLOCK];
    uint32_t h_clk_stop[BLOCK];
    cudaMemcpy(h_clk_start, d_clk_start, BLOCK / 32 * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clk_stop, d_clk_stop, BLOCK / 32 * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    uint32_t start_min = ~0;
    uint32_t stop_max = 0;

    for (int i = 0; i < BLOCK / 32; ++i)
    {
        if (h_clk_start[i] < start_min)
        {
            start_min = h_clk_start[i];
        }
        if (h_clk_stop[i] > stop_max)
        {
            stop_max = h_clk_stop[i];
        }
    }

    uint32_t smem_size = BLOCK * ROUND * sizeof(int4);
    uint32_t duration = stop_max - start_min;
    float bw_measured = float(smem_size) / duration;
    // round up by 32
    uint32_t bw_theoretical = ((uint32_t)bw_measured + 31) / 32 * 32;

    printf("shared memory accessed: %u byte\n", smem_size);
    printf("duration: %u cycles\n", duration);
    printf("shared memory bandwidth per SM (measured): %f byte/cycle\n",
           bw_measured);
    printf("shared memory bandwidth per SM (theoretical): %u byte/cycle\n",
           bw_theoretical);

    uint32_t clk = prop.clockRate / 1000;
    uint32_t sm = prop.multiProcessorCount;
    float chip_bandwidth = float(sm) * bw_theoretical * clk / 1000;
    printf("standard clock frequency: %u MHz\n", clk);
    printf("SM: %u\n", sm);
    printf("whole chip shared memory bandwidth (theoretical): %f GB/s\n",
           chip_bandwidth);

    // test load smem
    dim3 const block_dim{32u, 2u, 2u};
    dim3 const grid_dim{1, 1};
    half* dummy_data;
    const uint32_t data_M = 128;
    const uint32_t data_N = 32;
    const uint32_t data_size = data_M * data_N;
    const uint32_t data_bytes = data_size * sizeof(half);

    cudaMalloc(&dummy_data, data_bytes);
    for (int i = 0; i < WARMUP; i++)
    {
        load_smem_v1<<<grid_dim, block_dim, data_bytes>>>(dummy_data, data_N);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    load_smem_v1<<<grid_dim, block_dim, data_bytes>>>(dummy_data, data_N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("load_v1 shared memory bandwidth (theoretical): %f GB/s\n",
           data_bytes / ms / 1e6);

    for (int i = 0; i < WARMUP; i++)
    {
        load_smem_v2<<<grid_dim, block_dim, data_bytes>>>(dummy_data, data_N);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    load_smem_v2<<<grid_dim, block_dim, data_bytes>>>(dummy_data, data_N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("load_v2 shared memory bandwidth (theoretical): %f GB/s\n",
           data_bytes / ms / 1e6);

    cudaFree(dummy_data);

    cudaFree(d_ret);
    cudaFree(d_clk_start);
    cudaFree(d_clk_stop);

    return 0;
}

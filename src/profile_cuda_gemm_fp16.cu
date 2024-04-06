#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_gemm.hpp"
#include "profile_utils.cuh"

int main()
{
    print_device_info();

    constexpr size_t num_repeats{1U};
    constexpr size_t num_warmups{1U};

    __half const fp16_abs_tol{__float2half(5.0e-2f)};
    double const fp16_rel_tol{1.0e-1f};

    constexpr size_t m{5376u};
    constexpr size_t k{2048u};
    constexpr size_t n{5376u};

    constexpr size_t lda{(k + 16U - 1U) / 16U * 16U};
    constexpr size_t ldb{(n + 16U - 1U) / 16U * 16U};
    constexpr size_t ldc{(n + 16U - 1U) / 16U * 16U};

    static_assert(lda >= k);
    static_assert(ldb >= n);
    static_assert(ldc >= n);

    std::cout << "Matrix Size: "
              << "M = " << m << " N = " << n << " K = " << k << std::endl;
    std::cout << "Matrix A: " << m << " x " << k
              << " Leading Dimension Size = " << lda << std::endl;
    std::cout << "Matrix B: " << k << " x " << n
              << " Leading Dimension Size = " << ldb << std::endl;
    std::cout << "Matrix C: " << m << " x " << n
              << " Leading Dimension Size = " << ldc << std::endl;
    std::cout << std::endl;

    // Define all the GEMM kernel launch functions to be profiled.
    std::vector<std::pair<
        std::string,
        std::function<void(size_t, size_t, size_t, __half const*, __half const*,
                           size_t, __half const*, size_t, __half const*,
                           __half*, size_t, cudaStream_t)>>> const
        gemm_fp16_kernel_launch_functions{
            // {"Custom lly GEMM Kernel V00", launch_lly_gemm_kernel_v00<__half>},
            // {"Custom lly GEMM Kernel V01", launch_lly_gemm_kernel_v01<__half>},
            {"Custom lly GEMM Kernel V02", launch_lly_gemm_kernel_v02<__half>},
        };

    for (auto const& gemm_fp16_kernel_launch_function :
         gemm_fp16_kernel_launch_functions)
    {
        std::cout << gemm_fp16_kernel_launch_function.first << std::endl;
        std::pair<__half, __half> const gemm_kernel_profile_result{
            profile_gemm<__half>(
                m, n, k, lda, ldb, ldc, gemm_fp16_kernel_launch_function.second,
                fp16_abs_tol, fp16_rel_tol, num_repeats, num_warmups)};
        std::cout << std::endl;
    }

    return 0;
}
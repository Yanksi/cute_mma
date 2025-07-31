#pragma once

#include <cuda_fp16.h>

#define GEMM_LIKELY(x) __builtin_expect(!!(x), 1)
#define GEMM_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define GEMM_CHECK_CUDA(call)                                                                   \
    do {                                                                                        \
        cudaError_t status = call;                                                              \
        if (GEMM_UNLIKELY(status != cudaSuccess)) {                                             \
            throw std::runtime_error("CUDA call failed with status " + std::to_string(status)); \
        }                                                                                       \
    } while (0)
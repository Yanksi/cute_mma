#pragma once

#include <cuda_fp16.h>
#include <cublas_v2.h>

#define GEMM_LIKELY(x) __builtin_expect(!!(x), 1)
#define GEMM_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define GEMM_CHECK_CUBLAS(call)                                                                   \
    do {                                                                                          \
        cublasStatus_t status = call;                                                             \
        if (GEMM_UNLIKELY(status != CUBLAS_STATUS_SUCCESS)) {                                     \
            throw std::runtime_error("CUBLAS call failed with status " + std::to_string(status)); \
        }                                                                                         \
    } while (0)

#define GEMM_CHECK_CUDA(call)                                                                   \
    do {                                                                                        \
        cudaError_t status = call;                                                              \
        if (GEMM_UNLIKELY(status != cudaSuccess)) {                                             \
            throw std::runtime_error("CUDA call failed with status " + std::to_string(status)); \
        }                                                                                       \
    } while (0)

template <typename TA>
std::string getTypeName() {
    static_assert(sizeof(TA) == 0, "This function should not be called");
    return "";
}

template <>
std::string getTypeName<float>() {
    return "float";
}

template <>
std::string getTypeName<half>() {
    return "half";
}
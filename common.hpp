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

template <typename TA>
constexpr
int paddingSize() {
    static_assert(sizeof(TA) == 0, "This function should not be called");
    return 0;
}

template <>
constexpr
int paddingSize<float>() {
    return 4;
}

template <>
constexpr
int paddingSize<half>() {
    return 8;
}

namespace cute {
    enum CUTE_MMA_Layout { CUTE_MMA_T, CUTE_MMA_N };

    template <typename TO, typename TR, CUTE_MMA_Layout ALayout, CUTE_MMA_Layout BLayout>
    struct Params {
        static_assert(sizeof(TO) == 0, "This struct should not be used");
    };
}
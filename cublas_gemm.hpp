#pragma once

#include <cublas_v2.h>

#define GEMM_CHECK_CUBLAS(call)                                                                   \
    do {                                                                                          \
        cublasStatus_t status = call;                                                             \
        if (GEMM_UNLIKELY(status != CUBLAS_STATUS_SUCCESS)) {                                     \
            throw std::runtime_error("CUBLAS call failed with status " + std::to_string(status)); \
        }                                                                                         \
    } while (0)

void getCublasTensorOpHandle(cublasHandle_t* handle) {
    GEMM_CHECK_CUBLAS(cublasCreate(handle));
    GEMM_CHECK_CUBLAS(cublasSetMathMode(*handle, CUBLAS_TF32_TENSOR_OP_MATH));
}

template <typename TA>
void gemm_cublas(
    char transA, char transB, int m, int n, int k,
    TA const* A, int ldA,
    TA const* B, int ldB,
    TA      * C, int ldC,
    cublasHandle_t* handle
) {
    cublasOperation_t cuTransA = (toupper(transA) == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB = (toupper(transB) == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
    TA alpha = 1.0;
    TA beta = 0.0;
    if constexpr (std::is_same<TA, float>::value) {
        GEMM_CHECK_CUBLAS(cublasSgemm(*handle, cuTransA, cuTransB, m, n, k, &alpha, A, ldA, B, ldB, &beta, C, ldC));
    } else if constexpr (std::is_same<TA, half>::value) {
        GEMM_CHECK_CUBLAS(cublasHgemm(*handle, cuTransA, cuTransB, m, n, k, &alpha, A, ldA, B, ldB, &beta, C, ldC));
    } else {
        static_assert(sizeof(TA) == 0, "Unsupported type");
    }
}

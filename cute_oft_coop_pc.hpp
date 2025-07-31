#pragma once
#include <cute/tensor.hpp>

struct CurrKernelParams {
    static const unsigned int group_size = 256;
    static const unsigned int reconn_sz = 8;
};

template <class KernelParams>
void oft_tn(int m, int n, int k,
        half const* A, int ldA,
        half const* B, int ldB,
        half const* R, int ldR,
        half      * C, int ldC,
        cudaStream_t stream = 0);
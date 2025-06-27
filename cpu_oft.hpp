#pragma once

#include <thrust/host_vector.h>
#include <cute/tensor.hpp>

#define FLOAT float

void ez_gemm(
    Tensor A, Tensor B, Tensor C)
{
    for (int i = 0; i < size<0>(A); ++i)
    {
        for (int j = 0; j < size<0>(B); ++i)
        {
            FLOAT acc = static_cast<FLOAT>(C(i, j));
            for (int k = 0; k < size<1>(A); ++k)
            {
                acc += static_cast<FLOAT>(A(i, k)) * static_cast<FLOAT>(B(j, k));
            }
            C(i, j) = static_cast<half>(acc);
        }
    }
}

void cpu_oft(
    const thrust::host_vector<half> &h_A,
    const thrust::host_vector<half> &h_R,
    const thrust::host_vector<half> &h_B,
    thrust::host_vector<half> &h_C,
    int m, int group_size, int n_groups, int k,
    int reconn_sz,
    cublasHandle_t *handle,
    bool RW_mode)
{
    using namespace cute;
    int n = group_size * n_groups;
    int n_blocks = k / reconn_sz;
    Tensor h_A_tensor = flatten(
        make_tensor(
            h_A.data(),
            make_layout(
                make_shape(m, k),
                LayoutRight{}))
            .compose(
                make_tile(
                    _, make_layout(
                           make_shape(reconn_sz, n_blocks))))); // (M, RECONN_SZ, K_BLOCKS)

    Tensor h_R_tensor = flat_divide(
        make_tensor(
            h_R.data(),
            make_layout(
                make_shape(n_groups * reconn_sz, k),
                LayoutRight{})),
        make_tile(
            make_layout(reconn_sz),
            make_layout(reconn_sz))); // (RECONN_SZ, RECONN_SZ, N_GROUPS, K_BLOCKS)

    Tensor h_B_tensor = flat_divide(
        make_tensor(
            h_B.data(),
            make_layout(
                make_shape(n, k),
                LayoutRight{})),
        make_tile(
            make_layout(group_sz),
            make_layout(reconn_sz))); // (GROUP_SZ, RECONN_SZ, N_GROUPS, K_BLOCKS)

    Tensor h_C_tensor = flatten(
        make_tensor(
            h_C.data(),
            make_layout(
                make_shape(m, n),
                LayoutRight{}
            )
        ).compose(
            make_tile(
                _,
                make_layout(
                    make_shape(group_size, n_groups)
                )
            )
        )
    ); // (M, GROUP_SZ, N_GROUPS)

    thrust::host_vector<half> AR(m*reconn_sz);
    Tensor AR = make_tensor(
        AR.data(),
        make_layout(
            make_shape(m, reconn_sz),
            LayoutRight{})); // (M, RECONN_SZ)
    for (int g = 0; g < n_groups; ++g) {
        for (int b = 0; b < n_blocks; ++b) {
            fill(AR, static_cast<half>(0.0f));
            ez_gemm(
                h_A_tensor(_, _, b),
                h_R_tensor(_, _, g, b),
                AR
            );
            ez_gemm(
                AR,
                h_B_tensor(_, _, g, b),
                h_C_tensor(_, _, g)
            )
        }
    }
}
#include <cute/tensor.hpp>
#include <iostream>
#include <cute/atom/mma_atom.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <bool inplace, class TA, class TB, class TC, class MMA_ATOM, class Size>
__global__ static void inplace_mma_test(
    TA const *A, TB const *B, TC *C, MMA_ATOM atom, Size sz
) {
    using namespace cute;
    using mma_atom = MMA_Atom<MMA_ATOM>;
    Tensor mA = make_tensor(A, make_layout(
        make_shape(size<0>(typename mma_atom::Shape_MNK{}) * sz,
                   size<2>(typename mma_atom::Shape_MNK{})),
        LayoutRight{}
    ));
    Tensor mB = make_tensor(B, make_layout(
        make_shape(size<1>(typename mma_atom::Shape_MNK{}),
                   size<2>(typename mma_atom::Shape_MNK{})),
        LayoutRight{}
    ));
    Tensor mC = make_tensor(C, make_layout(
        make_shape(size<0>(typename mma_atom::Shape_MNK{}) * sz,
                   size<1>(typename mma_atom::Shape_MNK{})),
        LayoutRight{}
    ));

    TiledMMA mmaC = make_tiled_mma(
                    MMA_ATOM{},
                    make_layout(make_shape(_1{}, _1{})),
                    Tile<decltype(size<0>(mA)), decltype(size<0>(mB))>{});  // 16x8x8 TiledMMA
    
    ThrMMA thr_mma = mmaC.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(mC);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    Tensor tCgA = thr_mma.partition_A(mA);
    Tensor tCrA = thr_mma.make_fragment_A(tCgA);
    Tensor tCgB = thr_mma.partition_B(mB);
    Tensor tCrB = thr_mma.make_fragment_B(tCgB);

    if (threadIdx.x == 0) {
        print(tCrA); print("\n");
        print(tCrB); print("\n");
        print(tCrC); print("\n");
    }

    copy(tCgA, tCrA);
    copy(tCgB, tCrB);
    if constexpr (inplace) {
        gemm(mmaC, tCrA, tCrB, tCrA);
        copy(tCrA, tCgC);
    } else {
        copy(tCgC, tCrC);
        gemm(mmaC, tCrA, tCrB, tCrC);
        copy(tCrC, tCgC);
    }
    // gemm(mmaC, tCrA, tCrB, tCrC);
    // copy(tCrC, tCgC);
}

void test_inplace() {
    using namespace cute;
    thrust::host_vector<half_t> hA(16 * 8 * 3);
    thrust::host_vector<half_t> hB(8 * 8);
    thrust::host_vector<half_t> hC(16 * 8 * 3);

    for (int i = 0; i < 16 * 8 * 3; i++) {
        hA[i] = rand() / double(RAND_MAX);
    }

    for (int i = 0; i < 8 * 8; i++) {
        hB[i] = rand() / double(RAND_MAX);
        int ii = i / 8;
        int jj = i % 8;
        // if (ii == jj) {
        //     hB[i] = hB[i] - 1.0;
        // }
    }

    for (int i = 0; i < 16 * 8 * 3; i++) {
        hC[i] = hA[i];
    }

    thrust::device_vector<half_t> dA = hA;
    thrust::device_vector<half_t> dB = hB;
    thrust::device_vector<half_t> dC = hC;

    dim3 dimBlock(32);
    dim3 dimGrid(1);

    inplace_mma_test<false><<<dimGrid, dimBlock>>>(dA.data().get(), dB.data().get(), dC.data().get(), SM80_16x8x8_F16F16F16F16_TN{}, _3{});

    thrust::host_vector<half_t> hC2 = dC;
    for (int i = 0; i < 16 * 8 * 3; i++) {
        std::cout << hC2[i] << " ";
    }

    std::cout << std::endl;

    inplace_mma_test<true><<<dimGrid, dimBlock>>>(dA.data().get(), dB.data().get(), dC.data().get(), SM80_16x8x8_F16F16F16F16_TN{}, _3{});

    hC2 = dC;
    for (int i = 0; i < 16 * 8 * 3; i++) {
        std::cout << hC2[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    test_inplace();
    return 0;
}
#pragma once
#include <common.hpp>
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>

// #define CUTE_MMA_DTYPE_O float
// #define CUTE_MMA_DTYPE_R float

namespace cute {
    template <>
    struct Params <float, float, CUTE_MMA_T, CUTE_MMA_N> {
        static const int bM = 128;
        static const int bN = 256;
        static const int bK = 8;
        static const int bP = 3;
        static const bool block_tiling_copy = true;
        using warp_layout = Layout<Shape<Int<2>, Int<4>>>;
        using mma_atom = SM80_16x8x4_F32TF32TF32F32_TN;
    };

    template <>
    struct Params <float, float, CUTE_MMA_N, CUTE_MMA_T> {
        static const int bM = 128;
        static const int bN = 128;
        static const int bK = 16;
        static const int bP = 3;
        static const bool block_tiling_copy = true;
        using warp_layout = Layout<Shape<Int<2>, Int<2>>>;
        using mma_atom = SM80_16x8x8_F32TF32TF32F32_TN;
    };

    template <>
    struct Params <half, half, CUTE_MMA_T, CUTE_MMA_N> {
        static const int bM = 256;
        static const int bN = 256;
        static const int bK = 16;
        static const int bP = 3;
        static const bool block_tiling_copy = true;
        using warp_layout = Layout<Shape<Int<4>, Int<2>>>;
        using mma_atom = SM80_16x8x8_F16F16F16F16_TN;
    };

    template <>
    struct Params <half, half, CUTE_MMA_N, CUTE_MMA_T> {
        static const int bM = 128;
        static const int bN = 256;
        static const int bK = 32;
        static const int bP = 2;
        static const bool block_tiling_copy = true;
        using warp_layout = Layout<Shape<Int<1>, Int<4>>>;
        using mma_atom = SM80_16x8x8_F16F16F16F16_TN;
    };
}
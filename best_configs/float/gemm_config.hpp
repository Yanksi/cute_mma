#pragma once
#include <cute/layout.hpp>
using namespace cute;
#define DTYPE float
namespace ParamTN {
    const static int bM = 128;
    const static int bN = 256;
    const static int bK = 8;
    const static int bP = 3;
    const static bool block_tiling_copy = true;
    using warp_layout = Layout<Shape<Int<2>, Int<4>>>;
    using mma_atom = SM80_16x8x4_F32TF32TF32F32_TN;
}

namespace ParamNT {
    const static int bM = 128;
    const static int bN = 128;
    const static int bK = 16;
    const static int bP = 3;
    const static bool block_tiling_copy = true;
    using warp_layout = Layout<Shape<Int<2>, Int<2>>>;
    using mma_atom = SM80_16x8x8_F32TF32TF32F32_TN;
}
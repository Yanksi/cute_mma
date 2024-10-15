#pragma once
#include <cute/layout.hpp>
using namespace cute;
#define DTYPE half
#define IS_FLOAT 0
namespace ParamTN {
    const static int bM = 256;
    const static int bN = 256;
    const static int bK = 16;
    const static int bP = 3;
    const static bool block_tiling_copy = true;
    using warp_layout = Layout<Shape<Int<4>, Int<2>>>;
    using mma_atom = SM80_16x8x8_F16F16F16F16_TN;
}

namespace ParamNT {
    const static int bM = 128;
    const static int bN = 256;
    const static int bK = 32;
    const static int bP = 2;
    const static bool block_tiling_copy = true;
    using warp_layout = Layout<Shape<Int<1>, Int<4>>>;
    using mma_atom = SM80_16x8x8_F16F16F16F16_TN;
}
#pragma once
#include <cute/tensor.hpp>

struct CurrCompParams {
    static const unsigned int bM = 128;
    static const unsigned int bN = 256;
    static const unsigned int bK = 16;
    static const unsigned int bP1 = 3;
    static const unsigned int bP2 = 2;
    using warp_layout1 = cute::Layout<cute::Shape<cute::Int<2>>>;
    using warp_layout2 = cute::Layout<cute::Shape<cute::Int<2>, cute::Int<4>>>;
};

#pragma once
#include <cute/tensor.hpp>
namespace cute {

    struct CurrCompParams {
        static const unsigned int bM = 128;
        static const unsigned int bN = 128;
        static const unsigned int bK = 32;
        static const unsigned int c_width = 16;
        static const unsigned int bP_a_r = 2;
        static const unsigned int bP_ar = 2;
        static const unsigned int bP_b = 2;
        using warp_layout1 = cute::Layout<cute::Shape<cute::Int<2>>>;
        using warp_layout2 = cute::Layout<cute::Shape<cute::Int<2>, cute::Int<2>>>;
    };

}
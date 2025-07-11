#pragma once
#include <cute/tensor.hpp>

template <class WARP_N, class N_GROUPS>
CUTE_HOST_DEVICE constexpr
auto warp_group_mapping(WARP_N warp_n, N_GROUPS n_groups) {
    using namespace cute;
    // would generate a layout with the size of (groups_per_warp, n_warps)
    if constexpr (n_groups >= warp_n) {
        // if the number of groups is greater or equal to the number of warps, then each warp will handle at least one group
        CUTE_STATIC_ASSERT(n_groups % warp_n == 0, "Number of groups must be divisible by number of warps.");
        return make_layout(make_shape(n_groups / warp_n, warp_n));
    } else {
        // if the number of groups is less than the number of warps, then a group would be handled by multiple warps
        CUTE_STATIC_ASSERT(warp_n % n_groups == 0, "Number of warps must be divisible by number of groups.");
        return make_layout(
            make_shape(_1{}, make_shape(warp_n / n_groups, n_groups)),
            make_stride(_0{}, make_stride(_0{}, _1{}))
        );
    }
}

template <class WARP_N, class N_GROUPS, class GROUP_SIZE>
CUTE_HOST_DEVICE constexpr
auto warp_in_group_mapping(WARP_N warp_n, N_GROUPS n_groups, GROUP_SIZE group_sz) {
    using namespace cute;
    // would generate a layout with size of (warp_responsible_size, warp_n)
    if constexpr(n_groups >= warp_n) {
        // if the number of groups is greater or equal to the number of warps, then each warp would handle a full group
        return make_layout(make_shape(group_sz, warp_n), make_stride(_1{}, _0{}));
    } else {
        auto warp_per_group = warp_n / n_groups;
        auto warp_responsible_size = group_sz / warp_per_group;
        return make_layout(
            make_shape(warp_responsible_size, make_shape(warp_per_group, n_groups)),
            make_stride(_1{}, make_stride(warp_responsible_size, _0{}))
        );
    }
}

template <int _k_width>
CUTE_HOST_DEVICE constexpr
auto get_smem_atom(cute::Int<_k_width>) {
    using namespace cute;
    /*
    k=64: 3,3
    k=32: 2,4
    k=16: 1,5
    k= 8: 0,6
    */
    Int<_k_width> k_width;
    CUTE_STATIC_ASSERT(k_width % 8 == 0);
    CUTE_STATIC_ASSERT_V(k_width == bit_floor(k_width)); // k_width must be a power of two
    constexpr auto n_blocks = k_width / _8{};
    constexpr auto permutation_bits = log_2(static_cast<unsigned int>(_k_width)) - _3{};
    constexpr auto sw = Swizzle<permutation_bits, 6 - permutation_bits>{};
    return composition(
        sw, make_layout(
            make_shape(_8{}, make_shape(_8{}, n_blocks)),
            make_stride(_8{}, make_stride(_1{}, _64{}))
        )
    );
}

template <class CtaTiler, class ReconnSZ, class Pipeline>
size_t get_smem_size(CtaTiler cta_tiler, int group_sz, ReconnSZ reconn_sz, Pipeline pipeline) {
    using namespace cute;
    auto size_m = size<0>(cta_tiler);
    auto size_n = size<1>(cta_tiler);
    auto size_k = size<2>(cta_tiler);
    int size_A = size_m * size_k * pipeline;
    int size_B = size_n * size_k * pipeline;
    int n_groups = max(size_n / group_sz, 1); // Number of groups in the N dimension
    int size_R = reconn_sz * n_groups * size_k * pipeline;
    int size_smem = size_A + size_B + size_R;
    return size_smem * sizeof(half_t);
}
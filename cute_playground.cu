#include <cute/tensor.hpp>
#include <iostream>
#include <cute/atom/mma_atom.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define mmax(a,b) ((a) > (b) ? (a) : (b))
#define mmin(a,b) ((a) < (b) ? (a) : (b))

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
    // return composition(
    //     sw, make_layout(
    //         make_shape(_8{}, make_shape(_8{}, n_blocks)),
    //         make_stride(_8{}, make_stride(_1{}, _64{}))
    //     )
    // );
    return make_layout(
            make_shape(_8{}, make_shape(_8{}, n_blocks)),
            make_stride(_8{}, make_stride(_1{}, _64{}))
        );
    // return make_layout(
    //     make_shape(_2{}, k_width),
    //     LayoutRight{}
    // );
}

template <typename copy_as_t, typename ele_t, typename _BK, typename _N_Threads>
constexpr auto cp_layout(_BK bk, _N_Threads total_threads) {
  using namespace cute;
  auto vec_width = Int<sizeof(copy_as_t) / sizeof(ele_t)>{};
  auto threads_along_k = max(bk / vec_width, _1{});
  static_assert(is_static<decltype(threads_along_k)>::value, "threads_along_k must be static");
  auto threads_k_size = bk / threads_along_k;
  static_assert(is_static<decltype(threads_k_size)>::value, "threads_k_size must be static");
  auto threads_m_size = max(vec_width / bk, _1{});
  static_assert(is_static<decltype(threads_m_size)>::value, "threads_m_size must be static");
  auto threads_along_m = total_threads / threads_along_k;
  static_assert(is_static<decltype(threads_along_m)>::value, "threads_along_m must be static");
  return make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<copy_as_t>, ele_t>{},
                          make_layout(make_shape(threads_along_m, threads_along_k), LayoutRight{}),
                          make_layout(make_shape(threads_m_size, threads_k_size)));
}

int main() {
    using namespace cute;
    // auto swizzle_atom = composition(Swizzle<1, 5>{},
    //                               Layout<Shape <_8,Shape <_8, _2>>,
    //                                      Stride<_8,Stride<_1,_64>>>{});
    // auto sAl = tile_to_shape(swizzle_atom, make_shape(_64{},_16{}));
    // // print_latex(swizzle_atom); print("\n");
    // print_latex(sAl); print("\n");
    // // print_latex(select<0, 1>(sAl)); print("\n");
    // return 0;

    // auto a = _8{};
    // constexpr auto k_bit_a = log_2(static_cast<unsigned int>(decltype(a)::value));
    // print(Int<k_bit_a>{}); print("\n");
    // return 0;

    half* _M = new half[256*256*3];
    for (int i = 0; i < 256*256*3; ++i) {
        _M[i] = static_cast<half>(0.0f);
    }
    auto M = make_smem_ptr(_M);
    int group_sz = 128;
    auto reconn_sz = _8{};

    auto pipeline = _3{};
    auto cta_tiler = make_shape(
        _256{}, // BM
        _256{}, // BN
        _16{} // BK
    );

    int n_groups = size<1>(cta_tiler) / group_sz;

    auto smem_atom = get_smem_atom(size<2>(cta_tiler));
    
    auto sA_layout = coalesce(tile_to_shape(
        smem_atom,
        make_shape(
            size<0>(cta_tiler), // BLK_M
            size<2>(cta_tiler), // BLK_K
            pipeline            // PIPE
        )
    ), make_tuple(_1{}, _1{}, _1{})); // (BLK_M, BLK_K, PIPE)

    auto sB_layout = coalesce(tile_to_shape(
        smem_atom,
        make_shape(
            size<1>(cta_tiler), // BLK_N
            size<2>(cta_tiler), // BLK_K
            pipeline            // PIPE
        )
    ), make_tuple(_1{}, _1{}, _1{})); // (BLK_N, BLK_K, PIPE)

    auto sR_layout_2d = coalesce(tile_to_shape(
        smem_atom,
        make_shape(
            reconn_sz * n_groups,
            size<2>(cta_tiler),
            pipeline
            )
        ), make_tuple(_1{}, _1{}, _1{})); // (N_GROUPS * R, K, PIPE)
    
    auto sR_layout_4d = tiled_divide(
        sR_layout_2d,
        make_tile(
            make_layout(reconn_sz),
            make_layout(reconn_sz)
        )
    ); // ((R, R), BLK_N_GROUPS, BLK_K_GROUPS, PIPE)

    auto gC_layout = make_layout(
        make_shape(
            size<0>(cta_tiler),
            size<1>(cta_tiler)
        ),
        LayoutRight{}
    );

    Tensor sA = make_tensor(M, sA_layout);
    Tensor sB = make_tensor(M, sB_layout);
    Tensor sR = make_tensor(M, sR_layout_2d);
    Tensor gC = make_tensor(M, gC_layout);
   
    

    // Tensor r4d = make_tensor(M, sR_layout_4d);

    int warp_idx = 6;
    int lane_idx = 12;

    auto warp_layout = make_layout(
        make_shape(_4{}, _4{})
    );

    auto warp_coord = warp_layout.get_hier_coord(warp_idx);
    TiledCopy tiled_copy = cp_layout<uint128_t, half>(size<2>(cta_tiler), size(warp_layout) * _32{});

    for (int t = 0; t < size(warp_layout) * 32; ++t) {
        ThrCopy thr_copy = tiled_copy.get_slice(t);
        Tensor tRsR = thr_copy.partition_D(sR);
        int cp_sz = size(tRsR);
        int valid_threads = size(sR) / cp_sz;
        if (t < valid_threads) {
            for (int i = 0; i < cp_sz; ++i) {
                tRsR(i) += static_cast<half>(1.0f);
            }
        }
    }

    double total_sum = 0.0;
    for (int i = 0; i < 256*256*3; ++i) {
        total_sum += static_cast<double>(_M[i]);
    }
    std::cout << "Total sum: " << total_sum << std::endl;
    return 0;

    auto cta_atom_layout_n = make_layout(
        make_shape(
            size<1>(warp_layout), _8{} // WARPS_ALONG_N, 8
        ),
        LayoutRight{}
    );

    print(sB); print("\n");

    Tensor sB_warp_atom = logical_divide(
        sB,
        make_tile(
            cta_atom_layout_n
        )
    ); // ((WARPS_ALONG_N, 8), REST_N, BLK_K, PIPELINE)
    print(sB_warp_atom); print("\n");

    auto sB_warp_region = group_modes<0,2>(sB_warp_atom(
        make_coord(make_coord(get<1>(warp_coord), _), _), _, _
    )); // (WARP_N_REGION, BLK_K, PIPELINE)
    print(sB_warp_region); print("\n");
    using mma_atom2 = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;

    TiledMMA single_warp_mma2 = make_tiled_mma(
        mma_atom2{},
        make_layout(make_shape(_1{}, _1{})),
        Tile<_128, decltype(size<0>(sB_warp_region))>{}
    );

    ThrMMA thr_mma2 = single_warp_mma2.get_slice(lane_idx);
    Tensor tCsB = thr_mma2.partition_B(sB_warp_region); // (MMA, MMA_N, MMA_K, PIPELINE)
    print(tCsB); print("\n");

    auto tmp = tCsB.compose(
            make_tile(
                make_layout(
                    make_shape(_1{}, _2{})
                ), _, _, _
            )
        );
    print(tmp); print("\n");

    Tensor tCsB_grouped = logical_divide(
        tmp,
        make_tile(
            _, n_groups
        )
    );
    print(tCsB_grouped); print("\n");


    Tensor sR_warp_atom = flat_divide(
        sR,
        make_tile(
            make_layout(reconn_sz),
            make_layout(reconn_sz)
        )
    ); // (RECONN_SZ, RECONN_SZ, N_GROUPS, BLOCKS_ALONG_K, PIPELINE)
    print(sR_warp_atom(_, _, _0{}, _, _)); print("\n");
}
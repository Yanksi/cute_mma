#include <cute/tensor.hpp>
#include <iostream>
#include <cute/atom/mma_atom.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define mmax(a,b) ((a) > (b) ? (a) : (b))
#define mmin(a,b) ((a) < (b) ? (a) : (b))

template <typename copy_as_t, typename ele_t, bool k_major, bool block_tiling,
  typename _BM, typename _BK, typename _N_Threads>
constexpr auto cp_layout(_BM bm, _BK bk, _N_Threads _total_threads) {
  using namespace cute;
  constexpr int vec_width = sizeof(copy_as_t) / sizeof(ele_t);
  constexpr int total_elements = bm * bk;

  constexpr int needed_threads = total_elements / vec_width;
  CUTE_STATIC_ASSERT(total_elements % vec_width == 0);
  constexpr int total_threads = mmin(_total_threads, needed_threads);

  constexpr int elements_per_thread = total_elements / total_threads;
  CUTE_STATIC_ASSERT(total_elements % total_threads == 0);
  CUTE_STATIC_ASSERT(elements_per_thread % vec_width == 0);
  constexpr auto get_cp_width = []() {
    if constexpr (block_tiling) {
      return vec_width;
    } else {
      return elements_per_thread;
    }
  };
  constexpr int cp_width = get_cp_width();
  if constexpr (k_major) {
    CUTE_STATIC_ASSERT(!block_tiling || bk % cp_width == 0);
    CUTE_STATIC_ASSERT(block_tiling || (bk % cp_width == 0 || cp_width % bk == 0));
    constexpr int threads_along_k = mmax(bk / cp_width, 1);
    constexpr int threads_k_size = bk / threads_along_k;
    constexpr int threads_m_size = mmax(cp_width / bk, 1);
    constexpr int threads_along_m = total_threads / threads_along_k;
    return make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<copy_as_t>, ele_t>{},
                           make_layout(Shape<Int<threads_along_m>, Int<threads_along_k>>{}, LayoutRight{}),
                          //  Layout<Shape<Int<threads_along_m>, Int<threads_along_k>>>{},
                           Layout<Shape<Int<threads_m_size>, Int<threads_k_size>>>{});
  } else {
    // As it not really possible to have copy width greater than bm, we don't need to check for that
    CUTE_STATIC_ASSERT(bm % cp_width == 0);
    constexpr int threads_along_m = bm / cp_width;
    constexpr int threads_along_k = total_threads / threads_along_m;
    return make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<copy_as_t>, ele_t>{},
                           Layout<Shape<Int<threads_along_m>, Int<threads_along_k>>>{},
                           Layout<Shape<Int<cp_width>, _1>>{});
  }
}


template <class NWarp, class L>
CUTE_HOST_DEVICE constexpr
auto r_layout_mode1(NWarp n_warps, L layout_group) {
    using namespace cute;
    auto n_groups = size<0>(layout_group);
    if constexpr (n_groups >= n_warps) {
        // if the number of groups is greater or equal to the number of warps, then each warp will handle at least one group
        CUTE_STATIC_ASSERT(n_groups % n_warps == 0, "Number of groups must be divisible by number of warps.");
        return logical_divide(
            layout_group,
            make_shape(n_warps)
        );
    } else {
        // if the number of groups is less than the number of warps, then a group would be handled by multiple warps
        CUTE_STATIC_ASSERT(n_warps % n_groups == 0, "Number of warps must be divisible by number of groups.");
        auto n_warps_per_group = n_warps / n_groups;
        return make_layout(
            make_layout(
                Layout<decltype(n_warps_per_group), _0>{},
                layout_group
            ),
            Layout<_1, _0>{}
        );
    }
}

void ezprep() {
    // in this function, consider only the case where the first mma atom used is the 16x8x8 one for easier handling
    using namespace cute;
    
    half_t* M = new half_t[256*256*3];

    auto blocks_tiler = make_shape(
        _128{}, _1{}, _8{} // (BATCH_SZ, N_GROUPS, N_BLOCKS)
    );

    auto reconn_sz = _8{};
    auto group_sz = _128{};
    auto pipeline = _3{};

    auto cta_tiler = make_shape(
        size<0>(blocks_tiler), // BM
        size<1>(blocks_tiler) * group_sz, // BN
        size<2>(blocks_tiler) * reconn_sz // BK
    );

    auto smem_atom = make_layout( // two rows of padded shared memory layout atom, should be replaced by swizzle
        make_shape(_2{}, size<2>(cta_tiler)),
        make_stride(size<2>(cta_tiler) + _8{}, _1{})
    );

    print(cta_tiler); print("\n");
    print(smem_atom); print("\n");

    auto warp_layout = make_layout(
        make_shape(_4{}, _2{}, _1{})
    );

    auto warp_atom_mnk = make_shape(
        size<0>(cta_tiler) / size<0>(warp_layout), // WARP_ATOM_M
        size<1>(cta_tiler) / size<1>(warp_layout), // WARP_ATOM_N
        size<2>(cta_tiler) / size<2>(warp_layout)  // WARP_ATOM_K
    ); // A rather simple way to tile the warps, the shape of the tensor that each warp should handle

    // auto warp_atom_mnk = make_shape(_16{}, group_sz, _16{}); // the minimum sized block that each warp should work on at once

    // TiledMMA warp_mma = make_tiled_mma(SM80_16x8x8_F16F16F16F16_TN{});  // 16x8x8 TiledMMA
    using mma_atom1 = MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>;
    using mma_atom2 = mma_atom1;

    int thread_idx = 128;
    int warp_idx = thread_idx / 32;
    int lane_idx = thread_idx % 32;
    auto warp_idx_3d = warp_layout.get_hier_coord(warp_idx);
    // print(warp_idx_3d); print("\n");

    auto sA_layout = coalesce(tile_to_shape(
        smem_atom,
        make_shape(
            size<0>(cta_tiler),
            size<2>(cta_tiler),
            pipeline
            )
        ), make_tuple(_1{}, _1{}, _1{}));

    Tensor sA = make_tensor(M, sA_layout);
    {
        auto sA_warp_tile = make_tile(make_layout(size<0>(warp_atom_mnk)), 
                                      make_layout(size<2>(warp_atom_mnk)));
            
        auto a_tensor_layout = zipped_divide(take<0,2>(sA.layout()), sA_warp_tile); // ((WARP_ATOM_M, WARP_ATOM_K), (RESTM, RESTK))  ignore PIPE for now, describes how the matrix would be handled by the warps

        auto atom_tile = make_tile(make_layout(_16{}), make_layout(_8{})); // the minimum sized block that a warp should handle using mma atom during the reconnection stage
        
        // print(sA.layout()); print("\n");
        // print(a_tensor_layout); print("\n");
        // findout how each warp would have to handle the matrix in their region of responsibility
        auto mma_atom_layout = zipped_divide(get<0>(a_tensor_layout), atom_tile); // ((ATOM_SZ_M, ATOM_SZ_K), (ATOM_M, ATOM_K))
        print(mma_atom_layout); print("\n");

        auto mma_thr_layout = composition(mma_atom_layout, make_tile(mma_atom1::LayoutA_TV{}, _)); // ((THR, THR_DATA), (ATOM_M, ATOM_K))
        print(mma_thr_layout); print("\n");

        auto overall_layout = make_layout(mma_thr_layout, get<1>(a_tensor_layout), layout<2>(sA.layout()));
        print(overall_layout); print("\n");

        Tensor sA_blocked = make_tensor(M, overall_layout);
        Tensor thr_sA = sA_blocked(
            make_coord(make_coord(lane_idx, _), _),
            make_coord(get<0>(warp_idx_3d), get<2>(warp_idx_3d)),
            _
        );
        print(thr_sA.layout()); print("\n");
        Tensor thr_rA = make_tensor<half_t>(make_shape(shape<0>(thr_sA), shape<1,0>(thr_sA), _2{}));
        print(thr_rA.layout()); print("\n");
    }

    std::cout << "------------------" << std::endl;

    auto sB_layout = coalesce(tile_to_shape(
        smem_atom,
        make_shape(
            size<1>(cta_tiler),
            size<2>(cta_tiler),
            pipeline
            )
        ), make_tuple(_1{}, _1{}, _1{}));

    Tensor sB = make_tensor(M, sB_layout);
    {
        auto sB_warp_tile = make_tile(make_layout(size<1>(warp_atom_mnk)), 
                                      make_layout(size<2>(warp_atom_mnk)));
        print(sB_warp_tile); print("\n");
            
        auto b_tensor_layout = zipped_divide(take<0,2>(sB.layout()), sB_warp_tile); // ((WARP_ATOM_M, WARP_ATOM_K), (RESTM, RESTK))  ignore PIPE for now, describes how the matrix would be handled by the warps
        print(b_tensor_layout); print("\n");

        auto atom_tile = make_tile(make_layout(_8{}), make_layout(_8{})); // the minimum sized block that a warp should handle using mma atom during the reconnection stage

        // print(sA.layout()); print("\n");
        // print(a_tensor_layout); print("\n");
        // findout how each warp would have to handle the matrix in their region of responsibility
        auto mma_atom_layout = zipped_divide(get<0>(b_tensor_layout), atom_tile); // ((ATOM_SZ_M, ATOM_SZ_K), (ATOM_M, ATOM_K))
        print(mma_atom_layout); print("\n");

        auto mma_thr_layout = composition(mma_atom_layout, make_tile(mma_atom2::LayoutB_TV{}, _)); // ((THR, THR_DATA), (ATOM_M, ATOM_K))
        print(mma_thr_layout); print("\n");

        auto overall_layout = make_layout(mma_thr_layout, get<1>(b_tensor_layout), layout<2>(sB.layout()));
        print(overall_layout); print("\n");

        Tensor sB_blocked = make_tensor(M, overall_layout);
        Tensor thr_sB = sB_blocked(
            make_coord(make_coord(lane_idx, _), _),
            make_coord(get<1>(warp_idx_3d), get<2>(warp_idx_3d)),
            _
        );
        print(thr_sB.layout()); print("\n");
        Tensor thr_rB = make_tensor<half_t>(make_shape(shape<0>(thr_sB), shape<1,0>(thr_sB), _2{}));
        print(thr_rB.layout()); print("\n");
    }

    std::cout << "------------------" << std::endl;

    auto sR_layout = coalesce(tile_to_shape(
        smem_atom,
        make_shape(
            size<1>(blocks_tiler) * reconn_sz,
            size<2>(cta_tiler),
            pipeline
            )
        ), make_tuple(_1{}, _1{}, _1{}));
    print(sR_layout); print("\n");
    
    auto sR_layout_blocked = tiled_divide(
        sR_layout,
        make_tile(
            make_layout(reconn_sz),
            make_layout(reconn_sz)
        )
    );
    print(sR_layout_blocked); print("\n");

    Tensor sR = make_tensor(M, sR_layout_blocked);
    {
        auto r_layout_rest = prepend(take<2, 4>(sR.layout()), Layout<_1, _0>{}); // prepend a dummy layout for replace it with n_groups

        auto layout_group = layout<1>(sR.layout());
        auto n_warps = size<1>(warp_layout);
        auto r_warp_group_layout = r_layout_mode1(n_warps, layout_group);
        print(r_warp_group_layout); print("\n");
        // auto r_warp_layout = make_layout(layout<0>(sR.layout()), r_warp_group_layout, take<2, 4>(sR.layout()));
        auto r_warp_layout = replace<1>(sR_layout_blocked, r_warp_group_layout); // ((RECONN_SZ, RECONN_SZ), (N_WARPS, GROUPS_PER_WAPR), BLOCKS, PIPELINE)
        // TODO: should tile the reconnection blocks to match the oparand size of mma atom, but for now, we will just use the reconnection size since it is the same as the mma atom size
        auto mma_thr_layout = composition(
            r_warp_layout,
            make_tile(mma_atom1::LayoutB_TV{}, _, _, _)
        );
        print(mma_thr_layout); print("\n");
        Tensor sR_blocked = make_tensor(M, mma_thr_layout);
        Tensor thr_sR = sR_blocked(
            make_coord(lane_idx, _),
            make_coord(get<1>(warp_idx_3d), _),
            _, _
        );
        print(thr_sR.layout()); print("\n");
        Tensor thr_rR = make_tensor<half_t>(make_shape(shape<0>(thr_sR), shape<1>(thr_sR), _2{}));
        print(thr_rR.layout()); print("\n");

        // auto sR_warp_tile = make_tile(make_layout(reconn_sz), 
        //                               make_layout(size<2>(warp_atom_mnk)));
        // print(sR_warp_tile); print("\n");
            
        // auto r_tensor_layout = zipped_divide(take<0,2>(sB.layout()), sR_warp_tile); // ((WARP_ATOM_M, WARP_ATOM_K), (RESTM, RESTK))  ignore PIPE for now, describes how the matrix would be handled by the warps
        // print(b_tensor_layout); print("\n");

        // auto atom_tile = make_tile(make_layout(_8{}), make_layout(_8{})); // the minimum sized block that a warp should handle using mma atom during the reconnection stage

        // // print(sA.layout()); print("\n");
        // // print(a_tensor_layout); print("\n");
        // // findout how each warp would have to handle the matrix in their region of responsibility
        // auto mma_atom_layout = zipped_divide(get<0>(b_tensor_layout), atom_tile); // ((ATOM_SZ_M, ATOM_SZ_K), (ATOM_M, ATOM_K))
        // print(mma_atom_layout); print("\n");

        // auto mma_thr_layout = composition(mma_atom_layout, make_tile(mma_atom1::LayoutB_TV{}, _)); // ((THR, THR_DATA), (ATOM_M, ATOM_K))
        // print(mma_thr_layout); print("\n");

        // auto overall_layout = make_layout(mma_thr_layout, get<1>(b_tensor_layout), layout<2>(sB.layout()));
        // print(overall_layout); print("\n");

        // Tensor sB_blocked = make_tensor(M, overall_layout);
        // Tensor thr_sB = sB_blocked(
        //     make_coord(make_coord(lane_idx, _), _),
        //     make_coord(get<1>(warp_idx_3d), get<2>(warp_idx_3d)),
        //     _
        // );
        // print(thr_sB.layout()); print("\n");
        // Tensor thr_rB = make_tensor<half_t>(make_shape(shape<0>(thr_sB), shape<1,0>(thr_sB), _2{}));
        // print(thr_rB.layout()); print("\n");
    }
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

    half* M = new half[256*256*3];
    auto group_sz = _128{};
    auto reconn_sz = _8{};
    auto blocks_tiler = make_shape(
        _256{}, _2{}, _2{}
    );
    auto pipeline = _3{};
    auto cta_tiler = make_shape(
        size<0>(blocks_tiler), // BM
        size<1>(blocks_tiler) * group_sz, // BN
        size<2>(blocks_tiler) * reconn_sz // BK
    );

    auto smem_atom = composition(
          Swizzle<1,5>{},
          Layout<
            Shape <_8, Shape <_8,  _2>>,
            Stride<_8, Stride<_1, _64>>
          >{}
        );
    
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
            size<1>(blocks_tiler) * reconn_sz,
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

    Tensor sA = make_tensor(M, sA_layout);
    Tensor sB = make_tensor(M, sB_layout);
    Tensor r2d = make_tensor(M, sR_layout_2d);
    Tensor r4d = make_tensor(M, sR_layout_4d);

    int warp_idx = 3;

    auto warp_layout = make_layout(
        make_shape(_4{}, _2{})
    );

    auto warp_coord = warp_layout.get_hier_coord(warp_idx);

    auto warp_atom_mn = make_shape(
        // size<0>(cta_tiler) / size<0>(warp_layout), // WARP_ATOM_M
        _16{},
        size<1>(cta_tiler) / size<1>(warp_layout) // WARP_ATOM_N
    );

    print(sB); print("\n");

    Tensor b_atom_tiles = logical_divide(
        sB,
        make_tile(
            make_layout(
                make_shape(
                    size<1>(warp_atom_mn), // WARP_N
                    size<1>(warp_layout)
                )
            ),
            make_layout(
                reconn_sz
            )
        )
    );
    print(b_atom_tiles); print("\n");

    Tensor _b_warp_tensor = b_atom_tiles(make_coord(make_coord(_, get<1>(warp_coord)), _), make_coord(_, _), _); // (WARP_ATOM_N, REST_N, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)
    print(_b_warp_tensor); print("\n");
    Tensor b_warp_tensor = make_tensor(
        _b_warp_tensor.data(),
        coalesce(group<0,2>(_b_warp_tensor.layout()), Step<_1,_1,_1,_1>{}) // (WARP_N_REGION, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)
    );
    print(b_warp_tensor); print("\n");
    // create the group dimension for each warp
    auto warp_group_sz = min(size<0>(b_warp_tensor), group_sz);
    Tensor b_warp_grouped = logical_divide(
        b_warp_tensor,
        make_tile(make_layout(warp_group_sz))
    ); // ((WARP_GROUP_SIZE, WAPR_N_GROUPS), RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)
    print(b_warp_grouped); print("\n");

    // auto m_warp_tiles = tiled_m(make_coord(_, get<0>(warp_coord)), _, _, _, _);
    // print(m_warp_tiles); print("\n");
    // Tensor m_warp = make_tensor(
    //     m_warp_tiles.data(),
    //     select<0,2,1,3>(m_warp_tiles.layout())
    //     // make_layout(
    //     //     select<0,2>(m_warp_tiles.layout()),
    //     //     select<1,3,4>(m_warp_tiles.layout())
    //     // )
    // );
    // print(m_warp); print("\n");
    // // print(m_warp_tiles); print("\n");

    // print(flat_divide(m, atom_tiles)); print("\n");
    // print(tiled_divide(m, atom_tiles)); print("\n");

    // Tensor gm = local_tile(m, make_shape(_8{}, _16{}), make_coord(10, _));
    // print(gm); print("\n");
    // print(gm.data() - M); print("\n");
    return 0;
    // TiledCopy copyR = cp_layout<uint128_t, half_t, true, true>(
    //     _8{}, _16{}, _8{} * _32{}
    // );
    // print(copyR); print("\n");
    // print(size(copyR)); print("\n");
    // print(copyR.get_slice(0).partition_S(m(_, _, 0))); print("\n");
    // return 0;
    // // ezprep();
    // // return 0;
    

    // auto blocks_tiler = make_shape(
    //     _128{}, _1{}, _8{} // (BATCH_SZ, N_GROUPS, N_BLOCKS)
    // );

    // auto reconn_sz = _8{};
    // auto group_sz = _128{};
    // auto pipeline = _3{};

    // auto cta_tiler = make_shape(
    //     size<0>(blocks_tiler), // BM
    //     size<1>(blocks_tiler) * group_sz, // BN
    //     size<2>(blocks_tiler) * reconn_sz // BK
    // );

    // auto smem_atom = make_layout( // two rows of padded shared memory layout atom, should be replaced by swizzle
    //     make_shape(_2{}, size<2>(cta_tiler)),
    //     make_stride(size<2>(cta_tiler) + _8{}, _1{})
    // );

    // auto warp_layout = make_layout(
    //     make_shape(_4{}, _2{}, _1{})
    // );

    // auto warp_atom_mnk = make_shape(
    //     size<0>(cta_tiler) / size<0>(warp_layout), // WARP_ATOM_M
    //     size<1>(cta_tiler) / size<1>(warp_layout), // WARP_ATOM_N
    //     size<2>(cta_tiler) / size<2>(warp_layout)  // WARP_ATOM_K
    // ); // A rather simple way to tile the warps

    // // auto warp_atom_mnk = make_shape(_16{}, group_sz, _16{}); // the minimum sized block that each warp should work on at once

    // // TiledMMA warp_mma = make_tiled_mma(SM80_16x8x8_F16F16F16F16_TN{});  // 16x8x8 TiledMMA
    // using mma_atom1 = MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>;
    // using mma_atom2 = mma_atom1;

    // int thread_idx = 128;
    // int warp_idx = thread_idx / 32;
    // int lane_idx = thread_idx % 32;
    // auto warp_idx_3d = warp_layout.get_hier_coord(warp_idx);
    // print(warp_idx_3d); print("\n");

    // auto sA_layout = tile_to_shape(
    //     smem_atom,
    //     make_shape(
    //         size<0>(cta_tiler),
    //         size<2>(cta_tiler),
    //         pipeline
    //         )
    //     );

    // Tensor sA = make_tensor(M, sA_layout);
    // {
    //     auto sA_warp_tile = make_tile(make_layout(size<0>(warp_atom_mnk)), 
    //                                   make_layout(size<2>(warp_atom_mnk)));
            
    //     auto a_tensor_layout = zipped_divide(take<0,2>(sA.layout()), sA_warp_tile); // ((WARP_ATOM_M, WARP_ATOM_K), (RESTM, RESTK))  ignore PIPE for now

    //     auto atom_tile = make_tile(make_layout(size<0>(mma_atom1::Shape_MNK{})),
    //                                make_layout(
    //                                     make_shape(size<2>(mma_atom1::Shape_MNK{}),
    //                                     reconn_sz / size<2>(mma_atom1::Shape_MNK{}))
    //                                     )
    //                                 ); // the minimum sized block that a warp should handle using mma atom during the reconnection stage
        
    //     auto _mma_atom_layout = zipped_divide(get<0>(a_tensor_layout), atom_tile); // ((ATOM_SZ_M, (ATOM_SZ_K, ATOM_RECONNECT_K)), (ATOM_M, ATOM_K))

    //     auto mma_atom_layout_rank1 = group<0,2>(flatten(get<0>(_mma_atom_layout)));

    //     auto mma_atom_layout = replace<0>(_mma_atom_layout, mma_atom_layout_rank1); // (((ATOM_SZ_M, ATOM_SZ_K), ATOM_RECONNECT_K), (ATOM_M, ATOM_K))

    //     auto mma_thr_layout = composition(mma_atom_layout, make_tile(make_tile(mma_atom1::LayoutA_TV{}, _), _)); // (((THR, THR_DATA), ATOM_RECONNECT_K), (ATOM_M, ATOM_K))

    //     // print(mma_thr_layout); print("\n");
    //     auto warp_blocks_layout = zipped_divide(get<1>(a_tensor_layout),
    //                                             make_tile(
    //                                                 make_layout(size<0>(warp_layout)),
    //                                                 make_layout(size<2>(warp_layout))
    //                                             )); // ((WARP_GROUP_M, WARP_GROUP_K), (TILE_M, TILE_K))
        
    //     // mma_thr_layout is the TV layout within a single computation block
    //     // warp_blocks_layout is the layout of the computation blocks
    //     auto overall_layout = make_layout(mma_thr_layout, warp_blocks_layout, layout<2>(sA.layout()));
    //     Tensor sA_blocked = make_tensor(M, overall_layout);

    //     Tensor thr_sA = sA_blocked(
    //         make_coord(make_coord(make_coord(lane_idx, _), _), _),
    //         make_coord(make_coord(get<0>(warp_idx_3d), get<2>(warp_idx_3d)), _),
    //         _
    //     ); // (THR_DATA, ATOM_RECONNECT_K, (ATOM_M, ATOM_K), (TILE_M, TILE_K), PIPE)


    //     Tensor thr_rA = make_tensor<half_t>(shape(thr_sA(_, _, _, _, 0)));
    //     // print(thr_rA); print("\n");
                                                       
    //     // print(sA_blocked); print("\n");
    //     // print(thr_sA); print("\n");
    // }

    // // {
    // //     auto sA_warp_tile = make_tile(make_layout(size<0>(warp_atom_mnk)), 
    // //                                   make_layout(size<2>(warp_atom_mnk)));
            
    // //     auto a_tensor_layout = zipped_divide(take<0,2>(sA.layout()), sA_warp_tile); // ((WARP_ATOM_M, WARP_ATOM_K), (RESTM, RESTK))  ignore PIPE for now

    // //     auto atom_tile = make_tile(make_layout(size<0>(mma_atom1::Shape_MNK{})),
    // //                                make_layout(
    // //                                     make_shape(size<2>(mma_atom1::Shape_MNK{}),
    // //                                     reconn_sz / size<2>(mma_atom1::Shape_MNK{}))
    // //                                     )
    // //                                 ); // the minimum sized block that a warp should handle using mma atom during the reconnection stage
        
    // //     auto _mma_atom_layout = zipped_divide(get<0>(a_tensor_layout), atom_tile); // ((ATOM_SZ_M, (ATOM_SZ_K, ATOM_RECONNECT_K)), (ATOM_M, ATOM_K))

    // //     auto mma_atom_layout_rank1 = group<0,2>(flatten(get<0>(_mma_atom_layout)));

    // //     auto mma_atom_layout = replace<0>(_mma_atom_layout, mma_atom_layout_rank1); // (((ATOM_SZ_M, ATOM_SZ_K), ATOM_RECONNECT_K), (ATOM_M, ATOM_K))

    // //     auto mma_thr_layout = composition(mma_atom_layout, make_tile(make_tile(mma_atom1::LayoutA_TV{}, _), _)); // (((THR, THR_DATA), ATOM_RECONNECT_K), (ATOM_M, ATOM_K))

    // //     // print(mma_thr_layout); print("\n");
    // //     auto warp_blocks_layout = zipped_divide(get<1>(a_tensor_layout),
    // //                                             make_tile(
    // //                                                 make_layout(size<0>(warp_layout)),
    // //                                                 make_layout(size<2>(warp_layout))
    // //                                             )); // ((WARP_GROUP_M, WARP_GROUP_K), (TILE_M, TILE_K))
        
    // //     // mma_thr_layout is the TV layout within a single computation block
    // //     // warp_blocks_layout is the layout of the computation blocks
    // //     auto overall_layout = make_layout(mma_thr_layout, warp_blocks_layout, layout<2>(sA.layout()));
    // //     Tensor sA_blocked = make_tensor(M, overall_layout);

    // //     Tensor thr_sA = sA_blocked(
    // //         make_coord(make_coord(make_coord(lane_idx, _), _), _),
    // //         make_coord(make_coord(get<0>(warp_idx_3d), get<2>(warp_idx_3d)), _),
    // //         _
    // //     ); // (THR_DATA, ATOM_RECONNECT_K, (ATOM_M, ATOM_K), (TILE_M, TILE_K), PIPE)


    // //     Tensor thr_rA = make_tensor<half_t>(shape(thr_sA(_, _, _, _, 0)));
    // //     // print(thr_rA); print("\n");
                                                       
    // //     // print(sA_blocked); print("\n");
    // //     // print(thr_sA); print("\n");
    // // }

    // auto sW_layout = tile_to_shape(
    //     smem_atom,
    //     make_shape(
    //         size<1>(cta_tiler),
    //         size<2>(cta_tiler),
    //         pipeline
    //         )
    //     );

    // Tensor sW = make_tensor(M, sW_layout);
    // {
    //     auto sW_warp_tile = make_tile(make_layout(size<1>(warp_atom_mnk)), 
    //                                   make_layout(size<2>(warp_atom_mnk)));
    
    //     auto a_tensor_layout = zipped_divide(take<0,2>(sW.layout()), sW_warp_tile); // ((WARP_ATOM_N, WARP_ATOM_K), (RESTN, RESTK))  ignore PIPE for now

    //     auto atom_tile = make_tile(make_layout(size<0>(mma_atom2::Shape_MNK{})),
    //                                make_layout(size<2>(mma_atom2::Shape_MNK{})));
        
    //     auto mma_atom_layout = zipped_divide(get<0>(a_tensor_layout), atom_tile); // (ATOM_SZ, (ATOM_N, ATOM_K))

    //     auto mma_thr_layout = composition(mma_atom_layout, make_tile(mma_atom2::LayoutA_TV{}, _)); // ((THR, THR_DATA), (ATOM_N, ATOM_K))

    //     auto warp_blocks_layout = zipped_divide(get<1>(a_tensor_layout),
    //                                             make_tile(
    //                                                 make_layout(size<1>(warp_layout)),
    //                                                 make_layout(size<2>(warp_layout))
    //                                             )); // ((WARP_GROUP_N, WARP_GROUP_K), (TILE_N, TILE_K))
        
    //     // mma_thr_layout is the TV layout within a single computation block
    //     // warp_blocks_layout is the layout of the computation blocks
    //     auto overall_layout = make_layout(mma_thr_layout, warp_blocks_layout, layout<2>(sW.layout()));
        
    //     Tensor sW_blocked = make_tensor(M, overall_layout);

    //     Tensor thr_sW = sW_blocked(
    //         make_coord(make_coord(lane_idx, _), _),
    //         make_coord(make_coord(get<1>(warp_idx_3d), get<2>(warp_idx_3d)), _),
    //         _
    //     ); // (THR_DATA, (ATOM_N, ATOM_N), (TILE_N, TILE_N), PIPE)

    //     // print(sW_layout); print("\n");
    //     // print(sW_blocked); print("\n");
    //     // print(thr_sW); print("\n");

    //     // print(mma_atom::LayoutC_TV{}); print("\n");
    // }

    // auto sR_layout = tile_to_shape(
    //     smem_atom,
    //     make_shape(
    //         size<1>(blocks_tiler) * reconn_sz,
    //         size<2>(cta_tiler),
    //         pipeline
    //         )
    //     );
    
    // Tensor sR = make_tensor(M, sR_layout);
    // {
    //     auto sR_warp_tile = make_tile(make_layout(reconn_sz), 
    //                                   make_layout(size<2>(warp_atom_mnk)));
    
    //     auto a_tensor_layout = zipped_divide(take<0,2>(sR.layout()), sR_warp_tile); // ((WARP_ATOM_N, WARP_ATOM_K), (RESTN, RESTK))  ignore PIPE for now

    //     auto atom_tile = make_tile(make_layout(
    //                                     make_shape(
    //                                         size<1>(mma_atom1::Shape_MNK{}),  // ATOM_SZ_N
    //                                         reconn_sz / size<1>(mma_atom1::Shape_MNK{})  // ATOM_RECONN_N
    //                                         )
    //                                     ),
    //                                make_layout(
    //                                     make_shape(
    //                                         size<2>(mma_atom1::Shape_MNK{}),  // ATOM_SZ_K
    //                                         reconn_sz / size<2>(mma_atom1::Shape_MNK{})  // ATOM_RECONN_K
    //                                         )
    //                                     )
    //                                 ); // For the tiling that has to be done within one single reconnection matrix
        
    //     auto _mma_atom_layout = zipped_divide(get<0>(a_tensor_layout), atom_tile); // (((ATOM_SZ_N, ATOM_RECONN_N), (ATOM_SZ_K, ATOM_RECONN_K)), (ATOM_N, ATOM_K))
    //     // print(_mma_atom_layout); print("\n");
    //     auto mma_atom_layout_rank1_flattened = flatten(get<0>(_mma_atom_layout));
    //     auto mma_atom_layout_rank1 = make_layout(select<0,2>(mma_atom_layout_rank1_flattened), select<1,3>(mma_atom_layout_rank1_flattened));
    //     // print(mma_atom_layout_rank1); print("\n");
    //     auto mma_atom_layout = replace<0>(_mma_atom_layout, mma_atom_layout_rank1); // (((ATOM_SZ_N, ATOM_SZ_K), (ATOM_RECONN_N, ATOM_RECONN_K)), (ATOM_N, ATOM_K))
    //     auto mma_thr_layout = composition(mma_atom_layout, make_tile(make_tile(mma_atom1::LayoutB_TV{}, _), _)); // (((THR, THR_DATA), (ATOM_RECONN_N, ATOM_RECONN_K)), (ATOM_N, ATOM_K))
    //     // print(mma_thr_layout); print("\n");

    //     // auto mma_thr_layout = composition(mma_atom_layout, make_tile(mma_atom1::LayoutB_TV{}, _)); // ((THR, THR_DATA), (ATOM_N, ATOM_K))

    //     auto warp_blocks_layout = zipped_divide(get<1>(a_tensor_layout),
    //                                             make_tile(
    //                                                 make_layout(size<1>(warp_layout)),
    //                                                 make_layout(size<2>(warp_layout))
    //                                             )); // ((WARP_GROUP_N, WARP_GROUP_K), (TILE_N, TILE_K))
    //     auto overall_layout = make_layout(mma_thr_layout, warp_blocks_layout, layout<2>(sR.layout()));
    //     // print(overall_layout); print("\n");

    //     Tensor sR_blocked = make_tensor(M, overall_layout);
    //     Tensor thr_sR = sR_blocked(
    //         make_coord(make_coord(make_coord(lane_idx, _), _), _),
    //         make_coord(make_coord(get<1>(warp_idx_3d), get<2>(warp_idx_3d)), _),
    //         _
    //     ); // (THR_DATA, (ATOM_RECONN_N, ATOM_RECONN_K), (ATOM_N, ATOM_K), (TILE_N, TILE_K), PIPE)
    //     // print(thr_sR); print("\n");
        
    //     // // mma_thr_layout is the TV layout within a single computation block
    //     // // warp_blocks_layout is the layout of the computation blocks
    //     // auto overall_layout = make_layout(mma_thr_layout, warp_blocks_layout, layout<2>(sW.layout()));
        
    //     // Tensor sW_blocked = make_tensor(M, overall_layout);

    //     // Tensor thr_sW = sW_blocked(
    //     //     make_coord(make_coord(lane_idx, _), _),
    //     //     make_coord(make_coord(get<1>(warp_idx_3d), get<2>(warp_idx_3d)), _),
    //     //     _
    //     // ); // (THR_DATA, (ATOM_N, ATOM_N), (TILE_N, TILE_N), PIPE)

    //     // print(sW_layout); print("\n");
    //     // print(sW_blocked); print("\n");
    //     // print(thr_sW); print("\n");

    //     // print(mma_atom::LayoutC_TV{}); print("\n");
    }

    // auto sB_warp_tile = make_tile(make_layout(size<1>(warp_atom_mnk)), 
    //                               make_layout(size<2>(warp_atom_mnk)));
    // auto sR_warp_tile = make_tile(make_layout(reconn_sz),
    //                               make_layout(size<2>(warp_atom_mnk)));
    
    // auto sR_layout = tile_to_shape(
    //     smem_atom,
    //     make_shape(
    //         size<1>(blocks_tiler) * reconn_sz,
    //         size<2>(cta_tiler),
    //         pipeline
    //         )
    //     );
    
    
    


    // Tensor sR = make_tensor(M, sR_layout);
    // Tensor sB = make_tensor(M, sB_layout);
    // Tensor sC = make_tensor(M, make_shape(size<0>(cta_tiler), size<1>(cta_tiler)));
    
    

    // auto sA_warp_atom = make_shape(_16{}, make_shape(_2{}, _8{}));
    // auto sB_warp_atom = make_shape(make_shape(group_sz / _8{}, _8{}), _16{});
    // auto sR_warp_atom = make_shape(make_shape(_2{}, _8{}), make_shape(_2{}, _8{}));

    
    // auto sA_block_atom = make_tile(
    //     make_layout(make_shape(size<0>(warp_layout), get<0>(sA_warp_atom)), LayoutRight{}),
    //     make_layout(get<1>(sA_warp_atom), LayoutRight{})
    //     );
    // auto sB_block_atom = make_tile(
    //     make_layout(make_shape(size<1>(warp_layout), get<0>(sB_warp_atom)), LayoutRight{}),
    //     make_layout(get<1>(sB_warp_atom), LayoutRight{})
    //     );
    // auto sR_block_atom = make_tile(
    //     make_layout(make_shape(size<1>(warp_layout), get<0>(sR_warp_atom)), LayoutRight{}),
    //     make_layout(get<1>(sR_warp_atom), LayoutRight{})
    //     );

    // auto r_tiler = make_shape(
    //     size<1>(blocks_tiler),
    //     size<2>(blocks_tiler),
    //     reconn_sz,
    //     reconn_sz
    // );

    // auto sR_layout = tile_to_shape(
    //     smem_atom,
    //     make_shape(
    //         size<1>(blocks_tiler) * reconn_sz,
    //         size<2>(cta_tiler),
    //         pipeline
    //         )
    //     );
    
    // auto sA_layout = tile_to_shape(
    //     smem_atom,
    //     make_shape(
    //         size<0>(cta_tiler),
    //         size<2>(cta_tiler),
    //         pipeline
    //         )
    //     );
    
    // auto sB_layout = tile_to_shape(
    //     smem_atom,
    //     make_shape(
    //         size<1>(cta_tiler),
    //         size<2>(cta_tiler),
    //         pipeline
    //         )
    //     );

    // Tensor sR = make_tensor(M, sR_layout);
    // Tensor sA = make_tensor(M, sA_layout);
    // Tensor sB = make_tensor(M, sB_layout);
    // Tensor sC = make_tensor(M, make_shape(size<0>(cta_tiler), size<1>(cta_tiler)));

    // auto _sR_warp_blocks = zipped_divide(sR_layout, sR_block_atom);
    // auto sR_warp_blocks = make_layout(
    //     get<0,0,0>(_sR_warp_blocks), // (WARP_GROUP_N)
    //     get<1>(_sR_warp_blocks), // (TILE_GROUP, TILE_BLOCK, PIPE)
    //     make_layout(
    //         get<0,0,1,0>(_sR_warp_blocks),
    //         get<0,1,0>(_sR_warp_blocks)
    //     ),
    //     make_layout(
    //         get<0,0,1,1>(_sR_warp_blocks),
    //         get<0,1,1>(_sR_warp_blocks)
    //     )
    // );
    // // Tensor sR_blocked = make_tensor(M, sR_warp_blocks);
    // // print(sR_blocked(1, _, _, _)); print("\n");
    // // print(sR); print("\n");

    // auto _sA_warp_blocks = zipped_divide(sA_layout, sA_block_atom);
    // auto sA_warp_blocks = make_layout(
    //     get<0,0,0>(_sA_warp_blocks), // (WARP_GROUP_M)
    //     get<1>(_sA_warp_blocks), // (TILE_M, TILE_K, PIPE)
    //     make_layout(
    //         // make_layout(_1{}),
    //         get<0,1,0>(_sA_warp_blocks)
    //     ),
    //     make_layout(
    //         get<0,0,1>(_sA_warp_blocks),
    //         get<0,1,1>(_sA_warp_blocks)
    //     )
    // );
    // // print(sA_warp_blocks); print("\n");

    // auto _sB_warp_blocks = zipped_divide(sB_layout, sB_block_atom);
    // auto sB_warp_blocks = make_layout(
    //     get<0,0,0>(_sB_warp_blocks), // (WARP_GROUP_N)
    //     get<1>(_sB_warp_blocks), // (TILE_N, TILE_K, PIPE)
    //     make_layout(
    //         // make_layout(_1{}),
    //         get<0,0,1,0>(_sB_warp_blocks)
    //     ),
    //     make_layout(
    //         get<0,0,1,1>(_sB_warp_blocks),
    //         get<0,1>(_sB_warp_blocks)
    //     )
    // );
    // print(sB_warp_blocks); print("\n");
    // // print(_sA_warp_blocks); print("\n");


    // // auto sR_layout_blocked = select<1,0>(zipped_divide(sR_layout, make_shape(reconn_sz, reconn_sz)));
    // // static_assert(is_static<decltype(sR_layout_blocked)>::value);
    // // print(sR_layout); print("\n");
    // // print(product_each(shape(sR_layout_blocked))); print("\n");
    
    // TiledMMA mmaC = make_tiled_mma(
    //                 SM80_16x8x8_F16F16F16F16_TN{},
    //                 make_layout(make_shape(_4{}, _2{})),
    //                 Tile<decltype(size<0>(sA)), decltype(size<0>(sB))>{});  // 16x8x8 TiledMMA
    
    // print(mmaC.get_thr_layout_vmnk()); print("\n");
    // auto thr_C = mmaC.thrfrg_C(sC);
    // print(thr_C); print("\n");
    // ThrMMA thr_mma = mmaC.get_slice(0);
    // Tensor tCrA = thr_mma.partition_fragment_A(sR(_, _, 0));
    // Tensor tCrB = thr_mma.partition_fragment_B(sR(_, _, 0));


    // Tensor tCsC = thr_mma.partition_C()
    // print(tCrA); print("\n");
    // print(tCrB); print("\n");
    // print(sR); print("\n");
    // print(mmaC); print("\n");


    
    // print(sA); print("\n");

    // // Tensor mR = make_tensor(R, ar_layout);
    // auto reordered_ar_layout = make_layout(select<0,2>(ar_layout), select<1,3>(ar_layout));
    // Tensor mR = make_tensor(R, reordered_ar_layout);
    // print(reordered_ar_layout); print("\n");
    // auto r_tiler = make_shape(make_tuple(get<0>(block_sizes), size<2>(ar_layout)), make_tuple(get<1>(block_sizes), size<3>(ar_layout)));
    // Tensor gR = local_tile(mR, r_tiler, make_coord(make_tuple(0, _0{}), make_tuple(_, _0{})));

    // print(gR.layout()); print("\n");
// }
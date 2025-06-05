#include <cute/tensor.hpp>
#include <iostream>
#include <cute/atom/mma_atom.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <bool inplace, class TA, class TB, class TC, class MMA_ATOM>
__global__ static void inplace_mma_test(
    TA const *A, TB const *B, TC *C, MMA_ATOM atom
) {
    using namespace cute;
    using mma_atom = MMA_Atom<MMA_ATOM>;
    Tensor mA = make_tensor(A, make_layout(
        make_shape(size<0>(typename mma_atom::Shape_MNK{}),
                   size<2>(typename mma_atom::Shape_MNK{})),
        LayoutRight{}
    ));
    Tensor mB = make_tensor(B, make_layout(
        make_shape(size<1>(typename mma_atom::Shape_MNK{}),
                   size<2>(typename mma_atom::Shape_MNK{})),
        LayoutRight{}
    ));
    Tensor mC = make_tensor(C, make_layout(
        make_shape(size<0>(typename mma_atom::Shape_MNK{}),
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
    thrust::host_vector<half_t> hA(16 * 8);
    thrust::host_vector<half_t> hB(8 * 8);
    thrust::host_vector<half_t> hC(16 * 8);

    for (int i = 0; i < 16 * 8; i++) {
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

    for (int i = 0; i < 16 * 8; i++) {
        hC[i] = hA[i];
    }

    thrust::device_vector<half_t> dA = hA;
    thrust::device_vector<half_t> dB = hB;
    thrust::device_vector<half_t> dC = hC;

    dim3 dimBlock(32);
    dim3 dimGrid(1);

    inplace_mma_test<false><<<dimGrid, dimBlock>>>(dA.data().get(), dB.data().get(), dC.data().get(), SM80_16x8x8_F16F16F16F16_TN{});

    thrust::host_vector<half_t> hC2 = dC;
    for (int i = 0; i < 16 * 8; i++) {
        std::cout << hC2[i] << " ";
    }

    std::cout << std::endl;

    inplace_mma_test<true><<<dimGrid, dimBlock>>>(dA.data().get(), dB.data().get(), dC.data().get(), SM80_16x8x8_F16F16F16F16_TN{});

    hC2 = dC;
    for (int i = 0; i < 16 * 8; i++) {
        std::cout << hC2[i] << " ";
    }
    std::cout << std::endl;
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
    // test_inplace();
    // auto t = make_tuple(_256{}, R<1, 2>{});
    // R<1,2> r;

    // print(round_up(r * _3{})); print("\n");


    // return 0;
    ezprep();
    return 0;
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

    auto warp_layout = make_layout(
        make_shape(_4{}, _2{}, _1{})
    );

    auto warp_atom_mnk = make_shape(
        size<0>(cta_tiler) / size<0>(warp_layout), // WARP_ATOM_M
        size<1>(cta_tiler) / size<1>(warp_layout), // WARP_ATOM_N
        size<2>(cta_tiler) / size<2>(warp_layout)  // WARP_ATOM_K
    ); // A rather simple way to tile the warps

    // auto warp_atom_mnk = make_shape(_16{}, group_sz, _16{}); // the minimum sized block that each warp should work on at once

    // TiledMMA warp_mma = make_tiled_mma(SM80_16x8x8_F16F16F16F16_TN{});  // 16x8x8 TiledMMA
    using mma_atom1 = MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>;
    using mma_atom2 = mma_atom1;

    int thread_idx = 128;
    int warp_idx = thread_idx / 32;
    int lane_idx = thread_idx % 32;
    auto warp_idx_3d = warp_layout.get_hier_coord(warp_idx);
    print(warp_idx_3d); print("\n");

    auto sA_layout = tile_to_shape(
        smem_atom,
        make_shape(
            size<0>(cta_tiler),
            size<2>(cta_tiler),
            pipeline
            )
        );

    Tensor sA = make_tensor(M, sA_layout);
    {
        auto sA_warp_tile = make_tile(make_layout(size<0>(warp_atom_mnk)), 
                                      make_layout(size<2>(warp_atom_mnk)));
            
        auto a_tensor_layout = zipped_divide(take<0,2>(sA.layout()), sA_warp_tile); // ((WARP_ATOM_M, WARP_ATOM_K), (RESTM, RESTK))  ignore PIPE for now

        auto atom_tile = make_tile(make_layout(size<0>(mma_atom1::Shape_MNK{})),
                                   make_layout(
                                        make_shape(size<2>(mma_atom1::Shape_MNK{}),
                                        reconn_sz / size<2>(mma_atom1::Shape_MNK{}))
                                        )
                                    ); // the minimum sized block that a warp should handle using mma atom during the reconnection stage
        
        auto _mma_atom_layout = zipped_divide(get<0>(a_tensor_layout), atom_tile); // ((ATOM_SZ_M, (ATOM_SZ_K, ATOM_RECONNECT_K)), (ATOM_M, ATOM_K))

        auto mma_atom_layout_rank1 = group<0,2>(flatten(get<0>(_mma_atom_layout)));

        auto mma_atom_layout = replace<0>(_mma_atom_layout, mma_atom_layout_rank1); // (((ATOM_SZ_M, ATOM_SZ_K), ATOM_RECONNECT_K), (ATOM_M, ATOM_K))

        auto mma_thr_layout = composition(mma_atom_layout, make_tile(make_tile(mma_atom1::LayoutA_TV{}, _), _)); // (((THR, THR_DATA), ATOM_RECONNECT_K), (ATOM_M, ATOM_K))

        // print(mma_thr_layout); print("\n");
        auto warp_blocks_layout = zipped_divide(get<1>(a_tensor_layout),
                                                make_tile(
                                                    make_layout(size<0>(warp_layout)),
                                                    make_layout(size<2>(warp_layout))
                                                )); // ((WARP_GROUP_M, WARP_GROUP_K), (TILE_M, TILE_K))
        
        // mma_thr_layout is the TV layout within a single computation block
        // warp_blocks_layout is the layout of the computation blocks
        auto overall_layout = make_layout(mma_thr_layout, warp_blocks_layout, layout<2>(sA.layout()));
        Tensor sA_blocked = make_tensor(M, overall_layout);

        Tensor thr_sA = sA_blocked(
            make_coord(make_coord(make_coord(lane_idx, _), _), _),
            make_coord(make_coord(get<0>(warp_idx_3d), get<2>(warp_idx_3d)), _),
            _
        ); // (THR_DATA, ATOM_RECONNECT_K, (ATOM_M, ATOM_K), (TILE_M, TILE_K), PIPE)


        Tensor thr_rA = make_tensor<half_t>(shape(thr_sA(_, _, _, _, 0)));
        // print(thr_rA); print("\n");
                                                       
        // print(sA_blocked); print("\n");
        // print(thr_sA); print("\n");
    }

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

    auto sW_layout = tile_to_shape(
        smem_atom,
        make_shape(
            size<1>(cta_tiler),
            size<2>(cta_tiler),
            pipeline
            )
        );

    Tensor sW = make_tensor(M, sW_layout);
    {
        auto sW_warp_tile = make_tile(make_layout(size<1>(warp_atom_mnk)), 
                                      make_layout(size<2>(warp_atom_mnk)));
    
        auto a_tensor_layout = zipped_divide(take<0,2>(sW.layout()), sW_warp_tile); // ((WARP_ATOM_N, WARP_ATOM_K), (RESTN, RESTK))  ignore PIPE for now

        auto atom_tile = make_tile(make_layout(size<0>(mma_atom2::Shape_MNK{})),
                                   make_layout(size<2>(mma_atom2::Shape_MNK{})));
        
        auto mma_atom_layout = zipped_divide(get<0>(a_tensor_layout), atom_tile); // (ATOM_SZ, (ATOM_N, ATOM_K))

        auto mma_thr_layout = composition(mma_atom_layout, make_tile(mma_atom2::LayoutA_TV{}, _)); // ((THR, THR_DATA), (ATOM_N, ATOM_K))

        auto warp_blocks_layout = zipped_divide(get<1>(a_tensor_layout),
                                                make_tile(
                                                    make_layout(size<1>(warp_layout)),
                                                    make_layout(size<2>(warp_layout))
                                                )); // ((WARP_GROUP_N, WARP_GROUP_K), (TILE_N, TILE_K))
        
        // mma_thr_layout is the TV layout within a single computation block
        // warp_blocks_layout is the layout of the computation blocks
        auto overall_layout = make_layout(mma_thr_layout, warp_blocks_layout, layout<2>(sW.layout()));
        
        Tensor sW_blocked = make_tensor(M, overall_layout);

        Tensor thr_sW = sW_blocked(
            make_coord(make_coord(lane_idx, _), _),
            make_coord(make_coord(get<1>(warp_idx_3d), get<2>(warp_idx_3d)), _),
            _
        ); // (THR_DATA, (ATOM_N, ATOM_N), (TILE_N, TILE_N), PIPE)

        // print(sW_layout); print("\n");
        // print(sW_blocked); print("\n");
        // print(thr_sW); print("\n");

        // print(mma_atom::LayoutC_TV{}); print("\n");
    }

    auto sR_layout = tile_to_shape(
        smem_atom,
        make_shape(
            size<1>(blocks_tiler) * reconn_sz,
            size<2>(cta_tiler),
            pipeline
            )
        );
    
    Tensor sR = make_tensor(M, sR_layout);
    {
        auto sR_warp_tile = make_tile(make_layout(reconn_sz), 
                                      make_layout(size<2>(warp_atom_mnk)));
    
        auto a_tensor_layout = zipped_divide(take<0,2>(sR.layout()), sR_warp_tile); // ((WARP_ATOM_N, WARP_ATOM_K), (RESTN, RESTK))  ignore PIPE for now

        auto atom_tile = make_tile(make_layout(
                                        make_shape(
                                            size<1>(mma_atom1::Shape_MNK{}),  // ATOM_SZ_N
                                            reconn_sz / size<1>(mma_atom1::Shape_MNK{})  // ATOM_RECONN_N
                                            )
                                        ),
                                   make_layout(
                                        make_shape(
                                            size<2>(mma_atom1::Shape_MNK{}),  // ATOM_SZ_K
                                            reconn_sz / size<2>(mma_atom1::Shape_MNK{})  // ATOM_RECONN_K
                                            )
                                        )
                                    ); // For the tiling that has to be done within one single reconnection matrix
        
        auto _mma_atom_layout = zipped_divide(get<0>(a_tensor_layout), atom_tile); // (((ATOM_SZ_N, ATOM_RECONN_N), (ATOM_SZ_K, ATOM_RECONN_K)), (ATOM_N, ATOM_K))
        // print(_mma_atom_layout); print("\n");
        auto mma_atom_layout_rank1_flattened = flatten(get<0>(_mma_atom_layout));
        auto mma_atom_layout_rank1 = make_layout(select<0,2>(mma_atom_layout_rank1_flattened), select<1,3>(mma_atom_layout_rank1_flattened));
        // print(mma_atom_layout_rank1); print("\n");
        auto mma_atom_layout = replace<0>(_mma_atom_layout, mma_atom_layout_rank1); // (((ATOM_SZ_N, ATOM_SZ_K), (ATOM_RECONN_N, ATOM_RECONN_K)), (ATOM_N, ATOM_K))
        auto mma_thr_layout = composition(mma_atom_layout, make_tile(make_tile(mma_atom1::LayoutB_TV{}, _), _)); // (((THR, THR_DATA), (ATOM_RECONN_N, ATOM_RECONN_K)), (ATOM_N, ATOM_K))
        // print(mma_thr_layout); print("\n");

        // auto mma_thr_layout = composition(mma_atom_layout, make_tile(mma_atom1::LayoutB_TV{}, _)); // ((THR, THR_DATA), (ATOM_N, ATOM_K))

        auto warp_blocks_layout = zipped_divide(get<1>(a_tensor_layout),
                                                make_tile(
                                                    make_layout(size<1>(warp_layout)),
                                                    make_layout(size<2>(warp_layout))
                                                )); // ((WARP_GROUP_N, WARP_GROUP_K), (TILE_N, TILE_K))
        auto overall_layout = make_layout(mma_thr_layout, warp_blocks_layout, layout<2>(sR.layout()));
        // print(overall_layout); print("\n");

        Tensor sR_blocked = make_tensor(M, overall_layout);
        Tensor thr_sR = sR_blocked(
            make_coord(make_coord(make_coord(lane_idx, _), _), _),
            make_coord(make_coord(get<1>(warp_idx_3d), get<2>(warp_idx_3d)), _),
            _
        ); // (THR_DATA, (ATOM_RECONN_N, ATOM_RECONN_K), (ATOM_N, ATOM_K), (TILE_N, TILE_K), PIPE)
        // print(thr_sR); print("\n");
        
        // // mma_thr_layout is the TV layout within a single computation block
        // // warp_blocks_layout is the layout of the computation blocks
        // auto overall_layout = make_layout(mma_thr_layout, warp_blocks_layout, layout<2>(sW.layout()));
        
        // Tensor sW_blocked = make_tensor(M, overall_layout);

        // Tensor thr_sW = sW_blocked(
        //     make_coord(make_coord(lane_idx, _), _),
        //     make_coord(make_coord(get<1>(warp_idx_3d), get<2>(warp_idx_3d)), _),
        //     _
        // ); // (THR_DATA, (ATOM_N, ATOM_N), (TILE_N, TILE_N), PIPE)

        // print(sW_layout); print("\n");
        // print(sW_blocked); print("\n");
        // print(thr_sW); print("\n");

        // print(mma_atom::LayoutC_TV{}); print("\n");
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
}
#pragma once
#include <cute/tensor.hpp>

// template <class NWarp, class L>
// CUTE_HOST_DEVICE constexpr
// auto r_layout_mode1(NWarp n_warps, L layout_group) {
//     using namespace cute;
//     auto n_groups = size<0>(layout_group);
//     if constexpr (n_groups >= n_warps) {
//         // if the number of groups is greater or equal to the number of warps, then each warp will handle at least one group
//         CUTE_STATIC_ASSERT(n_groups % n_warps == 0, "Number of groups must be divisible by number of warps.");
//         return logical_divide(
//             layout_group,
//             make_shape(n_warps)
//         );
//     } else {
//         // if the number of groups is less than the number of warps, then a group would be handled by multiple warps
//         CUTE_STATIC_ASSERT(n_warps % n_groups == 0, "Number of warps must be divisible by number of groups.");
//         auto n_warps_per_group = n_warps / n_groups;
//         return make_layout(
//             make_layout(
//                 Layout<decltype(n_warps_per_group), _0>{},
//                 layout_group
//             ),
//             Layout<_1, _0>{}
//         );
//     }
// }

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

template <class ProblemShape, class BlocksTiler,
          class ALayout, class TiledCopyA,
          class RLayout, class TiledCopyR, class GroupSize, class ReconnectSize,
          class BLayout, class TiledCopyB,
          class CLayout, class WarpLayout, class Pipeline>
__global__ static __launch_bounds__(decltype(size(WarpLayout{}) * cute::_32{})::value)
void oft_device(ProblemShape shape_MNK, BlocksTiler blocks_tiler,
                half const *A, ALayout layout_a, TiledCopyA copy_a,
                half const *R, RLayout layout_r, TiledCopyR copy_r, GroupSize group_sz, ReconnectSize reconn_sz,
                half const *B, BLayout layout_b, TiledCopyB copy_b,
                half       *C, CLayout layout_c, WarpLayout warp_layout, Pipeline pipeline)
{
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});    // (M, N, K)
    CUTE_STATIC_ASSERT_V(size<1>(blocks_tiler) >= Int<1>{}); // Assume each thread block would handle at least one groups
    CUTE_STATIC_ASSERT_V(is_integral<decltype(size<1>(blocks_tiler))>{});
    static_assert(is_static<BlocksTiler>::value);
    CUTE_STATIC_ASSERT_V(rank(blocks_tiler) == Int<3>{}); // (BLK_M, BLK_N_GROUPS, BLK_K_BLOCKS)
    // CUTE_STATIC_ASSERT_V(reconn_sz == _8{}); // Assume the reconnection size is 8, which is the size of the atom

    auto cta_tiler = make_shape(
        size<0>(blocks_tiler),                    // BLK_M
        size<1>(blocks_tiler) * group_sz,       // BLK_N
        size<2>(blocks_tiler) * reconn_sz         // BLK_K
    );
    
    auto warp_atom_mn = make_shape(
        size<0>(cta_tiler) / size<0>(warp_layout), // BLK_M / WARP_M
        size<1>(cta_tiler) / size<1>(warp_layout)  // BLK_N / WARP_N  <- the size of this shall not be changed, otherwiseit will be wasteful in terms of register usage
    ); // A rather simple way to tile the warps, the shape of the tensor that each warp should handles

    using mma_atom1 = MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>;
    using mma_atom2 = mma_atom1;

    constexpr auto k_bit_width = log_2(static_cast<unsigned int>(decltype(size<2>(blocks_tiler))::value)); // assume that the reconnect size matches up with the k of the atom
    // auto smem_atom = composition(
    //       Swizzle<k_bit_width,6 - k_bit_width>{},
    //       Layout<
    //         Shape <_8, Shape <_8,  decltype(size<2>(blocks_tiler))>>,
    //         Stride<_8, Stride<_1, _64>>
    //       >{}
    //     );

    auto smem_atom = make_layout(
        make_shape(_2{}, size<2>(cta_tiler)),
        LayoutRight{}
    );
    CUTE_STATIC_ASSERT_V(size<1>(smem_atom) == size<2>(cta_tiler)); // Ensure the shared memory atom size matches the K dimension of the CTA tiler
    

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

    auto sR_layout = coalesce(tile_to_shape(
        smem_atom,
        make_shape(
            size<1>(blocks_tiler) * reconn_sz,
            size<2>(cta_tiler),
            pipeline
            )
        ), make_tuple(_1{}, _1{}, _1{})); // (N_GROUPS * R, K, PIPE)

    // Shared memory buffers
    __shared__ half smemA[cosize_v<decltype(sA_layout)>];
    __shared__ half smemR[cosize_v<decltype(sR_layout)>];
    __shared__ half smemB[cosize_v<decltype(sB_layout)>];

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M, BLK_K, PIPE)
    Tensor sR = make_tensor(make_smem_ptr(smemR), sR_layout); // (GROUP * R, BLOCK * R, PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N, BLK_K, PIPE)

    // Full and Tiled Tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), layout_a); // (M,K)
    Tensor mR = make_tensor(make_gmem_ptr(R), layout_r); // (GROUP * R, BLOCK * R)
    Tensor mB = make_tensor(make_gmem_ptr(B), layout_b); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), layout_c); // (M,N)

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1,X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1,X>{});  // (BLK_M,BLK_N)
    Tensor gR = local_tile(mR,
        make_shape(
            size<1>(blocks_tiler) * reconn_sz,
            size<2>(cta_tiler)
        ), make_coord(blockIdx.y, _)); // (N_GROUPS * R, BLK_K, k)， assuming one thread block would handle at least one group of R
    //
    // Partition the copying of A and B tiles across the threads
    //
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_r = copy_r.get_slice(threadIdx.x);
    Tensor tAgR = thr_copy_r.partition_S(gR); // (CPY,CPY_R,CPY_K,k)
    Tensor tAsR = thr_copy_r.partition_D(sR); // (CPY,CPY_R,CPY_K,PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

    auto K_PIPE_MAX = size<3>(tAsA);
    int k_tile_count = size<3>(tAgA);
    int k_tile_next = 0; // Current tile index in gmem to read from

    // Start async loads for all pipes but the last
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
        if (threadIdx.x < size(copy_a)) {
            // Only copy A if the threadIdx.x is within the range of copy_a
            copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
        }
        if (threadIdx.x < size(copy_r)) {
            // Only copy R if the threadIdx.x is within the range of copy_r
            copy(copy_r, tAgR(_,_,_,k_tile_next), tAsR(_,_,_,k_pipe));
        }
        if (threadIdx.x < size(copy_b)) {
            // Only copy B if the threadIdx.x is within the range of copy_b
            copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
        }
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    //
    // Define A/B partitioning and C accumulators manually
    //
    
    int warp_idx = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;
    auto warp_coord = warp_layout.get_hier_coord(warp_idx); // (WARP_M, WARP_N)

    Tensor _sA_warp_atom = logical_divide(
        sA,
        make_tile(
            make_layout(
                make_shape(
                    size<0>(warp_atom_mn), // WARP_M
                    size<0>(warp_layout)
                )
            ),
            make_layout(
                reconn_sz
            )
        )
    )( // (((WARP_ATOM_M, WAPRS_ALONG_M), REST_M), (RECONN_SZ, BLOCKS_ALONG_K), PIPELINE)
        make_coord(
            make_coord(_, get<0>(warp_coord)),
            _
        ),
        make_coord(_, _), _
    ); // (WARP_ATOM_M, REST_M, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)

    Tensor sA_warp_atom = make_tensor(
        _sA_warp_atom.data(),
        coalesce(group<0,2>(_sA_warp_atom.layout()), Step<_1,_1,_1,_1>{}) // (WARP_M_REGION, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)
    );

    Tensor sB_warp_atom = tiled_divide(
        sB,
        make_tile(
            make_layout(
                group_sz
            ),
            make_layout(
                reconn_sz
            )
        )
    ).compose( // ((GROUP_SZ, RECONN_SZ), N_GROUPS, BLOCKS_ALONG_K, PIPELINE)
        make_tile(
            make_tile(
                warp_in_group_mapping(size<1>(warp_layout), size<1>(blocks_tiler), group_sz), _
            ),
            warp_group_mapping(size<1>(warp_layout), size<1>(blocks_tiler)), _, _
        )
    )( // (((WARP_RESPONSIBLE_SIZE, WARP_N), RECONN_SZ), (GROUP_PER_WARP, WAPR_N), BLOCKS_ALONG_K, PIPELINE)
        make_coord(make_coord(_, get<1>(warp_coord)), _),
        make_coord(_, get<1>(warp_coord)), _, _
    ); // (WARP_RESPONSIBLE_SIZE, RECONN_SZ, GROUP_PER_WARP, BLOCKS_ALONG_K, PIPELINE)

    Tensor sR_warp_atom = tiled_divide(
        sR,
        make_tile(
            make_layout(reconn_sz),
            make_layout(reconn_sz)
        )
    ).compose( // ((R, R), BLK_N_GROUPS, BLOCKS_ALONG_K, PIPELINE)
        make_tile(
            _, warp_group_mapping(size<1>(warp_layout), size<1>(blocks_tiler)),
            _, _
        )
    )( // ((R, R), (GROUP_PER_WARP, WAPR_N), BLOCKS_ALONG_K, PIPELINE)
        make_coord(_, _), make_coord(_, get<1>(warp_coord)), _, _
    ); // (R, R, GROUP_PER_WARP, BLOCKS_ALONG_K, PIPELINE)

    CUTE_STATIC_ASSERT_V(size<1>(sB_warp_atom) == size<1>(sR_warp_atom));

    Tensor _gC_warp = logical_divide(
        gC,
        make_tile(
            make_layout(
                make_shape(
                    size<0>(warp_atom_mn), // WARP_M
                    size<0>(warp_layout)
                )
            ),
            make_layout(
                group_sz
            )
        )
    ).compose( // (((WARP_ATOM_M, WAPRS_ALONG_M), REST_M), (GROUP_SZ, N_GROUPS))
        make_tile(
            _,
            make_tile(
                warp_in_group_mapping(size<1>(warp_layout), size<1>(blocks_tiler), group_sz),
                warp_group_mapping(size<1>(warp_layout), size<1>(blocks_tiler))
            )
        )
    )( // (((WARP_ATOM_M, WAPRS_ALONG_M), REST_M), ((WARP_RESPONSIBLE_SIZE, WARP_N), (GROUP_PER_WARP, WAPR_N)))
        make_coord(make_coord(_, get<0>(warp_coord)), _),
        make_coord(
            make_coord(_, get<1>(warp_coord)),
            make_coord(_, get<1>(warp_coord))
        )
    ); // (WARP_ATOM_M, REST_M, WARP_RESPONSIBLE_SIZE, GROUP_PER_WARP)

    Tensor gC_warp = make_tensor(
        _gC_warp.data(),
        coalesce(group<0,2>(_gC_warp.layout()), Step<_1,_1,_1>{})
    ); // (WARP_M_REGION, WARP_RESPONSIBLE_SIZE, GROUP_PER_WARP)

    TiledMMA single_warp_mma1 = make_tiled_mma(
        mma_atom1{},
        make_layout(make_shape(_1{}, _1{})),
        Tile<decltype(size<0>(sA_warp_atom)), decltype(reconn_sz)>{}
    );
    
    TiledMMA single_warp_mma2 = make_tiled_mma(
        mma_atom2{},
        make_layout(make_shape(_1{}, _1{})),
        Tile<decltype(size<0>(sA_warp_atom)), decltype(size<0>(sB_warp_atom))>{}
    );


    ThrMMA thr_mma1 = single_warp_mma1.get_slice(lane_idx);
    ThrMMA thr_mma2 = single_warp_mma2.get_slice(lane_idx);
    Tensor tCsA = thr_mma1.partition_A(sA_warp_atom); // (MMA, MMA_M, MMA_K, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCsR = thr_mma1.partition_B(sR_warp_atom); // (MMA, MMA_N, MMA_K, GROUP_PER_WARP, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCsB = thr_mma2.partition_B(sB_warp_atom); // (MMA, MMA_N, MMA_K, GROUP_PER_WARP, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCgC = thr_mma2.partition_C(gC_warp); // (MMA, MMA_M, MMA_N, GROUP_PER_WARP)

    Tensor tCrI = thr_mma1.partition_fragment_C(sA_warp_atom(_, _, 0, 0)); // (MMA, MMA_M, MMA_K), Fragment for storing the intermediate result of AR
    Tensor tCrA = thr_mma1.make_fragment_A(tCsA(_, _, _, _, 0)); // (MMA, MMA_M, MMA_K, BLOCKS_ALONG_K)
    Tensor tCrR = thr_mma1.make_fragment_B(tCsR(_, _, _, _, _, 0)); // (MMA, MMA_N, MMA_K, GROUP_PER_WARP, BLOCKS_ALONG_K)
    Tensor tCrB = thr_mma2.make_fragment_B(tCsB(_, _, _, _, _, 0)); // (MMA, MMA_N, MMA_K, GROUP_PER_WARP, BLOCKS_ALONG_K)
    Tensor tCrC = thr_mma2.make_fragment_C(tCgC); // (MMA, MMA_M, MMA_N, GROUP_PER_WARP)
    clear(tCrC); // Clear the accumulators

    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX-1;

    Tensor tCsA_p = tCsA(_,_,_,_,smem_pipe_read);
    Tensor tCsR_p = tCsR(_,_,_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<3>(tCrA);

    // Number of groups for each warp
    auto GROUP_PER_WARP = size<3>(tCrC);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
        // Wait util our first prefetched tile is loaded in
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
        // if (threadIdx.x == 0) {
        //     printf("gA:\n");
        //     Tensor curr_gA = gA(_, _, 0);
        //     for (int i = 0; i < size<0>(curr_gA); ++i) {
        //         for (int j = 0; j < size<1>(curr_gA); ++j) {
        //             printf("%6.3f ", static_cast<float>(curr_gA(i, j)));
        //         }
        //         printf("\n");
        //     }
        //     printf("\ngR:\n");
        //     Tensor curr_gR = gR(_, _, 0);
        //     for (int i = 0; i < size<0>(curr_gR); ++i) {
        //         for (int j = 0; j < size<1>(curr_gR); ++j) {
        //             printf("%6.3f ", static_cast<float>(curr_gR(i, j)));
        //         }
        //         printf("\n");
        //     }
        //     printf("\ngB:\n");
        //     Tensor curr_gB = gB(_, _, 0);
        //     for (int i = 0; i < size<0>(curr_gB); ++i) {
        //         for (int j = 0; j < size<1>(curr_gB); ++j) {
        //             printf("%6.3f ", static_cast<float>(curr_gB(i, j)));
        //         }
        //         printf("\n");
        //     }

        //     // print out the content in the shared memory for debugging
        //     printf("SA:\n");
        //     Tensor curr_sA = sA(_, _, 0);
        //     for (int i = 0; i < size<0>(curr_sA); ++i) {
        //         for (int j = 0; j < size<1>(curr_sA); ++j) {
        //             printf("%6.3f ", static_cast<float>(curr_sA(i, j)));
        //         }
        //         printf("\n");
        //     }
        //     printf("\nSR:\n");
        //     Tensor curr_sR = sR(_, _, 0);
        //     for (int i = 0; i < size<0>(curr_sR); ++i) {
        //         for (int j = 0; j < size<1>(curr_sR); ++j) {
        //             printf("%6.3f ", static_cast<float>(curr_sR(i, j)));
        //         }
        //         printf("\n");
        //     }
        //     printf("\nSB:\n");
        //     Tensor curr_sB = sB(_, _, 0);
        //     for (int i = 0; i < size<0>(curr_sB); ++i) {
        //         for (int j = 0; j < size<1>(curr_sB); ++j) {
        //             printf("%6.3f ", static_cast<float>(curr_sB(i, j)));
        //         }
        //         printf("\n");
        //     }
        // }
        // __syncthreads();

        for (int w = 0; w < size(warp_layout); ++w) {
            if (threadIdx.x == w * 32) {
                printf("Warp (%d, %d):\n", get<0>(warp_coord), get<1>(warp_coord));
                printf("sA:\n");
                Tensor currA = sA_warp_atom(_, _, 0, 0);
                for (int i = 0; i < size<0>(currA); ++i) {
                    for (int j = 0; j < size<1>(currA); ++j) {
                        printf("%6.3f ", static_cast<float>(currA(i, j)));
                    }
                    printf("\n");
                }
                printf("\nsR:\n");
                Tensor currR = sR_warp_atom(_, _, _, 0, 0);
                for (int g = 0; g < size<2>(currR); ++g) {
                    printf("Group %d:\n", g);
                    for (int i = 0; i < size<0>(currR); ++i) {
                        for (int j = 0; j < size<1>(currR); ++j) {
                            printf("%6.3f ", static_cast<float>(currR(i, j, g)));
                        }
                        printf("\n");
                    }
                }
                printf("\nsB:\n");
                Tensor currB = sB_warp_atom(_, _, _, 0, 0);// (WARP_RESPONSIBLE_SIZE, RECONN_SZ, GROUP_PER_WARP, BLOCKS_ALONG_K, PIPELINE)
                for (int g = 0; g < size<2>(currB); ++g) {
                    printf("Group %d:\n", g);
                    for (int i = 0; i < size<0>(currB); ++i) {
                        for (int j = 0; j < size<1>(currB); ++j) {
                            printf("%6.3f ", static_cast<float>(currB(i, j, g)));
                        }
                        printf("\n");
                    }
                }
                printf("\n");
            }
            __syncthreads();
        }
        // if (threadIdx.x == 0) {
        //     Tensor curr = sA_warp_atom(_,_,0,0);
        //     for (int i = 0; i < size<0>(curr); ++i) {
        //         for (int j = 0; j < size<1>(curr); ++j) {
        //             printf("%6.3f ", static_cast<float>(curr(i, j)));
        //         }
        //         printf("\n");
        //     }
        // }
        // __syncthreads();

        // Prefetch the first rmem from the first k-tile
        copy(tCsA_p(_,_,_,Int<0>{}), tCrA(_,_,_,Int<0>{}));
        copy(tCsR_p(_,_,_,_,Int<0>{}), tCrR(_,_,_,_,Int<0>{}));
        copy(tCsB_p(_,_,_,_,Int<0>{}), tCrB(_,_,_,_,Int<0>{}));
    }

    // Don't need the register pipeline with the use of tensor cores
    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1))
    {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            if (k_block == K_BLOCK_MAX - 1) {
                // Slice the smem_pipe_read smem
                tCsA_p = tCsA(_,_,_,_,smem_pipe_read);
                tCsR_p = tCsR(_,_,_,_,_,smem_pipe_read);
                tCsB_p = tCsB(_,_,_,_,_,smem_pipe_read);

                // Commit the smem for smem_pipe_read
                cp_async_wait<K_PIPE_MAX-2>();
                __syncthreads();
            }

            // Load A, B shmem->regs for k_block+1
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static
            copy(tCsA_p(_,_,_,k_block_next), tCrA(_,_,_,k_block_next));
            copy(tCsR_p(_,_,_,_,k_block_next), tCrR(_,_,_,_,k_block_next));
            copy(tCsB_p(_,_,_,_,k_block_next), tCrB(_,_,_,_,k_block_next));

            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == 0) {
                if (threadIdx.x < size(copy_a)) {
                    // Only copy A if the threadIdx.x is within the range of copy_a
                    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
                }
                if (threadIdx.x < size(copy_r)) {
                    // Only copy R if the threadIdx.x is within the range of copy_r
                    copy(copy_r, tAgR(_,_,_,k_tile_next), tAsR(_,_,_,smem_pipe_write));
                }
                if (threadIdx.x < size(copy_b)) {
                    // Only copy B if the threadIdx.x is within the range of copy_b
                    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
                }
                cp_async_fence();

                // Advance the gmem tile
                --k_tile_count;
                if (k_tile_count > 0) { ++k_tile_next; }

                // Advance the smem pipe
                smem_pipe_write = smem_pipe_read;
                ++smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
            }

            // if (threadIdx.x == 32) {
            //     Tensor curr_sA = sA_warp_atom(_, _, k_block, smem_pipe_read);
            //     Tensor curr_sR = sR_warp_atom(_, _, _, k_block, smem_pipe_read);
            //     Tensor curr_sB = sB_warp_atom(_, _, _, k_block, smem_pipe_read);
            //     for (int i = 0; i < size<0>(curr_sA); ++i) {
            //         for (int j = 0; j < size<1>(curr_sA); ++j) {
            //             printf("%6.3f ", static_cast<float>(curr_sA(i, j)));
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            // }
            // __syncthreads();

            CUTE_UNROLL
            for (int group = 0; group < GROUP_PER_WARP; ++group) {
                clear(tCrI); // clear the accumulator for storing AR
                gemm(single_warp_mma1, tCrA(_,_,_,k_block), tCrR(_,_,_,group,k_block),              tCrI);
                gemm(single_warp_mma2,                tCrI, tCrB(_,_,_,group,k_block), tCrC(_,_,_,group));
            }
        }
    }

    // // PREFETCH register pipeline
    // if (K_BLOCK_MAX > 1) {
    //     // Wait util our first prefetched tile is loaded in
    //     cp_async_wait<K_PIPE_MAX-2>();
    //     __syncthreads();

    //     // Prefetch the first rmem from the first k-tile
    //     copy(tCsA_p(_,_,_,Int<0>{}), tCrA(_,_,_,Int<0>{}));
    //     copy(tCsR_p(_,_,_,_,Int<0>{}), tCrR(_,_,_,_,Int<0>{}));
    //     copy(tCsB_p(_,_,_,_,Int<0>{}), tCrB(_,_,_,_,Int<0>{}));
    // }

    // // Don't need the register pipeline with the use of tensor cores
    // CUTE_NO_UNROLL
    // while (k_tile_count > -(K_PIPE_MAX - 1))
    // {
    //     CUTE_UNROLL
    //     for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
    //         if (k_block == K_BLOCK_MAX - 1) {
    //             // Slice the smem_pipe_read smem
    //             tCsA_p = tCsA(_,_,_,_,smem_pipe_read);
    //             tCsR_p = tCsR(_,_,_,_,_,smem_pipe_read);
    //             tCsB_p = tCsB(_,_,_,_,_,smem_pipe_read);

    //             // Commit the smem for smem_pipe_read
    //             cp_async_wait<K_PIPE_MAX-2>();
    //             __syncthreads();
    //         }

    //         // Load A, B shmem->regs for k_block+1
    //         auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static

            

    //         copy(tCsA_p(_,_,_,k_block_next), tCrA(_,_,_,k_block_next));
    //         copy(tCsR_p(_,_,_,_,k_block_next), tCrR(_,_,_,_,k_block_next));
    //         copy(tCsB_p(_,_,_,_,k_block_next), tCrB(_,_,_,_,k_block_next));

    //         // Copy gmem to smem before computing gemm on each k-pipe
    //         if (k_block == 0) {
    //             copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
    //             if (threadIdx.x < size(copy_r)) {
    //                 copy(copy_r, tAgR(_,_,_,k_tile_next), tAsR(_,_,_,smem_pipe_write));
    //             }
    //             copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
    //             cp_async_fence();

    //             // Advance the gmem tile
    //             --k_tile_count;
    //             if (k_tile_count > 0) { ++k_tile_next; }

    //             // Advance the smem pipe
    //             smem_pipe_write = smem_pipe_read;
    //             ++smem_pipe_read;
    //             smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
    //         }

    //         if (threadIdx.x == 0) {
    //             Tensor curr_sA = sA_warp_atom(_, _, k_block, smem_pipe_read);
    //             Tensor curr_sR = sR_warp_atom(_, _, _, k_block, smem_pipe_read);
    //             Tensor curr_sB = sB_warp_atom(_, _, _, k_block, smem_pipe_read);
    //             for (int i = 0; i < size<0>(curr_sA); ++i) {
    //                 for (int j = 0; j < size<1>(curr_sA); ++j) {
    //                     printf("%6.3f ", static_cast<float>(curr_sA(i, j)));
    //                 }
    //                 printf("\n");
    //             }
    //             printf("\n");
    //         }
    //         __syncthreads();

    //         CUTE_UNROLL
    //         for (int group = 0; group < GROUP_PER_WARP; ++group) {
    //             clear(tCrI); // clear the accumulator for storing AR
    //             gemm(single_warp_mma1, tCrA(_,_,_,k_block), tCrR(_,_,_,group,k_block),              tCrI);
    //             gemm(single_warp_mma2,                tCrI, tCrB(_,_,_,group,k_block), tCrC(_,_,_,group));
    //         }
    //     }
    // }
    copy(tCrC, tCgC); // Write back the result to global memory
}
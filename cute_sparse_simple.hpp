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
          class RLayout, class TiledCopyR, class GroupSize,
          class BLayout, class TiledCopyB,
          class CLayout, class WarpLayout, class Pipeline>
__global__ static __launch_bounds__(decltype(size(WarpLayout{}) * cute::_32{})::value)
void oft_device(ProblemShape shape_MNK, BlocksTiler blocks_tiler,
                half const *A, ALayout layout_a, TiledCopyA copy_a,
                half const *R, RLayout layout_r, TiledCopyR copy_r, GroupSize group_size,
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
    auto reconn_sz = size<2>(layout_r);
    CUTE_STATIC_ASSERT_V(reconn_sz == _8{}); // Assume the reconnection size is 8, which is the size of the atom

    auto cta_tiler = make_shape(
        size<0>(blocks_tiler),                    // BLK_M
        size<1>(blocks_tiler) * group_size,       // BLK_N
        size<2>(blocks_tiler) * reconn_sz         // BLK_K
    );

    // auto smem_atom = make_layout( // two rows of padded shared memory layout atom, should be replaced by swizzle
    //     make_shape(_2{}, size<2>(cta_tiler)),
    //     make_stride(size<2>(cta_tiler) + _8{} /*padding size required for half*/, _1{})    
    // );

    constexpr auto k_bit_width = log_2(static_cast<unsigned int>(decltype(size<2>(blocks_tiler))::value));

    auto smem_atom = composition(
          Swizzle<k_bit_width,6 - k_bit_width>{},
          Layout<
            Shape <_8, Shape <_8,  decltype(size<2>(blocks_tiler))>>,
            Stride<_8, Stride<_1, _64>>
          >{}
        );
    
    CUTE_STATIC_ASSERT_V(size<2>(smem_atom) == size<2>(cta_tiler)); // Ensure the shared memory atom size matches the K dimension of the CTA tiler
    
    auto warp_atom_mn = make_shape(
        size<0>(cta_tiler) / size<0>(warp_layout), // BLK_M / WARP_M
        size<1>(cta_tiler) / size<1>(warp_layout)  // BLK_N / WARP_N  <- the size of this shall not be changed, otherwiseit will be wasteful in terms of register usage
    ); // A rather simple way to tile the warps, the shape of the tensor that each warp should handles

    using mma_atom1 = MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>;
    using mma_atom2 = mma_atom1;

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

    // Shared memory buffers
    __shared__ half smemA[cosize_v<decltype(sA_layout)>];
    __shared__ half smemR[cosize_v<decltype(sR_layout_2d)>];
    __shared__ half smemB[cosize_v<decltype(sB_layout)>];

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M, BLK_K, PIPE)
    Tensor sR2d = make_tensor(make_smem_ptr(smemR), sR_layout_2d); // (GROUP * R, BLOCK * R, PIPE)
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
    Tensor tAsR = thr_copy_r.partition_D(sR2d); // (CPY,CPY_R,CPY_K,PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

    auto K_PIPE_MAX = size<3>(tAsA);
    int k_tile_count = size<3>(tAgA);
    int k_tile_next = 0; // Current tile index in gmem to read from

    // Start async loads for all pipes but the last
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
        if (threadIdx.x < size(copy_r)) {
            // Only copy R if the threadIdx.x is within the range of copy_r
            copy(copy_r, tAgR(_,_,_,k_tile_next), tAsR(_,_,_,k_pipe));
        }
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
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

    Tensor a_atom_tiles = logical_divide(
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
    ); // (((WARP_ATOM_M, WAPRS_ALONG_M), REST_M), (RECONN_SZ, BLOCKS_ALONG_K), PIPELINE)

    Tensor _a_warp_tensor = a_atom_tiles(make_coord(make_coord(_, get<0>(warp_coord)), _), make_coord(_, _), _); // (WARP_ATOM_M, REST_M, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)
    Tensor a_warp_tensor = make_tensor(
        _a_warp_tensor.data(),
        coalesce(group<0,2>(_a_warp_tensor.layout()), Step<_1,_1,_1,_1>{}) // (WARP_M_REGION, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)
    );


    Tensor b_warp_tensor = logical_divide(
        sB,
        make_tile(
            make_layout(
                group_sz
            ),
            make_layout(
                reconn_sz
            )
        )
    ).compose( // ((GROUP_SZ, N_GROUPS), (RECONN_SZ, BLOCKS_ALONG_K), PIPELINE)
        make_tile(
            make_tile(_,
                warp_group_mapping(size<1>(warp_layout), size<1>(blocks_tiler))
            ), _, _
        )
    )( // ((GROUP_SZ, (GROUP_PER_WARP, WAPR_N)), (RECONN_SZ, BLOCKS_ALONG_K), PIPELINE)
        make_coord(_,
            make_coord(_, get<1>(warp_coord))
        ), _, _
    ).compose( // (GROUP_SZ, GROUP_PER_WARP, (RECONN_SZ, BLOCKS_ALONG_K), PIPELINE)
        make_tile(
            warp_in_group_mapping(
                size<1>(warp_layout),
                size<1>(blocks_tiler),
                group_sz
            ), _, _, _)
    )( // ((WARP_RESPONSIBLE_SIZE, WARP_N), GROUP_PER_WARP, (RECONN_SZ, BLOCKS_ALONG_K), PIPELINE)
        make_coord(_, get<1>(warp_coord)),
        _, make_coord(_, _), _
    );  // (WARP_RESPONSIBLE_SIZE, GROUP_PER_WARP, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)


    Tensor r_warp_tensor = logical_divide(
        sR2d,
        make_tile(
            make_layout(reconn_sz),
            make_layout(reconn_sz)
        )
    ).compose( // ((R, BLK_N_GROUPS), (R, BLOCKS_ALONG_K), PIPELINE)
        make_tile(
            make_tile(_, warp_group_mapping(size<1>(warp_layout), size<1>(blocks_tiler))),
            _, _
        )
    )( // ((R, (GROUP_PER_WARP, WAPR_N)), (R, BLOCKS_ALONG_K), PIPELINE)
        make_coord(_, make_coord(_, get<1>(warp_coord))),
        make_coord(_, _), _
    ); // (R, GROUP_PER_WARP, R, BLOCKS_ALONG_K, PIPELINE)

    CUTE_STATIC_ASSERT_V(size<1>(b_warp_tensor) == size<1>(r_warp_tensor));

    Tensor gC_warp_all = flat_divide(gC, gC_warp_tile); // (WARP_SIZE_M, WARP_SIZE_N, WARP_M, WARP_N)
    Tensor gC_warp = gC_warp_all(_, _, get<0>(warp_coord), get<1>(warp_coord)); // (WARP_SIZE_M, WARP_SIZE_N)

    TiledMMA single_warp_mma1 = make_tiled_mma(
        mma_atom1{},
        make_layout(make_shape(_1{}, _1{})),
        Tile<decltype(size<0>(warp_atom_mnk)), decltype(reconn_sz)>{}
    );
    
    TiledMMA single_warp_mma2 = make_tiled_mma(
        mma_atom2{},
        make_layout(make_shape(_1{}, _1{})),
        Tile<decltype(size<0>(warp_atom_mnk)), decltype(size<1>(warp_atom_mnk))>{}
    );

    

    ThrMMA thr_mma1 = single_warp_mma1.get_slice(warp_idx);
    ThrMMA thr_mma2 = single_warp_mma2.get_slice(warp_idx);
    Tensor tCsA = thr_mma1.partition_A(sA_warp_atom(_, _, _, get<0>(warp_coord), get<2>(warp_coord), _)); // (MMA, MMA_M, MMA_K, BLOCKS, PIPELINE)
    Tensor tCsR = thr_mma1.partition_B(sR_warp_atom(_, _, _, get<1>(warp_coord), _)); // (MMA, MMA_N, MMA_K, BLOCKS, PIPELINE) // TODO: Group ignorance here
    Tensor tCsB = thr_mma2.partition_B(sB_warp_atom(_, _, _, get<1>(warp_coord), get<2>(warp_coord), _)); // (MMA, MMA_N, MMA_K, BLOCKS, PIPELINE)
    Tensor tCgC = thr_mma2.partition_C(gC_warp);

    Tensor tCrA = thr_mma1.make_fragment_A(tCsA(_, _, _, _, 0)); // (MMA, MMA_M, MMA_K, BLOCKS)
    Tensor tCrR = thr_mma1.make_fragment_B(tCsR(_, _, _, _, 0)); // (MMA, MMA_N, MMA_K, BLOCKS)
    Tensor tCrB = thr_mma2.make_fragment_B(tCsB(_, _, _, _, 0)); // (MMA, MMA_N, MMA_K, BLOCKS)
    Tensor tCrC = thr_mma2.make_fragment_C(tCgC); // (MMA, MMA_M, MMA_N)
    clear(tCrC); // Clear the accumulators

    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX-1;

    Tensor tCsA_p = tCsA(_,_,_,_,smem_pipe_read);
    Tensor tCsR_p = tCsR(_,_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<3>(tCrA);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
        // Wait util our first prefetched tile is loaded in
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();

        // Prefetch the first rmem from the first k-tile
        copy(tCsA_p(_,_,_,Int<0>{}), tCrA(_,_,_,Int<0>{}));
        copy(tCsR_p(_,_,_,Int<0>{}), tCrR(_,_,_,Int<0>{}));
        copy(tCsB_p(_,_,_,Int<0>{}), tCrB(_,_,_,Int<0>{}));
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
                tCsB_p = tCsB(_,_,_,_,smem_pipe_read);

                // Commit the smem for smem_pipe_read
                cp_async_wait<K_PIPE_MAX-2>();
                __syncthreads();
            }

            // Load A, B shmem->regs for k_block+1
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static
            copy(tCsA_p(_,_,_,k_block_next), tCrA(_,_,_,k_block_next));
            copy(tCsR(_,_,_,k_block_next), tCrR(_,_,_,k_block_next));
            copy(tCsB_p(_,_,_,k_block_next), tCrB(_,_,_,k_block_next));

            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == 0) {
                copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
                if (threadIdx.x < size(copy_r)) {
                    copy(copy_r, tAgR(_,_,_,k_tile_next), tAsR(_,_,_,smem_pipe_write));
                }
                copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
                cp_async_fence();

                // Advance the gmem tile
                --k_tile_count;
                if (k_tile_count > 0) { ++k_tile_next; }

                // Advance the smem pipe
                smem_pipe_write = smem_pipe_read;
                ++smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
            }
            gemm(single_warp_mma1, tCrA(_,_,_,k_block), tCrR(_,_,_,k_block), tCrA(_,_,_,k_block));
            gemm(single_warp_mma2, tCrA(_,_,_,k_block), tCrB(_,_,_,k_block), tCrC);
        }
    }
    copy(tCrC, tCgC); // Write back the result to global memory
}
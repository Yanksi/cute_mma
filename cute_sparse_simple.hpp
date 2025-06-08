#pragma once
#include <cute/tensor.hpp>

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

template <class ProblemShape, class BlocksTiler,
          class ALayout, class TiledCopyA,
          class RLayout, class TiledCopyR, class GroupSize,
          class BLayout, class TiledCopyB,
          class CLayout, class WarpLayout>
__global__ static __launch_bounds__(decltype(size(WarpLayout{}) * _32{})::value)
void oft_device(ProblemShape shape_MNK, BlocksTiler blocks_tiler,
                half_t const *A, ALayout layout_a, TiledCopyA copy_a,
                half_t const *R, RLayout layout_r, TiledCopyR copy_r, GroupSize group_size,
                half_t const *B, BLayout layout_b, TiledCopyB copy_b,
                half_t *C, CLayout layout_c, WarpLayout warp_layout)
{
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});    // (M, N, K)
    static_assert(is_static<blocks_tiler>::value);
    CUTE_STATIC_ASSERT_V(rank(blocks_tiler) == Int<3>{}); // (BLK_M, BLK_N_GROUPS, BLK_K_BLOCKS)
    auto reconn_sz = size<2>(layout_r);

    Shape cta_tiler = make_shape(
        size<0>(blocks_tiler),                    // BLK_M
        size<1>(blocks_tiler) * group_size,       // BLK_N
        size<2>(blocks_tiler) * reconn_sz         // BLK_K
    );

    auto smem_atom = make_layout( // two rows of padded shared memory layout atom, should be replaced by swizzle
        make_shape(_2{}, size<2>(cta_tiler)),
        make_stride(size<2>(cta_tiler) + _8{} /*padding size required for half*/, _1{})    
    );

    auto warp_atom_mnk = make_shape(
        size<0>(cta_tiler) / size<0>(warp_layout), // BLK_M / WARP_M
        size<1>(cta_tiler) / size<1>(warp_layout), // BLK_N / WARP_N
        size<2>(cta_tiler) / size<2>(warp_layout)  // BLK_K / WARP_K
    ); // A rather simple way to tile the warps, the shape of the tensor that each warp should handles

    using mma_atom1 = MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>;
    using mma_atom2 = mma_atom1;

    auto pipeline = _3{};

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
        ), make_tuple(_1{}, _1{}, _1{}));
    
    auto sR_layout_4d = tiled_divide(
        sR_layout_2d,
        make_tile(
            make_layout(reconn_sz),
            make_layout(reconn_sz)
        )
    ); // ((R, R), BLK_N_GROUPS, BLK_K_GROUPS, PIPE)

    // Shared memory buffers
    __shared__ half_t smemA[cosize_v<decltype(sA_layout)>];
    __shared__ half_t smemR[cosize_v<decltype(sR_layout_4d)>];
    __shared__ half_t smemB[cosize_v<decltype(sB_layout)>];

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M, BLK_K, PIPE)
    Tensor sR2d = make_tensor(make_smem_ptr(smemR), sR_layout_2d); // (GROUP * R, BLOCK * R, PIPE)
    Tensor sR4d = make_tensor(make_smem_ptr(smemR), sR_layout_4d); // ((R, R), (BLK_N_GROUPS, BLK_K_BLOCKS, PIPE))
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N, BLK_K, PIPE)

    // Full and Tiled Tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), layout_a); // (M,K)
    Tensor mR = make_tensor(make_gmem_ptr(R), layout_r); // (GROUP * R, BLOCK * R)
    Tensor mB = make_tensor(make_gmem_ptr(B), layout_b); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), layout_c); // (M,N)

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1,X,_1>{});  // (BLK_M,BLK_K,k)
    int group_coord = blockIdx.y * size<1>(cta_tiler) / group_size; // (GROUP, BLOCK)
    Tensor gR = local_tile(mR, cta_tiler, make_coord(_, group_coord, _), Step<X,_1,_1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1,X>{});  // (BLK_M,BLK_N)

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
        copy(copy_r, tAgR(_,_,_,k_tile_next), tAsR(_,_,_,k_pipe));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    //
    // Define A/B partitioning and C accumulators manually
    //

    auto sA_warp_tile = make_tile(make_layout(size<0>(warp_atom_mnk)), 
                                  make_layout(size<2>(warp_atom_mnk)));
    auto sB_warp_tile = make_tile(make_layout(size<1>(warp_atom_mnk)),
                                  make_layout(size<2>(warp_atom_mnk)));
        
    auto a_atom_tile = make_tile(make_layout(size<0>(sA_warp_tile)), make_layout(reconn_sz));
    auto a_tensor_layout = flat_divide(sA.layout(), sA_warp_tile);
    auto a_atom_layout = select<0, 1, 3, 4, 5, 6>(flat_divide(a_tensor_layout, a_atom_tile)); // （ATOM_M, ATOM_K, BLOCKS, WARP_M, WARP_K, PIPELINE)
    Tensor sA_warp_atom = make_tensor(make_smem_ptr(smemA), a_atom_layout); // (ATOM_M, ATOM_K, BLOCKS, WARP_M, WARP_K, PIPELINE)

    auto r_warp_group_layout = r_layout_mode1(size<1>(warp_layout), layout<1>(sR4d));
    auto _r_warp_layout = replace<1>(sR4d, r_layout_mode1); // ((RECONN_SZ, RECONN_SZ), (WARP_N, GROUPS_PER_WARP), BLOCKS, PIPELINE)
    auto r_warp_layout = flatten(select<0, 2, 1, 3>(_r_warp_layout)); // (RECONN_SZ, RECONN_SZ, BLOCKS, WARP_N, GROUPS_PER_WARP, PIPELINE)
    CUTE_STATIC_ASSERT_V(size<4>(r_warp_layout) == Int<1>{}); // TODO: for now, suppose each warp would only handle one group, then ignore the group dimension
    Tensor sR_warp_atom = make_tensor(make_smem_ptr(smemR), select<0, 1, 2, 3, 5>(r_warp_layout)); // (RECONN_SZ, RECONN_SZ, BLOCKS, WARP_N, GROUPS_PER_WARP, PIPELINE)

    // For b, there's no need to define the atom tiling
    auto b_tensor_layout = flat_divide(take<1,2>(sB.layout()), sB_warp_tile) // (ATOM_N, ATOM_K, WARP_N, WARP_K)
    Tensor sB_warp_atom = make_tensor(make_smem_ptr(smemB), b_tensor_layout); // (ATOM_N, ATOM_K, WARP_N, WARP_K, PIPELINE)
    
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

    int warp_idx = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;
    auto warp_coord = warp_layout.get_hier_coord(warp_idx); // (WARP_M, WARP_N, WARP_K)

    ThrMMA thr_mma1 = single_warp_mma.get_slice(warp_idx);
    Tensor tCsA = thr_mma1.partition_A(sA_warp_atom(_, _, _, get<0>(warp_coord), get<2>(warp_coord), _)); // (MMA, MMA_M, MMA_K, BLOCKS, PIPELINE)
    Tensor tCsR = thr_mma1.partition_B(sR_warp_atom(_, _, _, get<1>(warp_coord), _)); // (MMA, MMA_N, MMA_K, BLOCKS, PIPELINE) // TODO: Group ignorance here

    Tensor tCrA = thr_mma1.make_fragment_A(tCsA(_, _, _, _, 0)); // (MMA, MMA_M, MMA_K, BLOCKS)
    Tensor tCrR = thr_mma1.make_fragment_B(tCsR(_, _, _, _, 0)); // (MMA, MMA_N, MMA_K, BLOCKS)
}
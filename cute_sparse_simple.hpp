#pragma once
#include <cute/tensor.hpp>

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
    );

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


}
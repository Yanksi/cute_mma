#pragma once
#include <cute/tensor.hpp>


template <class WarpLayout, class GroupSize>
__device__ static auto get_spmma_tiler(WarpLayout warp_layout, GroupSize group_size) {
    static_assert(is_static<WarpLayout>::value);
    static_assert(is_static<GroupSize>::value);
    return make_shape(
        _16{} * size<0>(warp_layout),
        group_size * size<1>(warp_layout),
        _8{}
    );
}


template <class ProblemShape, class BlocksTiler, class Pipeline,
          class ALayout, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class RLayout, class RSmemLayout, class TiledCopyR, class S2RAtomR, class GroupSize,
          class BLayout, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class CLayout, class CSmemLayout, class WarpLayout>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value)
void sparse_device(ProblemShape shape_MNK, BlocksTiler blocks_tiler, Pipeline pipeline,
                   half_t const *A, ALayout layout_a, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
                   half_t const *R, RLayout layout_r, RSmemLayout sR_layout, TiledCopyR copy_r, S2RAtomR s2r_atom_r, GroupSize group_size,
                   half_t const *B, BLayout layout_b, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
                   half_t *C, CLayout layout_c, CSmemLayout, WarpLayout warp_layout)
{
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});    // (M, N, K)
    static_assert(is_static<blocks_tiler>::value);
    static_assert(is_static<warp_layout>::value);
    CUTE_STATIC_ASSERT_V(rank(blocks_tiler) == Int<3>{}); // (BLK_M, BLK_N_GROUPS, BLK_K_BLOCKS)
    CUTE_STATIC_ASSERT_V(rank(layout_r) == Int<4>{});           // (GROUP, BLOCK, R, R)
    CUTE_STATIC_ASSERT_V(size<2>(layout_r) == size<3>(layout_r));     // dR should be square matrices
    auto reconn_sz = size<2>(layout_r);

    Shape reconn_mat = make_shape(reconn_sz, reconn_sz);

    Shape cta_tiler = make_shape(
        size<0>(blocks_tiler),                    // BLK_M
        size<1>(blocks_tiler) * group_size,       // BLK_N
        size<2>(blocks_tiler) * reconn_sz // BLK_K
    );

    Shape r_tiler = make_shape(
        size<1>(blocks_tiler) * reconn_sz, // GROUP * R
        size<2>(blocks_tiler) * reconn_sz, // BLOCK * R
    );

    Shape spmma_atom_shape = get_spmma_tiler(warp_layout, group_size);

    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(spmma_atom_shape) == Int<0>{});
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(spmma_atom_shape) == Int<0>{});
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<2>(spmma_atom_shape) == Int<0>{});


    CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma)); // NumThreads
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma)); // NumThreads

    auto smem_atom = make_layout(
        make_shape(_2{}, size<2>(cta_tiler)),
        make_stride(size<2>(cta_tiler) + _8{} /*padding size required for half*/, _1{})
    )
    
    auto sA_layout = tile_to_shape(smem_atom, make_shape(size<0>(cta_tiler), size<2>(cta_tiler), pipeline));
    auto sB_layout = tile_to_shape(smem_atom, make_shape(size<1>(cta_tiler), size<2>(cta_tiler), pipeline));
    auto sC_layout = make_layout(make_shape(size<0>(cta_tiler), size<1>(cta_tiler)));
    auto sR_layout = tile_to_shape(smem_atom, make_shape(size<0>(r_tiler), size<1>(r_tiler), pipeline));
    auto sR_layout_blocked = select<1,0>(
        zipped_divide(
            sR_layout,
            make_tile(reconn_sz, reconn_sz)
            )
        ); // ((BLK_N_GROUPS, BLK_K_BLOCKS, PIPE), (R, R))
    
    static_assert(is_static<decltype(sA_layout)>::value);
    static_assert(is_static<decltype(sR_layout)>::value);
    static_assert(is_static<decltype(sB_layout)>::value);
    static_assert(is_static<decltype(sC_layout)>::value);

    CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), layout_a.shape()));
    CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), layout_b.shape()));
    CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), layout_c.shape()));

    //
    // Full and Tiled Tensors
    //

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), layout_a); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), layout_b); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), layout_c); // (M,N)
    Tensor mR = make_tensor(make_gmem_ptr(R), layout_r); // (GROUP * R, BLOCK * R)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)
    Tensor gR = local_tile(mR, r_tiler, make_coord(blockIdx.y, _)); // (BLK_N_GROUPS * R, BLK_K_BLOCKS * R, k)

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<decltype(sA_layout)>];
    __shared__ TB smemB[cosize_v<decltype(sB_layout)>];
    __shared__ TR smemR[cosize_v<decltype(sR_layout)>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);    // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);    // (BLK_N,BLK_K,PIPE)
    Tensor sR = make_tensor(make_smem_ptr(smemR), sR_layout);    // (GROUP * R,BLOCK * R,PIPE)
    Tensor sR_blocked = make_tensor(make_smem_ptr(smemR), sR_layout_blocked); // ((BLK_N_GROUPS,BLK_K_BLOCKS,PIPE),(R,R))

    //
    // Partition the copying of A and B tiles across the threads
    //

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

    ThrCopy thr_copy_r = copy_r.get_slice(threadIdx.x);
    Tensor tRgR = thr_copy_r.partition_S(gR); // (CPY,CPY_GROUP,CPY_BLOCK,k)
    Tensor tRsR = thr_copy_r.partition_D(sR); // (CPY,CPY_GROUP,CPY_BLOCK,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tRgR) == size<1>(tRsR)); // CPY_GROUP
    CUTE_STATIC_ASSERT_V(size<2>(tRgR) == size<2>(tRsR)); // CPY_BLOCK

    //
    // PREFETCH
    //

    auto K_PIPE_MAX = size<3>(tAsA);

    // Total count of tiles
    int k_tile_count = size<3>(tAgA);
    // Current tile index in gmem to read from
    int k_tile_next = 0;

    // Start async loads for all pipes but the last
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe)
    {
        copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
        copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        copy(copy_r, tRgR(_, _, _, k_tile_next), tRsR(_, _, _, k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0)
        {
            ++k_tile_next;
        }
    }

    using mma_atom = SM80_16x8x8_F32F16F16F32_TN;
    //
    // Define A/B partitioning and C accumulators
    //

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

    // Allocate registers for pipelining
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0)); // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0)); // (MMA,MMA_N,MMA_K)
    // Allocate the accumulators -- same size as the projected data
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)

    CUTE_STATIC_ASSERT_V((shape(tCrC) == take<0, 3>(shape(tCgC)))); // (MMA,MMA_M,MMA_N)
    CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA)));         // MMA_M
    CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB)));         // MMA_N
    CUTE_STATIC_ASSERT_V((size<2>(tCrA) == size<2>(tCrB)));         // MMA_K

    // Clear the accumulators
    clear(tCrC);

    //
    // Copy Atom retiling
    //

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA); // (CPY,MMA_M,MMA_K,PIPE)
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);  // (CPY,MMA_M,MMA_K)

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB); // (CPY,MMA_N,MMA_K,PIPE)
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);  // (CPY,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(tXrA)); // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tXrA)); // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<1>(tXrB)); // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrB) == size<2>(tXrB)); // MMA_K

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrA : "); print(tCrA); print("\n");
    print("tCrB : "); print(tCrB); print("\n");
    print("tCrC : "); print(tCrC); print("\n");

    print("tXsA : "); print(tXsA); print("\n");
    print("tXrA : "); print(tXrA); print("\n");
    print("tXsB : "); print(tXsB); print("\n");
    print("tXrB : "); print(tXrB); print("\n");
  }
#endif

#if 1

    // Current pipe index in smem to read from
    int smem_pipe_read = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX - 1;

    // Pipe slice
    Tensor tXsA_p = tXsA(_, _, _, smem_pipe_read);
    Tensor tXsB_p = tXsB(_, _, _, smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1)
    {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();

        // Prefetch the first rmem from the first k-tile
        copy(s2r_atom_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
        copy(s2r_atom_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));
    }

    //
    // PIPELINED MAIN LOOP
    // TUTORIAL: Example of a gemm loop that pipelines shared memory using SM80's cp.async instructions
    //           and explicit pipelines in shared memory.
    //   Data is read from global(k_tile_next) to shared(smem_pipe_write).
    //   Data is read from shared(smem_pipe_read) to registers(k_block_next).
    //   Data is computed on registers(b_block).
    //
    //   This allows all copies and compute to overlap:
    //     Copy from gmem->smem can overlap with copies from smem->rmem and compute on rmem.
    //     Copy from smem->rmem can overlap with compute on rmem.
    //

    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1))
    {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
        {
            if (k_block == K_BLOCK_MAX - 1)
            {
                // Slice the smem_pipe_read smem
                tXsA_p = tXsA(_, _, _, smem_pipe_read);
                tXsB_p = tXsB(_, _, _, smem_pipe_read);

                // Commit the smem for smem_pipe_read
                cp_async_wait<K_PIPE_MAX - 2>();
                __syncthreads();
            }

            // Load A, B shmem->regs for k_block+1
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX; // static
            copy(s2r_atom_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
            copy(s2r_atom_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));

            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == 0)
            {
                copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
                copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
                cp_async_fence();

                // Advance the gmem tile
                --k_tile_count;
                if (k_tile_count > 0)
                {
                    ++k_tile_next;
                }

                // Advance the smem pipe
                smem_pipe_write = smem_pipe_read;
                ++smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
            }

            // Thread-level register gemm for k_block
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

#endif

    //
    // Epilogue
    //

    copy(tCrC, tCgC);

    // axpby(alpha, tCrC, beta, tCgC);
}
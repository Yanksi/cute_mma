#pragma once
#include "cute_oft_util.hpp"

template <class ProblemShape, class CtaTiler,
          class ALayout, class TiledCopyA,
          class RLayout, class TiledCopyR, class ReconnSZ
          class BLayout, class TiledCopyB,
          class CLayout, class WarpLayout, class Pipeline>
__global__ static __launch_bounds__(decltype(size(WarpLayout{}) * cute::_32{})::value)
void oft_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                half const *A, ALayout layout_a, TiledCopyA copy_a,
                half const *R, RLayout layout_r, TiledCopyR copy_r, int group_sz, ReconnSZ reconn_sz,
                half const *B, BLayout layout_b, TiledCopyB copy_b,
                half       *C, CLayout layout_c, WarpLayout warp_layout, Pipeline pipeline)
{
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});    // (M, N, K)
    static_assert(is_static<CtaTiler>::value);
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N_GROUPS, BLK_K_BLOCKS)
    // CUTE_STATIC_ASSERT_V(reconn_sz == _8{}); // Assume the reconnection size is 8, which is the size of the atom

    auto smem_atom = get_smem_atom(size<2>(cta_tiler));

    
    CUTE_STATIC_ASSERT_V(size<1>(smem_atom) == size<2>(cta_tiler)); // Ensure the shared memory atom size matches the K dimension of the CTA tiler

    int n_groups = max(size<1>(cta_tiler) / group_sz, 1); // Number of groups in the N dimension
    int thread_block_per_group = max(group_sz / size<1>(cta_tiler), 1); // Number of thread blocks per group

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
            reconn_sz * n_groups,
            size<2>(cta_tiler),
            pipeline
            )
        ), make_tuple(_1{}, _1{}, _1{})); // (GROUP * R, BLOCK * R, PIPE)

    // Shared memory buffers
    extern __shared__ half smem[];
    half *smemA = smem;
    half *smemB = smem + cosize_v<decltype(sA_layout)>;
    half *smemR = smem + cosize_v<decltype(sA_layout)> + cosize_v<decltype(sB_layout)>;

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
            reconn_sz * n_groups, // GROUP * R
            size<2>(cta_tiler)
        ), make_coord(blockIdx.y / thread_block_per_group, _)); // (N_GROUPS * R, BLK_K, k)， assuming one thread block would handle at least one group of R
    //
    // Partition the copying of A and B tiles across the threads
    //
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)
    auto A_valid_threads = size(sA) / size(tAsA); // Number of threads that are valid for the A copy

    ThrCopy thr_copy_r = copy_r.get_slice(threadIdx.x);
    Tensor tAgR = thr_copy_r.partition_S(gR); // (CPY,CPY_R,CPY_K,k)
    Tensor tAsR = thr_copy_r.partition_D(sR); // (CPY,CPY_R,CPY_K,PIPE)
    auto R_valid_threads = size(sR) / size(tRsR); // Number of threads that are valid for the R copy

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)
    auto B_valid_threads = size(sB) / size(tBsB); // Number of threads that are valid for the B copy

    //
    // Define A/B partitioning and C accumulators manually
    //
    
    int warp_idx = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;
    auto warp_coord = warp_layout.get_hier_coord(warp_idx); // (WARP_M, WARP_N)
    uint32_t warp_m = get<0>(warp_coord);
    uint32_t warp_n = get<1>(warp_coord);

    uint32_t threads_along_m = size<0>(warp_layout) * 32;
    uint32_t threads_along_n = size<1>(warp_layout) * 32;

    auto cta_atom_layout_m = make_layout(
        make_shape(
            shape(warp_layout), _8{}
        ),
        LayoutRight{}
    ); // the layout design without breaking 8 contiguous rows

    Tensor _sA_warp_atom = group_modes<1,3>(
        logical_divide(
            sA,
            make_tile(
                cta_atom_layout_m,
                make_layout(reconn_sz)
            )
        )( // ((((WARPS_ALONG_M, WARPS_ALONG_N), 8), REST_M), (RECONN_SZ, BLOCKS_ALONG_K), PIPELINE)
            make_coord(make_coord(make_coord(get<0>(warp_coord), _), _), _),
            make_coord(_,_), _
        ) // (WARPS_ALONG_N, 8, REST_M, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)
    ); // (WARPS_ALONG_N, WARP_M_REGION, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)

    Tensor sA_warp_atom = _sA_warp_atom(get<1>(warp_coord), _, _, _, _); // (WARP_M_REGION, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)
    Tensor sAR_warp_atom_stage_1 = sA_warp_atom; // (WARP_M_REGION, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)
    Tensor sAR_warp_atom_stage_2 = group<1,3>(group<0,2>(_sA_warp_atom)); // (WARP_M_REGION_STAGE2, BLK_K, PIPELINE)
    // For now, store the intermediate result back the shared memory also for the result of the current warp

    auto cta_atom_layout_n = make_layout(
        make_shape(
            size<1>(warp_layout), _8{}
        ),
        LayoutRight{}
    ); // (WAPRS_ALONG_N, _8)

   Tensor sB_warp_atom = group_modes<0,2>(logical_divide(
        sB,
        make_tile(
            cta_atom_layout_n
        )
    )( // (((WARPS_ALONG_N, 8), REST_N), BLK_K, PIPELINE)
        make_coord(
            make_coord(
                get<1>(warp_coord), _
            ), _
        ), _, _
    )); // (8, REST_N, BLK_K, PIPELINE) -> (WARP_N_REGION, BLK_K, PIPELINE)
    
    Tensor sR_warp_atom = flat_divide(
        sR,
        make_tile(
            make_layout(reconn_sz),
            make_layout(reconn_sz)
        )
    ); // (RECONN_SZ, RECONN_SZ, N_GROUPS, BLOCKS_ALONG_K, PIPELINE)

    CUTE_STATIC_ASSERT_V(size<2>(sB_warp_atom) == size<2>(sR_warp_atom)); // N_GROUPS

    Tensor gC_warp = group_modes<1,3>(
            group_modes<0,3>(
                logical_divide(
                    gC,
                    make_tile(
                        cta_atom_layout_m,
                        cta_atom_layout_n
                    )
                )( // ((((WARPS_ALONG_M, WARPS_ALONG_N), _8), REST_M), ((WAPRS_ALONG_N, _8), REST_N))
                    make_coord(make_coord(make_coord(get<0>(warp_coord),_),_),_),
                    make_coord(make_coord(get<1>(warp_coord),_),_)
                ) // (WARPS_ALONG_N, _8, REST_M, _8, REST_N)
            ) // ((WARPS_ALONG_N, _8, REST_M), _8, REST_N)
        ); // ((WARPS_ALONG_N, _8, REST_M), (_8, REST_N)) => (WARP_M_REGION, WARP_N_REGION)

    using mma_atom1 = MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>;
    using mma_atom2 = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;

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

    using s2r_atom_A = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    using s2r_atom_R = Copy_Atom<SM75_U32x1_LDSM_N, half_t>;
    using s2r_atom_B = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;

    ThrMMA thr_mma1 = single_warp_mma1.get_slice(lane_idx);
    ThrMMA thr_mma2 = single_warp_mma2.get_slice(lane_idx);
    Tensor tCsA = thr_mma1.partition_A(sA_warp_atom); // (MMA, MMA_M, MMA_K, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCsR = thr_mma1.partition_B(sR_warp_atom); // (MMA, MMA_N, MMA_K, N_GROUPS, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCsAR_stage1 = thr_mma1.partition_C(sAR_warp_atom_stage1);  // (MMA, MMA_M, MMA_N, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCsAR_stage2 = thr_mma2.partition_A(sAR_warp_atom_stage2); // (MMA, MMA_M, MMA_K, PIPELINE)
    Tensor tCsB = thr_mma2.partition_B(sB_warp_atom); // (MMA, MMA_N, MMA_K, PIPELINE)
    Tensor tCgC = thr_mma2.partition_C(gC_warp); // (MMA, MMA_M, MMA_N)

    Tensor tCrA   = thr_mma1.make_fragment_A(tCsA(_, _, _, _, 0)); // (MMA, MMA_M, MMA_K, BLOCKS_ALONG_K)
    Tensor tCrR   = thr_mma1.make_fragment_B(tCsR(_, _, _, 0, _, 0)); // (MMA, MMA_N, MMA_K, BLOCKS_ALONG_K)
    Tensor tCrAR1 = thr_mma1.make_fragment_C(tCsAR_stage1(_, _, _, _, 0)); // (MMA, MMA_M, MMA_N, BLOCKS_ALONG_K)
    Tensor tCrAR2 = thr_mma2.make_fragment_A(tCsAR_stage2(_, _, _, 0)); // (MMA, MMA_M, MMA_K)
    Tensor tCrB   = thr_mma2.make_fragment_B(tCsB(_, _, _, 0)); // (MMA, MMA_N, MMA_K)
    Tensor tCrC   = thr_mma2.make_fragment_C(tCgC); // (MMA, MMA_M, MMA_N)
    clear(tCrC); // Clear the accumulators

    int blocks_in_group = size<1>(tCrB) / n_groups; // Number of blocks in each group
    // Futher segment the MMA_N dimension in the second stage mma to groups
    Tensor tCsB_grouped = logical_divide(
        tCsB,
        make_tile(
            _, make_layout(blocks_in_group)
        )
    ); // (MMA, (MMA_N, N_GROUPS), MMA_K, PIPELINE)

    Tensor tCrB_grouped = logical_divide(
        tCrB,
        make_tile(
            _, make_layout(blocks_in_group)
        )
    ); // (MMA, (MMA_N, N_GROUPS), MMA_K)

    Tensor tCgC_grouped = logical_divide(
        tCgC,
        make_tile(
            _, _, make_layout(blocks_in_group)
        )
    ); // (MMA, MMA_M, (MMA_N, N_GROUPS))

    Tensor tCrC_grouped = logical_divide(
        tCrC,
        make_tile(
            _, _, make_layout(blocks_in_group)
        )
    ); // (MMA, MMA_M, (MMA_N, N_GROUPS))

    // #ifdef DEBUG
    // if (threadIdx.x == 0) {
    //     print("tCrA: "); print(tCrA);print("\n");
    //     print("tCsA: "); print(tCsA.layout());print("\n");
    //     print(sA.layout());print("\n");
    //     print(sA_warp_atom.layout());print("\n");
    //     print("tCrR: "); print(tCrR);print("\n");
    //     print("tCsR: "); print(tCsR.layout());print("\n");
    //     print("tCrB: "); print(tCrB);print("\n");
    //     print("tCsB: "); print(tCsB.layout());print("\n");
    // }
    // #endif

    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<4>(tCsR));
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<4>(tCsB));
    CUTE_STATIC_ASSERT_V(size<3>(tCsR) == size<3>(tCsB));

    int smem_pipe_read  = 0;
    auto K_PIPE_MAX = size<3>(tAsA);
    int k_tile_next = 0; // Current tile index in gmem to read from
    int k_tile_count = size<3>(tAgA);
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX-1;
    // Number of groups for each warp
    int GROUP_PER_WARP = size<2,1>(tCrC_grouped);

    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
        if (threadIdx.x < A_valid_threads) {
            // Only copy A if the threadIdx.x is within the range of copy_a
            copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
        }
        if (threadIdx.x < R_valid_threads) {
            // Only copy R if the threadIdx.x is within the range of copy_r
            copy(copy_r, tAgR(_,_,_,k_tile_next), tAsR(_,_,_,k_pipe));
        }
        if (threadIdx.x < B_valid_threads) {
            // Only copy B if the threadIdx.x is within the range of copy_b
            copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
        }
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1)) {
        // slice the shared memory for the reading of this round
        Tensor tCsA_p = tCsA(_,_,_,_,smem_pipe_read); // (MMA, MMA_M, MMA_K, BLOCKS_ALONG_K)
        Tensor tCsR_p = tCsR(_,_,_,_,_,smem_pipe_read); // (MMA, MMA_N, MMA_K, N_GROUPS, BLOCKS_ALONG_K)
        Tensor tCsAR_stage1_p = tCsAR_stage1(_,_,_,_,smem_pipe_read);  // (MMA, MMA_M, MMA_N, BLOCKS_ALONG_K)
        Tensor tCsAR_stage2_p = tCsAR_stage2(_,_,_,smem_pipe_read); // (MMA, MMA_M, MMA_K)
        Tensor tCsB_grouped_p = tCsB(_,_,_,smem_pipe_read); // (MMA, (MMA_N, N_GROUPS), MMA_K)
        // Wait for the tile to be read from arrives
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();

        // Load all A into shared memory for releasing the space in shared memory
        copy(tCsA_p, tCrA);
        CUTE_NO_UNROLL
        for (int g = 0; g < n_groups; ++g) {
            copy(tCsR_p(_,_,_,_0{},_), tCrR);
            clear(tCrAR1); // clear the accumulator for storing AR
            for (int j = 0; j < size<3>(tCrR); ++j) {
                // apply the AR transformation
                gemm(single_warp_mma1, tCrA(_,_,_,j), tCrR(_,_,_,j), tCrAR1(_,_,_,j));
                copy(tCrAR1(_,_,_,j), tCsAR_stage1_p(_,_,_,j)); // copy the intermediate result into shared memory
            }
            asm volatile("bar.sync %0, %1;\n"
                             :
                             : "r"(warp_m), "r"(threads_along_n)); // wait for the data to be ready in the smem
            copy(tCsAR_stage2_p, tCrAR2); // load the transformed result into rmem
            copy(tCsB_grouped_p(_, make_coord(_, g), _), tCrB_grouped(_, make_coord(_, g), _));
            gemm(single_warp_mma2, tCrAR2, tCrB_grouped(_, make_coord(_, g), _), tCrC_grouped(_, _, make_coord(_, g)));
        }
        
        // Issue new copy into shared memory if there's still tiles left
        if (k_tile_count > 0) {
            if (threadIdx.x < A_valid_threads) {
                // Only copy A if the threadIdx.x is within the range of copy_a
                copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
            }
            if (threadIdx.x < R_valid_threads) {
                // Only copy R if the threadIdx.x is within the range of copy_r
                copy(copy_r, tAgR(_,_,_,k_tile_next), tAsR(_,_,_,smem_pipe_write));
            }
            if (threadIdx.x < B_valid_threads) {
                // Only copy B if the threadIdx.x is within the range of copy_b
                copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
            }
            cp_async_fence();
        }

        --k_tile_count;
        ++k_tile_next;
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
    }
    copy(tCrC, tCgC); // Write back the result to global memory
}
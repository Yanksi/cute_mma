#pragma once
#include "cute_oft_util.hpp"


template <class TiledCopyA,
          class TiledCopyR, class ReconnectSize,
          class WarpLayoutStage1, class WarpLayoutStage2>
__device__ static inline
void oft_ar(Tensor const &gA, Tensor &sA, TiledCopyA copy_a,
            Tensor const &gR, Tensor &sR, TiledCopyR copy_r, ReconnectSize reconn_sz,
            Tensor &sAR, int thread_idx,
            WarpLayoutStage1 warps_stage1, WarpLayoutStage2 warps_stage2)
{
    // This function should revice blocked corresponding tensors
    using namespace cute;
    ThrCopy thr_copy_a = copy_a.get_slice(thread_idx);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_r = copy_r.get_slice(thread_idx);
    Tensor tRgR = thr_copy_r.partition_S(gR); // (CPY,CPY_M,CPY_K,k)
    Tensor tRsR = thr_copy_r.partition_D(sR); // (CPY,CPY_M,CPY_K,PIPE)

    auto n_warps = size(warps_stage1);
    constexpr uint32_t n_threads1 = n_warps * 32;
    constexpr uint32_t n_threads_total = size(warps_stage2) * 32 + n_threads1;
    uint32_t warp_idx = thread_idx / 32;
    uint32_t lane_idx = thread_idx % 32;

    auto cta_atom_layout_m = make_layout(
        make_shape(n_warps, _8{}), LayoutRight{}
    ); // the layout design without breaking 8 contiguous rows

    Tensor sA_warp_atom = group_modes<0,2>(
        logical_divide(
            sA,
            make_tile(
                cta_atom_layout_m,
                make_layout(reconn_sz)
            )
        )( // (((N_WARPS, 8), REST_M), (RECONN_SZ, BLOCKS_ALONG_K), PIPELINE)
            make_coord(make_coord(warp_idx, _), _),
            make_coord(_,_), _
        ) // (8, REST_M, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)
    ); // (WARP_M_REGION, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)

    Tensor sAR_warp_atom = group_modes<0,2>(
        logical_divide(
            sAR,
            make_tile(cta_atom_layout_m)
        )( // (((N_WARPS, 8), REST_M), RECONN_SZ * N_GROUPS)
            make_coord(make_coord(warp_idx, _), _), _
        ) // (8, REST_M, RECONN_SZ * N_GROUPS)
    ); // (WARP_M_REGION, RECONN_SZ * N_GROUPS)

    Tensor sR_warp_atom = flat_divide(
        sR,
        make_tile(
            _,
            make_layout(reconn_sz)
        )
    ); // (RECONN_SZ * N_GROUPS, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)

    using mma_atom1 = MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>;

    TiledMMA single_warp_mma1 = make_tiled_mma(
        mma_atom1{},
        make_layout(make_shape(_1{}, _1{})),
        Tile<decltype(size<0>(sA_warp_atom)), decltype(reconn_sz)>{}
    );

    using s2r_atom_A = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    using s2r_atom_R = Copy_Atom<SM75_U32x1_LDSM_N, half_t>;

    ThrMMA thr_mma1 = single_warp_mma1.get_slice(lane_idx);
    Tensor tCsA = thr_mma1.partition_A(sA_warp_atom); // (MMA, MMA_M, MMA_K, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCsR = thr_mma1.partition_B(sR_warp_atom); // (MMA, MMA_N, MMA_K, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCsAR = thr_mma1.partition_C(sAR_warp_atom); // (MMA, MMA_M, MMA_N)

    Tensor tCrA = thr_mma1.make_fragment_A(tCsA(_, _, _, _, _0{})); // (MMA, MMA_M, MMA_K, BLOCKS_ALONG_K)
    Tensor tCrR = thr_mma1.make_fragment_B(tCsR(_, _, _, _, _0{})); // (MMA, MMA_N, MMA_K, BLOCKS_ALONG_K)
    Tensor tCrAR = thr_mma1.make_fragment_C(tCsAR); // (MMA, MMA_M, MMA_N)

    int smem_pipe_read = 0;
    auto K_PIPE_MAX = size<3>(tAsA);
    int k_tile_next = 0;
    int k_tile_count = size<3>(tAgA);
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX - 1;
    auto K_BLOCK_MAX = size<3>(tCrA);

    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
        if (lane_idx < size(copy_r)) {
            copy(copy_r, tRgR(_,_,_,k_tile_next), tRsR(_,_,_,k_pipe));
        }
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1)) {
        // slice the shared memory for the reading of this round
        Tensor tCsA_p = tCsA(_,_,_,_,smem_pipe_read);
        Tensor tCsR_p = tCsR(_,_,_,_,smem_pipe_read);
        if (k_tile_count > 0) {
            copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
            if (lane_idx < size(copy_r)) {
                // Only copy R if the threadIdx.x is within the range of copy_r
                copy(copy_r, tRgR(_,_,_,k_tile_next), tRsR(_,_,_,smem_pipe_write));
            }
            cp_async_fence();
        }
        cp_async_wait<K_PIPE_MAX-2>();
        asm volatile("bar.sync 0, %0;\n"
                             :
                             : "n"(n_threads1));
        // wait for the data to be ready in the smem

        // load the first block of smem to rmem
        copy(tCsA_p(_,_,_,_0{}), tCrA(_,_,_,_0{}));
        copy(tCsR_p(_,_,_,_0{}), tCrR(_,_,_,_0{}));

        CUTE_UNROLL
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            auto rmem_pipe_read = j;
            auto rmem_pipe_write = (j + _1{}) % K_BLOCK_MAX;
            if (j < K_BLOCK_MAX - 1) {
                // load the next block
                copy(tCsA_p(_,_,_,j + _1{}), tCrA(_,_,_,rmem_pipe_write));
                copy(tCsR_p(_,_,_,j + _1{}), tCrR(_,_,_,rmem_pipe_write));
            }
            clear(tCrAR);
            gemm(single_warp_mma1, tCrA(_,_,_,rmem_pipe_read), tCrR(_,_,_,rmem_pipe_read), tCrAR);
            asm volatile("bar.sync 2, %0;\n"
                             :
                             : "n"(n_threads_total)); // wait for the previous data to be consumed
            copy(tCrAR, tCsAR); // copy the intermediate result into shared memory
            asm volatile("bar.arrive 1, %0;\n"
                             :
                             : "n"(n_threads_total)); // signal that the data is ready
        }
        --k_tile_count;
        ++k_tile_next;
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
    }
}

template <class TiledCopyB, class GroupSize,
          class WarpLayoutStage1, class WarpLayoutStage2>
__device static inline
void oft_arb(Tensor const &gB, Tensor &sB, TiledCopy copy_b,
             Tensor const &sAR, GroupSize group_sz,
             Tensor &gC, int thread_idx,
             WarpLayoutStage1 warps_stage1, WarpLayoutStage2 warp_layout_stage2)
{
    using namespace cute;
    ThrCopy thr_copy_b = copy_b.get_slice(thread_idx);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

    int warp_idx = thread_idx / 32;
    int lane_idx = thread_idx % 32;
    auto warp_coord = warp_layout_stage2.get_hier_coord(warp_idx); // (WARP_M, WARP_N)
    uint32_t warp_m = get<0>(warp_coord);
    uint32_t warp_n = get<1>(warp_coord);


    auto cta_atom_layout_m = make_layout(
        make_shape(
            size<0>(warp_layout_stage2), _8{}
        ),
        LayoutRight{}
    );

    constexpr uint32_t n_threads2 = size(warp_layout_stage2) * 32;
    constexpr uint32_t n_threads_total = size(warps_stage1) * 32 + n_threads2;
    auto n_groups = max(tile_size_n / group_sz, _1{});
    auto warp_per_group = n_groups / size<1>(warp_layout_stage2);
    auto tile_size_n = size<0>(gB);
    auto warp_tile_n = tile_size_n / size<1>(warp_layout_stage2);
    uint32_t warp_group_id = warp_n / warp_per_group; // The group id of the current warp
    CUTE_STATIC_ASSERT_V(tile_size_n % size<1>(warp_layout_stage2) == _0{}); // Ensure the tile size is divisible by the number of warps in N dimension
    CUTE_STATIC_ASSERT_V(warp_tile_n <= group_sz); // Each warp should handle at most one group
    CUTE_STATIC_ASSERT_V(warp_tile_n >= _8{}); // The size of the atom should be at least 8
    
    Tensor sAR_warp_atom = group_modes<0,2>(
        logical_divide(
            sAR,
            make_tile(
                cta_atom_layout_m,
                make_layout(reconn_sz)
            )
        )( // (((M_WARPS, 8), REST_M), (RECONN_SZ, N_GROUPS))
            make_coord(make_coord(warp_m, _), _), make_coord(_, warp_group_id)
        ) // (8, REST_M, RECONN_SZ)
    ); // (WARP_M_REGION, RECONN_SZ)

    Tensor sB_warp_atom = logical_divide(
        sB,
        make_tile(
            make_layout(warp_tile_n),
            make_layout(reconn_sz)
        )
    )( // ((TILE_N, WARP_ALONG_N), (RECONN_SZ, BLOCKS_ALONG_K), PIPELINE)
        make_coord(_, warp_n), make_coord(_, _), _
    ); // (TILE_N, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)

    Tensor gC_warp = group_modes<0,2>(
        logical_divide(
            gC,
            make_tile(
                cta_atom_layout_m,
                make_layout(warp_tile_n)
            )
        )( // (((M_WARPS, 8), REST_M), (TILE_N, WARP_ALONG_N))
            make_coord(make_coord(warp_m, _), _), make_coord(_, warp_n)
        ) // (8, REST_M, TILE_N)
    );

    using mma_atom2 = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;
    TiledMMA single_warp_mma2 = make_tiled_mma(
        mma_atom2{},
        make_layout(make_shape(_1{}, _1{})),
        Tile<decltype(size<0>(gC_warp)), decltype(size<1>(gC_warp))>{}
    );

    ThrMMA thr_mma2 = single_warp_mma2.get_slice(lane_idx);
    Tensor tCsAR = thr_mma2.partition_A(sAR_warp_atom); // (MMA, MMA_M, MMA_K)
    Tensor tCsB  = thr_mma2.partition_B(sB_warp_atom);  // (MMA, MMA_N, MMA_K, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCgC  = thr_mma2.partition_C(gC_warp);       // (MMA, MMA_M, MMA_N)
    
    Tensor tCrAR = thr_mma2.make_fragment_A(tCsAR); // (MMA, MMA_M, MMA_K)
    Tensor tCrB  = thr_mma2.make_fragment_B(tCsB(_, _, _, _, _0{})); // (MMA, MMA_N, MMA_K, BLOCKS_ALONG_K)
    Tensor tCrC  = thr_mma2.make_fragment_C(tCgC); // (MMA, MMA_M, MMA_N)
    clear(tCrC); // Clear the accumulators

    int smem_pipe_read = 0;
    auto K_PIPE_MAX = size<3>(tBsB);
    int k_tile_next = 0; // Current tile index in gmem to read from
    int k_tile_count = size<3>(tBgB);
    int smem_pipe_write = K_PIPE_MAX - 1;
    auto K_BLOCK_MAX = size<3>(tCrB);
    
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    asm volatile("bar.arrive 2, %0;\n"
                    :
                    : "n"(n_threads_total)); // signal the producer that current threads are ready to consume data

    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1)) {
        // slice the shared memory for the reading of this round
        Tensor tCsB_p = tCsB(_,_,_,_,smem_pipe_read);
        if (k_tile_count > 0) {
            copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
            cp_async_fence();
        }
        // wait for the data to be ready in the smem
        cp_async_wait<K_PIPE_MAX-2>();
        asm volatile("bar.sync 3, %0;\n"
                             :
                             : "n"(n_threads2));

        // load the first block of smem to rmem
        copy(tCsB_p(_,_,_,_0{}), tCrB(_,_,_,_0{}));

        CUTE_UNROLL
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            auto rmem_pipe_read = j;
            auto rmem_pipe_write = (j + _1{}) % K_BLOCK_MAX;
            // wait for producer's data
            asm volatile("bar.sync 1, %0;\n"
                                :
                                : "n"(n_threads2));
            if (j < K_BLOCK_MAX - 1) {
                // load the next block
                copy(tCsAR, tCrAR);
                copy(tCsB_p(_,_,_,j + _1{}), tCrB(_,_,_,rmem_pipe_write));
            }
            asm volatile("bar.arrive 2, %0;\n"
                             :
                             : "n"(n_threads_total)); // signal the producer
            gemm(single_warp_mma2, tCrAR, tCrB(_,_,_,rmem_pipe_write), tCrC);
        }
        --k_tile_count;
        ++k_tile_next;
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
    }
    copy(tCrC, tCgC); // Copy the final result to gC
}

template <class ProblemShape, class CtaTiler,
          class ALayout, class TiledCopyA,
          class RLayout, class TiledCopyR, class GroupSize, class ReconnectSize,
          class BLayout, class TiledCopyB,
          class CLayout, class WarpLayoutStage1, class WarpLayoutStage2, class Pipeline>
__global__ static __launch_bounds__(decltype((size(WarpLayoutStage1{}) + size(WarpLayoutStage2{})) * cute::_32{})::value)
void oft_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                half const *A, ALayout layout_a, TiledCopyA copy_a,
                half const *R, RLayout layout_r, TiledCopyR copy_r, GroupSize group_sz, ReconnectSize reconn_sz,
                half const *B, BLayout layout_b, TiledCopyB copy_b,
                half       *C, CLayout layout_c, WarpLayoutStage1 warp_layout_stage1, WarpLayoutStage2 warp_layout_stage2, Pipeline pipeline)
{
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});    // (M, N, K)
    CUTE_STATIC_ASSERT_V(is_integral<decltype(size<1>(cta_tiler))>{});
    static_assert(is_static<CtaTiler>::value);
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N_GROUPS, BLK_K_BLOCKS)
    // CUTE_STATIC_ASSERT_V(reconn_sz == _8{}); // Assume the reconnection size is 8, which is the size of the atom

    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % group_sz == _0{}); // Ensure the N dimension of the CTA tiler is divisible by group_sz
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % reconn_sz == _0{}); // Ensure the K dimension of the CTA tiler is divisible by reconn_sz
    auto blocks_tiler = make_shape(
        size<0>(cta_tiler),                    // BLK_M
        size<1>(cta_tiler) / group_sz,         // BLK_N_GROUPS
        size<2>(cta_tiler) / reconn_sz          // BLK_K_BLOCKS
    );
    
    auto smem_atom = get_smem_atom(size<2>(cta_tiler));

    
    CUTE_STATIC_ASSERT_V(size<1>(smem_atom) == size<2>(cta_tiler)); // Ensure the shared memory atom size matches the K dimension of the CTA tiler
 
    auto sA_layout = coalesce(tile_to_shape(
        smem_atom,
        make_shape(
            size<0>(cta_tiler), // BLK_M
            size<2>(cta_tiler), // BLK_K
            pipeline            // PIPE
        )
    ), make_tuple(_1{}, _1{}, _1{})); // (BLK_M, BLK_K, PIPE)

    auto ar_smem_atom = get_smem_atom(reconn_sz);
    // for storing the intermediate result of AR
    auto sAR_layout = coalesce(tile_to_shape(
        ar_smem_atom,
        make_shape(
            size<0>(cta_tiler), // BLK_M
            reconn_sz
        )
    ), make_tuple(_1{}, _1{})); // (BLK_M, RECONN_SZ)

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
    __shared__ half smemAR[cosize_v<decltype(sAR_layout)>];

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M, BLK_K, PIPE)
    Tensor sR = make_tensor(make_smem_ptr(smemR), sR_layout); // (GROUP * R, BLOCK * R, PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N, BLK_K, PIPE)
    Tensor sAR = make_tensor(make_smem_ptr(smemAR), sAR_layout); // (BLK_M, RECONN_SZ)

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

    Tensor sA_warp_atom = group_modes<0,2>(
        logical_divide(
            sA,
            make_tile(
                cta_atom_layout_m,
                make_layout(reconn_sz)
            )
        )( // ((((WARPS_ALONG_M, WARPS_ALONG_N), 8), REST_M), (RECONN_SZ, BLOCKS_ALONG_K), PIPELINE)
            make_coord(make_coord(warp_coord, _), _),
            make_coord(_,_), _
        ) // (8, REST_M, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)
    ); // (WARP_M_REGION, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)

    Tensor sAR_warp_atom = group_modes<1,3>(
        logical_divide(
            sAR,
            make_tile(cta_atom_layout_m)
        )( // ((((WARPS_ALONG_M, WARPS_ALONG_N), 8), REST_M), RECONN_SZ)
            make_coord(make_coord(make_coord(get<0>(warp_coord), _), _), _), _
        ) // (WARPS_ALONG_N, 8, REST_M, RECONN_SZ)
    ); // (WARPS_ALONG_N, WARP_M_REGION, RECONN_SZ)

    Tensor sAR_warp_atom_stage1 = sAR_warp_atom(get<1>(warp_coord), _, _); // (WARP_M_REGION, RECONN_SZ)
    Tensor sAR_warp_atom_stage2 = group_modes<0,2>(sAR_warp_atom); // (WARP_M_REGION_STAGE2, RECONN_SZ)

    // For now, store the intermediate result back the shared memory also for the result of the current warp

    auto cta_atom_layout_n = layout<0>(tile_to_shape(make_layout(
        make_shape(
            make_shape(size<1>(warp_layout), _8{})
        ),
        LayoutRight{}
    ), make_shape(group_sz))); // ((WAPRS_ALONG_N, _8), REST_N)

    Tensor sB_warp_atom = group_modes<0,2>(
        flat_divide(
            sB,
            make_tile(
                cta_atom_layout_n,
                make_layout(reconn_sz)
            )
        )( // (((WAPRS_ALONG_N, _8), REST_N), RECONN_SZ, N_GROUPS, BLOCKS_ALONG_K, PIPELINE)
            make_coord(make_coord(get<1>(warp_coord), _), _),
            _, _, _, _
        ) // (_8, REST_N, RECONN_SZ, N_GROUPS, BLOCKS_ALONG_K, PIPELINE)
    ); // (WARP_N_REGION_IN_GROUP, RECONN_SZ, N_GROUPS, BLOCKS_ALONG_K, PIPELINE)
    
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
            )( // ((((WARPS_ALONG_M, WARPS_ALONG_N), _8), REST_M), (((WAPRS_ALONG_N, _8), REST_N), N_GROUPS))
                make_coord(make_coord(make_coord(get<0>(warp_coord),_),_),_),
                make_coord(make_coord(make_coord(get<1>(warp_coord),_),_),_)
            ) // (WARPS_ALONG_N, _8, REST_M, _8, REST_N, N_GROUPS)
        ) // ((WARPS_ALONG_N, _8, REST_M), _8, REST_N, N_GROUPS)
    ); // ((WARPS_ALONG_N, _8, REST_M), (_8, REST_N), N_GROUPS) => (WARP_M_REGION, WARP_N_REGION_IN_GROUP, N_GROUPS)

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
    Tensor tCsAR_stage1 = thr_mma1.partition_C(sAR_warp_atom_stage1);  // (MMA, MMA_M, MMA_N)
    Tensor tCsAR_stage2 = thr_mma2.partition_A(sAR_warp_atom_stage2); // (MMA, MMA_M, MMA_K)
    Tensor tCsB = thr_mma2.partition_B(sB_warp_atom); // (MMA, MMA_N, MMA_K, N_GROUPS, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCgC = thr_mma2.partition_C(gC_warp); // (MMA, MMA_M, MMA_N, N_GROUPS)

    Tensor tCrA   = thr_mma1.make_fragment_A(tCsA(_, _, _, _, 0)); // (MMA, MMA_M, MMA_K, BLOCKS_ALONG_K)
    Tensor tCrR   = thr_mma1.make_fragment_B(tCsR(_, _, _, _, _, 0)); // (MMA, MMA_N, MMA_K, N_GROUPS, BLOCKS_ALONG_K)
    Tensor tCrAR1 = thr_mma1.make_fragment_C(tCsAR_stage1); // (MMA, MMA_M, MMA_N)
    Tensor tCrAR2 = thr_mma2.make_fragment_A(tCsAR_stage2); // (MMA, MMA_M, MMA_K)
    Tensor tCrB   = thr_mma2.make_fragment_B(tCsB(_, _, _, _, _, 0)); // (MMA, MMA_N, MMA_K, N_GROUPS, BLOCKS_ALONG_K)
    Tensor tCrC   = thr_mma2.make_fragment_C(tCgC); // (MMA, MMA_M, MMA_N, N_GROUPS)
    clear(tCrC); // Clear the accumulators

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
    // Size of the register pipeline
    auto K_BLOCK_MAX = size<3>(tCrA);
    // Number of groups for each warp
    auto GROUP_PER_WARP = size<3>(tCrC);

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

    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1)) {
        // slice the shared memory for the reading of this round
        Tensor tCsA_p = tCsA(_,_,_,_,smem_pipe_read);
        Tensor tCsR_p = tCsR(_,_,_,_,_,smem_pipe_read);
        Tensor tCsB_p = tCsB(_,_,_,_,_,smem_pipe_read);
        // Issue new copy into shared memory if there's still tiles left
        if (k_tile_count > 0) {
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
        }
        // Wait for the tile to be read from arrives
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();

        // Load the first block of smem to rmem
        copy(tCsA_p(_,_,_,_0{}), tCrA(_,_,_,_0{}));
        copy(tCsR_p(_,_,_,_,_0{}), tCrR(_,_,_,_,_0{}));
        copy(tCsB_p(_,_,_,_,_0{}), tCrB(_,_,_,_,_0{}));

        CUTE_UNROLL
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            // auto rmem_pipe_read = j % _2{};
            // auto rmem_pipe_write = (j + _1{}) % _2{};
            auto rmem_pipe_read = j;
            auto rmem_pipe_write = (j + _1{}) % K_BLOCK_MAX;
            // Load the next block
            if (j < K_BLOCK_MAX - 1) {
                copy(tCsA_p(_,_,_,  j + _1{}), tCrA(_,_,_,rmem_pipe_write));
                copy(tCsR_p(_,_,_,_,j + _1{}), tCrR(_,_,_,_,rmem_pipe_write));
                copy(tCsB_p(_,_,_,_,j + _1{}), tCrB(_,_,_,_,rmem_pipe_write));
            }
            CUTE_UNROLL
            for (int group = 0; group < GROUP_PER_WARP; ++group) {
                clear(tCrAR1); // clear the accumulator for storing AR
                gemm(single_warp_mma1, tCrA(_,_,_,rmem_pipe_read), tCrR(_,_,_,group,rmem_pipe_read), tCrAR1);
                copy(tCrAR1, tCsAR_stage1); // copy the intermediate result into shared memory
                // __syncthreads();
                asm volatile("bar.sync %0, %1;\n"
                             :
                             : "r"(warp_m), "r"(threads_along_n)); // wait for the data to be ready in the smem
                copy(tCsAR_stage2, tCrAR2); // load the transformed result into rmem
                gemm(single_warp_mma2, tCrAR2, tCrB(_,_,_,group,rmem_pipe_read), tCrC(_,_,_,group));
            }
        }
        --k_tile_count;
        ++k_tile_next;
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
    }
    copy(tCrC, tCgC); // Write back the result to global memory
}
#pragma once
#include "cute_oft_util.hpp"
#include "z_curve.hpp"


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

    Tensor sR_warp_atom = logical_divide(
        sR,
        make_tile(
            _,
            make_layout(reconn_sz)
        )
    )( // (RECONN_SZ * N_GROUPS, (RECONN_SZ, BLOCKS_ALONG_K), PIPELINE)
        _, make_coord(_,_), _
    ); // (RECONN_SZ * N_GROUPS, RECONN_SZ, BLOCKS_ALONG_K, PIPELINE)
    
    using mma_atom1 = MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>;

    TiledMMA single_warp_mma1 = make_tiled_mma(
        mma_atom1{},
        make_layout(make_shape(_1{}, _1{})),
        Tile<decltype(size<0>(sA_warp_atom)), decltype(reconn_sz)>{}
    );

    ThrMMA thr_mma1 = single_warp_mma1.get_slice(lane_idx);
    Tensor tCsA = thr_mma1.partition_A(sA_warp_atom); // (MMA, MMA_M, MMA_K, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCsR = thr_mma1.partition_B(sR_warp_atom); // (MMA, MMA_N, MMA_K, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCsAR = thr_mma1.partition_C(sAR_warp_atom); // (MMA, MMA_M, MMA_N)

    Tensor tCrA = thr_mma1.make_fragment_A(tCsA(_, _, _, _0{}, _0{})); // (MMA, MMA_M, MMA_K)
    Tensor tCrR = thr_mma1.make_fragment_B(tCsR(_, _, _, _0{}, _0{})); // (MMA, MMA_N, MMA_K)
    Tensor tCrAR = thr_mma1.make_fragment_C(tCsAR); // (MMA, MMA_M, MMA_N)

    using s2r_atom_A = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    using s2r_atom_R = Copy_Atom<SM75_U32x1_LDSM_N, half_t>;

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_A{}, single_warp_mma1);
    ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(lane_idx);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA_warp_atom);  // (CPY, CPY_M, CPY_K, BLOCKS_ALONG_K, PIPELINE)
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

    TiledCopy s2r_copy_r = make_tiled_copy_B(s2r_atom_R{}, single_warp_mma1);
    ThrCopy s2r_thr_copy_r = s2r_copy_r.get_slice(lane_idx);
    Tensor tXsR = s2r_thr_copy_r.partition_S(sR_warp_atom); // (CPY, CPY_N, CPY_K, BLOCKS_ALONG_K, PIPELINE)
    Tensor tXrR = s2r_thr_copy_r.retile_D(tCrR); // (CPY, CPY_N, CPY_K)

    int smem_pipe_read = 0;
    auto K_PIPE_MAX = size<3>(tAsA);
    int k_tile_next = 0;
    int k_tile_count = size<3>(tAgA);
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX - 1;
    auto K_BLOCK_MAX = size<3>(tCsA);

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
        Tensor tXsA_p = tXsA(_,_,_,_,smem_pipe_read);
        Tensor tXsR_p = tXsR(_,_,_,_,smem_pipe_read);
        if (k_tile_count > 0) {
            copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
            if (lane_idx < size(copy_r)) {
                // Only copy R if the threadIdx.x is within the range of copy_r
                copy(copy_r, tRgR(_,_,_,k_tile_next), tRsR(_,_,_,smem_pipe_write));
            }
            cp_async_fence();
        }
        cp_async_wait<K_PIPE_MAX-1>();
        asm volatile("bar.sync 0, %0;\n"
                            :
                            : "n"(n_threads1));
        // wait for the data to be ready in the smem

        CUTE_UNROLL
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            copy(s2r_atom_A{}, tXsA_p(_,_,_,j), tXrA);
            copy(s2r_atom_R{}, tXsR_p(_,_,_,j), tXrR);
            clear(tCrAR);
            gemm(single_warp_mma1, tCrA, tCrR, tCrAR);
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
    auto warp_per_group = size<1>(warp_layout_stage2) / n_groups;
    auto tile_size_n = size<0>(gB);
    auto warp_tile_n = tile_size_n / size<1>(warp_layout_stage2);
    uint32_t warp_group_id = warp_n / warp_per_group; // The group id of the current warp
    CUTE_STATIC_ASSERT_V(tile_size_n % size<1>(warp_layout_stage2) == _0{}); // Ensure the tile size is divisible by the number of warps in N dimension
    CUTE_STATIC_ASSERT_V(warp_tile_n <= group_sz); // Each warp should handle at most one group
    CUTE_STATIC_ASSERT_V(warp_tile_n >= _8{}); // The size of the atom should be at least 8
    CUTE_STATIC_ASSERT_V(warp_per_group >= _1{}); // Each group should have at least one warp
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
    );  // (WARP_M_REGION, TILE_N)

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
    Tensor tCrB  = thr_mma2.make_fragment_B(tCsB(_, _, _, _0{}, _0{})); // (MMA, MMA_N, MMA_K)
    Tensor tCrC  = thr_mma2.make_fragment_C(tCgC); // (MMA, MMA_M, MMA_N)
    clear(tCrC); // Clear the accumulators

    using s2r_atom_AR = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    using s2r_atom_B = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;

    TiledCopy s2r_copy_ar = make_tiled_copy_A(s2r_atom_AR{}, single_warp_mma2);
    ThrCopy s2r_thr_copy_ar = s2r_copy_ar.get_slice(lane_idx);
    Tensor tXsAR = s2r_thr_copy_ar.partition_S(sAR_warp_atom); // (CPY, CPY_M, CPY_K)
    Tensor tXrAR = s2r_thr_copy_ar.retile_D(tCrAR); // (CPY, CPY_M, CPY_K)

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_B{}, single_warp_mma2);
    ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(lane_idx);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB_warp_atom); // (CPY, CPY_N, CPY_K, BLOCKS_ALONG_K, PIPELINE)
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

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
        Tensor tXsB_p = tXsB(_,_,_,_,smem_pipe_read);
        if (k_tile_count > 0) {
            copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
        }
        cp_async_fence();
        // wait for the data to be ready in the smem
        cp_async_wait<K_PIPE_MAX-1>();
        asm volatile("bar.sync 3, %0;\n"
                            :
                            : "n"(n_threads2));

        CUTE_UNROLL
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            copy(s2r_atom_B{}, tXsB_p(_,_,_,j), tXrB);
            // wait for producer's data
            asm volatile("bar.sync 1, %0;\n"
                                :
                                : "n"(n_threads_total));
            copy(s2r_atom_AR{}, tXsAR, tXrAR);
            asm volatile("bar.arrive 2, %0;\n"
                                :
                                : "n"(n_threads_total)); // signal the producer
            gemm(single_warp_mma2, tCrAR, tCrB, tCrC);
        }
        --k_tile_count;
        ++k_tile_next;
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
    }
    copy(tCrC, tCgC); // Copy the final result to gC
}

template <class GridShape, class CtaTiler,
          class ALayout, class TiledCopyA,
          class RLayout, class TiledCopyR, class GroupSize, class ReconnectSize,
          class BLayout, class TiledCopyB,
          class CLayout, class WarpLayoutStage1, class WarpLayoutStage2, class Pipeline>
__global__ static __launch_bounds__(decltype((size(WarpLayoutStage1{}) + size(WarpLayoutStage2{})) * cute::_32{})::value)
void oft_device(GridShape grid_shape, CtaTiler cta_tiler,
                half const *A, ALayout layout_a, TiledCopyA copy_a,
                half const *R, RLayout layout_r, TiledCopyR copy_r, GroupSize group_sz, ReconnectSize reconn_sz,
                half const *B, BLayout layout_b, TiledCopyB copy_b,
                half       *C, CLayout layout_c, WarpLayoutStage1 warp_layout_stage1, WarpLayoutStage2 warp_layout_stage2, Pipeline pipeline)
{
    using namespace cute;

    // Preconditions
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

    auto grid_coord = z_curve(grid_shape, blockIdx.x);
    // auto grid_coord = make_tuple(blockIdx.x / get<1>(grid_shape),
    //                              blockIdx.x % get<1>(grid_shape)); // (m,n)
    auto cta_coord =  append<3>(grid_coord, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1,X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1,X>{});  // (BLK_M,BLK_N)
    Tensor gR = local_tile(mR,
        make_shape(
            size<1>(blocks_tiler) * reconn_sz,
            size<2>(cta_tiler)
        ), make_coord(blockIdx.y, _)); // (N_GROUPS * R, BLK_K, k)， assuming one thread block would handle at least one group of R
    
    int thread_idx = threadIdx.x;
    int stage1_threads = size(warp_layout_stage1) * 32;
    if (thread_idx < stage1_threads) {
        // Call the AR kernel
        oft_ar(
            gA, sA, copy_a,
            gR, sR, copy_r, reconn_sz,
            sAR, thread_idx,
            warp_layout_stage1, warp_layout_stage2
        );
    } else {
        // Call the ARB kernel
        oft_arb(
            gB, sB, copy_b,
            sAR, group_sz,
            gC, thread_idx - stage1_threads,
            warp_layout_stage1, warp_layout_stage2
        );
    }
}
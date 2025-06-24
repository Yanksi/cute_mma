#pragma once
#include "cute_oft_util.hpp"

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

    // for storing the intermediate result of AR
    auto sAR_layout = coalesce(tile_to_shape(
        smem_atom,
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

    Tensor _sAR_warp_atom = logical_divide(
        sAR,
        make_tile(
            make_layout(
                make_shape(
                    size(warp_atom_mn), // all warps will work together for transforming A
                    size<0>(warp_layout)
                )
            ),
            _
        )
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
    Tensor tCsR = thr_mma1.partition_B(sR_warp_atom); // (MMA, MMA_N, MMA_K, GROUP_PER_WARP, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCsB = thr_mma2.partition_B(sB_warp_atom); // (MMA, MMA_N, MMA_K, GROUP_PER_WARP, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCgC = thr_mma2.partition_C(gC_warp); // (MMA, MMA_M, MMA_N, GROUP_PER_WARP)

    // Only for two stage rmem pipeline
    // auto rmem_pipe_len = min(size<3>(tCsA), _2{});
    // Tensor tCrA = make_tensor_like<half>(replace<3>(tCsA(_, _, _, _, 0).layout(), make_layout(rmem_pipe_len, _2048{}))); // (MMA, MMA_M, MMA_K, BLOCKS_ALONG_K)
    // Tensor tCrR = make_tensor_like<half>(replace<4>(tCsR(_, _, _, _, _, 0).layout(), make_layout(rmem_pipe_len, _2048{}))); // (MMA, MMA_N, MMA_K, GROUP_PER_WARP, BLOCKS_ALONG_K)
    // Tensor tCrB = make_tensor_like<half>(replace<4>(tCsB(_, _, _, _, _, 0).layout(), make_layout(rmem_pipe_len, _2048{}))); // (MMA, MMA_N, MMA_K, GROUP_PER_WARP, BLOCKS_ALONG_K)

    // Tensor tCrA = make_tensor_like<half>(replace<3>(tCsA(_, _, _, _, 0).layout(), make_layout(_2{}, _2048{}))); // (MMA, MMA_M, MMA_K, BLOCKS_ALONG_K)
    // Tensor tCrR = make_tensor_like<half>(replace<4>(tCsR(_, _, _, _, _, 0).layout(), make_layout(_2{}, _2048{}))); // (MMA, MMA_N, MMA_K, GROUP_PER_WARP, BLOCKS_ALONG_K)
    // Tensor tCrB = make_tensor_like<half>(replace<4>(tCsB(_, _, _, _, _, 0).layout(), make_layout(_2{}, _2048{}))); // (MMA, MMA_N, MMA_K, GROUP_PER_WARP, BLOCKS_ALONG_K)

    // Tensor tCrA = make_tensor_like<half>(tCsA(_, _, _, _, 0).layout()); // (MMA, MMA_M, MMA_K, BLOCKS_ALONG_K)
    // Tensor tCrR = make_tensor_like<half>(tCsR(_, _, _, _, _, 0).layout()); // (MMA, MMA_N, MMA_K, GROUP_PER_WARP, BLOCKS_ALONG_K)
    // Tensor tCrB = make_tensor_like<half>(tCsB(_, _, _, _, _, 0).layout()); // (MMA, MMA_N, MMA_K, GROUP_PER_WARP, BLOCKS_ALONG_K)

    Tensor tCrA = thr_mma1.make_fragment_A(tCsA(_, _, _, _, 0)); // (MMA, MMA_M, MMA_K, BLOCKS_ALONG_K)
    Tensor tCrR = thr_mma1.make_fragment_B(tCsR(_, _, _, _, _, 0)); // (MMA, MMA_N, MMA_K, GROUP_PER_WARP, BLOCKS_ALONG_K)
    Tensor tCrB = thr_mma2.make_fragment_B(tCsB(_, _, _, _, _, 0)); // (MMA, MMA_N, MMA_K, GROUP_PER_WARP, BLOCKS_ALONG_K)
    
    Tensor tCrC = thr_mma2.make_fragment_C(tCgC); // (MMA, MMA_M, MMA_N, GROUP_PER_WARP)
    Tensor tCrI = thr_mma1.partition_fragment_C(sA_warp_atom(_, _, 0, 0)); // (MMA, MMA_M, MMA_K), Fragment for storing the intermediate result of AR
    clear(tCrC); // Clear the accumulators

    #ifdef DEBUG
    if (threadIdx.x == 0) {
        print("tCrA: "); print(tCrA);print("\n");
        print("tCsA: "); print(tCsA.layout());print("\n");
        print(sA.layout());print("\n");
        print(sA_warp_atom.layout());print("\n");
        print("tCrR: "); print(tCrR);print("\n");
        print("tCsR: "); print(tCsR.layout());print("\n");
        print("tCrB: "); print(tCrB);print("\n");
        print("tCsB: "); print(tCsB.layout());print("\n");
    }
    #endif

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
                clear(tCrI); // clear the accumulator for storing AR
                gemm(single_warp_mma1, tCrA(_,_,_,rmem_pipe_read), tCrR(_,_,_,group,rmem_pipe_read),              tCrI);
                gemm(single_warp_mma2,                       tCrI, tCrB(_,_,_,group,rmem_pipe_read), tCrC(_,_,_,group));
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
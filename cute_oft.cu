/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <random>
// #include "cute_oft_simple.hpp"
#include "cute_oft_coop.hpp"
#ifdef USE_CUBLAS
#include "cublas_oft.hpp"
#endif
#include "cpu_oft.hpp"
#include "z_curve.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

#include <cutlass/util/print_error.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/helper_cuda.hpp>

#include <argparse/argparse.hpp>
#include <map>
#define mmax(a,b) ((a) > (b) ? (a) : (b))
#define mmin(a,b) ((a) < (b) ? (a) : (b))

// #ifdef DEBUG
// #define GROUP_SIZE 64
// #else
// #define GROUP_SIZE 256
// #endif
#define GROUP_SIZE 256

namespace cute {
  template <typename TO, typename TR>
  struct Params {
    static_assert(sizeof(TO) == 0, "This struct should not be used");
  };

  template <>
  struct Params <half, half> {
    static const unsigned int bM = 128;
    static const unsigned int group_size = 256;
    static const unsigned int reconn_sz = 8;
    static const unsigned int bN_group = 1;
    static const unsigned int bK_block = 2;
    static const unsigned int bN = bN_group * group_size;
    static const unsigned int bK = bK_block * reconn_sz;
    static const unsigned int bP = 3;
    static const bool block_tiling_copy = true;
    using warp_layout = Layout<Shape<Int<4>, Int<2>>>;
    // using mma_atom = SM80_16x8x8_F16F16F16F16_TN;
    // using s2r_atom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
  };
}

template <typename copy_as_t, typename ele_t, bool k_major, bool block_tiling,
  typename _BM, typename _BK, typename _N_Threads>
constexpr auto cp_layout(_BM bm, _BK bk, _N_Threads _total_threads) {
  using namespace cute;
  constexpr int vec_width = sizeof(copy_as_t) / sizeof(ele_t);
  constexpr int total_elements = bm * bk;

  constexpr int needed_threads = total_elements / vec_width;
  CUTE_STATIC_ASSERT(total_elements % vec_width == 0, "total number of elements shall be divisible by the vector length");
  constexpr int total_threads = mmin(_total_threads, needed_threads);

  constexpr int elements_per_thread = total_elements / total_threads;
  CUTE_STATIC_ASSERT(total_elements % total_threads == 0, "total number of elements shall be divisible by the number of threads using");
  CUTE_STATIC_ASSERT(elements_per_thread % vec_width == 0, "number of elements handled by each thread should be divisible by the vector width");
  constexpr int cp_width = (block_tiling) ? vec_width : elements_per_thread;
  if constexpr (k_major) {
    CUTE_STATIC_ASSERT(!block_tiling || bk % cp_width == 0);
    CUTE_STATIC_ASSERT(block_tiling || (bk % cp_width == 0 || cp_width % bk == 0));
    constexpr int threads_along_k = mmax(bk / cp_width, 1);
    constexpr int threads_k_size = bk / threads_along_k;
    constexpr int threads_m_size = mmax(cp_width / bk, 1);
    constexpr int threads_along_m = total_threads / threads_along_k;
    return make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<copy_as_t>, ele_t>{},
                           make_layout(Shape<Int<threads_along_m>, Int<threads_along_k>>{}, LayoutRight{}),
                          //  Layout<Shape<Int<threads_along_m>, Int<threads_along_k>>>{},
                           Layout<Shape<Int<threads_m_size>, Int<threads_k_size>>>{});
  } else {
    // As it not really possible to have copy width greater than bm, we don't need to check for that
    CUTE_STATIC_ASSERT(bm % cp_width == 0);
    constexpr int threads_along_m = bm / cp_width;
    constexpr int threads_along_k = total_threads / threads_along_m;
    // return make_tiled_copy(Copy_Atom<UniversalCopy<copy_as_t>, ele_t>{},
    return make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<copy_as_t>, ele_t>{},
                           Layout<Shape<Int<threads_along_m>, Int<threads_along_k>>>{},
                           Layout<Shape<Int<cp_width>, _1>>{});
  }
}

// // Setup params for a TN GEMM, K-Major inputs
void oft_tn(int m, int n, int k,
        half const* A, int ldA,
        half const* B, int ldB,
        half const* R, int ldR,
        half      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  using CurrParams = Params<half, half>;

  // Define CTA tile sizes (static)
  auto group_size = Int<CurrParams::group_size>{}; // Group size for the block tiling
  auto reconn_sz = Int<CurrParams::reconn_sz>{}; // Reconnection size for the block tiling
  auto bM = Int<CurrParams::bM>{};
  auto bN_group = Int<CurrParams::bN_group>{};
  auto bN = bN_group * group_size;
  auto bK_block = Int<CurrParams::bK_block>{};
  auto bK = bK_block * reconn_sz;
  auto blocks_tiler = make_shape(bM, bN_group, bK_block);                   // (BLK_M, BLK_N, BLK_K)
  auto cta_tiler = make_shape(bM, bN, bK);                   // (CTA_M, CTA_N, CTA_K)
  auto bP = Int<CurrParams::bP>{};  // Pipeline
  int n_groups = N / group_size;
  auto warp_layout = typename CurrParams::warp_layout{};

  // Define the gmem layouts
  auto A_layout = make_layout(
    make_shape(M, K),
    make_stride(ldA, Int<1>{})
  );

  auto B_layout = make_layout(
    make_shape(N, K),
    make_stride(ldB, Int<1>{})
  );

  auto R_layout = make_layout(
    make_shape(n_groups * reconn_sz, K),
    make_stride(ldR, Int<1>{})
  );

  auto C_layout = make_layout(
    make_shape(M, N),
    make_stride(ldC, Int<1>{})
  );

  TiledCopy copyA = cp_layout<uint128_t, half, true, CurrParams::block_tiling_copy>(bM, bK, size(warp_layout) * _32{});
  TiledCopy copyB = cp_layout<uint128_t, half, true, CurrParams::block_tiling_copy>(bN, bK, size(warp_layout) * _32{});
  TiledCopy copyR = cp_layout<uint128_t, half, true, CurrParams::block_tiling_copy>(bN_group * reconn_sz, bK, size(warp_layout) * _32{});

  dim3 dimBlock(size(warp_layout) * _32{});
  auto grid_shape = make_shape(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  dim3 dimGrid(get<0>(grid_shape) * get<1>(grid_shape));

  thrust::host_vector<tuple<uint, uint>> h_G(size(grid_shape));
  for (uint i = 0; i < size(h_G); ++i) {
    h_G[i] = z_curve(grid_shape, i);
  }

  thrust::device_vector<tuple<uint, uint>> d_G = h_G;

  #ifdef DEBUG
  printf("dimGrid: (%d, %d), dimBlock: (%d, %d)\n",
         dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
  #endif
  oft_device<<<dimGrid, dimBlock, 0, stream>>>
      (d_G, cta_tiler,
       A, A_layout, copyA,
       R, R_layout, copyR, group_size, reconn_sz,
       B, B_layout, copyB,
       C, C_layout, warp_layout, bP);
}

uint64_t check_result(
  int m, int n,
  thrust::host_vector<half>& h_C_result,
  thrust::host_vector<half>& h_C_ref,
  bool verbose = false
) {
  using namespace cute;
  uint64_t error_count = 0;
  auto h_C_layout = make_layout(
    make_shape(m, n),
    LayoutRight{}
  );
  for (int i = 0; i < h_C_result.size(); ++i) {
    float ref_val = static_cast<float>(h_C_ref[i]);
    float result_val = static_cast<float>(h_C_result[i]);
    if (result_val < 0.0f) {
      printf("Result value is negative, which is unexpected. "
             "This might indicate an error in the computation or initialization.");
      return -1;
    }
    if (abs((ref_val - result_val) / ref_val) > 5e-3f) {
      auto coord = h_C_layout.get_hier_coord(i);
      if (verbose) {
        printf("Mismatch at (%d, %d): %f != %f\n", get<0>(coord), get<1>(coord),
               static_cast<float>(h_C_result[i]), static_cast<float>(h_C_ref[i]));
      }
      error_count++;
    }
  }
  return error_count;
}

int main(int argc, char** argv)
{
  using namespace cute;
  argparse::ArgumentParser program(std::string("oft"));
  program.add_argument("-m", "--m")
    .help("Number of rows in matrix A")
    .default_value(4096)
    .action([](const std::string& value) { return std::stoi(value); });
  program.add_argument("-n", "--n")
    .help("Number of columns in matrix B")
    .default_value(4096)
    .action([](const std::string& value) { return std::stoi(value); });
  program.add_argument("-k", "--k")
    .help("Number of columns in matrix A and rows in matrix B")
    .default_value(4096)
    .action([](const std::string& value) { return std::stoi(value); });
  program.add_argument("-t", "--timing_iterations")
    .help("Number of iterations to time")
    .default_value(100)
    .action([](const std::string& value) { return std::stoi(value); });
  program.add_argument("--sparse_speedup")
    .help("the assumed speedup of the sparse tensor core")
    .default_value(2.0)
    .action([](const std::string& value) { return std::stod(value); });
  program.add_argument("-rs", "--random_seed")
    .help("Random seed for the input matrices")
    .default_value(static_cast<int>(std::time(nullptr))) // Use current time as default seed
    .action([](const std::string& value) { return std::stoi(value); });
  
  #ifdef DEBUG
  program.add_argument("--verbose")
    .help("Print verbose output")
    .default_value(0)
    .action([](const std::string& value) { return std::stoi(value); });
  #endif

  #ifdef USE_CUBLAS
  program.add_argument("--correctness")
    .help("Check correctness of the kernel against cublas")
    .default_value(false)
    .implicit_value(true);
  program.add_argument("--correctness_cpu")
    .help("Check correctness of the kernel against CPU reference implementation")
    .default_value(false)
    .implicit_value(true);
  program.add_argument("--01init")
    .help("Initialize input matrices with 0s and 1s instead of random floats")
    .default_value(false)
    .implicit_value(true);
  program.add_argument("--cublas_mode")
    .help("The mode for the cublas kernel, either 'AR_W' or 'A_RW'")
    .default_value(std::string(""))
    .action([](const std::string& value) { return value; });
  #endif
  
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    return 1;
  }
  
  int verbose_level = 0;
  #ifdef DEBUG
  verbose_level = program.get<int>("--verbose");
  #endif

  int m = program.get<int>("--m");
  int n = program.get<int>("--n");
  int k = program.get<int>("--k");
  int timing_iterations = program.get<int>("--timing_iterations");

  #ifdef USE_CUBLAS
  std::string cublas_mode = program.get<std::string>("--cublas_mode");
  bool correctness_check = program.get<bool>("--correctness");
  #endif

  int n_groups = n / GROUP_SIZE;
  int reconn_sz = 8; // hardcoded reconnection size
  int n_blocks = k / reconn_sz; // hardcoded reconnection size

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "Number of groups: " << n_groups << std::endl;
  std::cout << "Number of blocks: " << n_blocks << std::endl;

  thrust::host_vector<half> h_A(m * k);
  thrust::host_vector<half> h_B(n * k);
  thrust::host_vector<half> h_R(n_groups * reconn_sz * k); // 8 is the hardcoded reconnection size
  thrust::host_vector<half> h_C(m * n);
  thrust::device_vector<half> d_C(m * n);

  Tensor h_A_tensor = make_tensor(h_A.data(), make_shape(m, k), LayoutRight{});
  Tensor h_B_tensor = make_tensor(h_B.data(), make_shape(n, k), LayoutRight{});
  Tensor h_R_tensor = make_tensor(h_R.data(), make_shape(n_groups * reconn_sz, k), LayoutRight{});
  Tensor h_R_4d = zipped_divide(
    h_R_tensor,
    make_tile(
      make_layout(reconn_sz), // hardcoded reconnection size
      make_layout(reconn_sz)  // hardcoded reconnection size
    )
  );

  // set a time based random seed
  int random_seed = program.get<int>("--random_seed");
  int zo_init = program.get<bool>("--01init");
  std::srand(static_cast<unsigned int>(random_seed));

  for (int i = 0; i < size<0>(h_A_tensor); ++i) {
    for (int j = 0; j < size<1>(h_A_tensor); ++j) {
      if (zo_init) {
        // Initialize with 0s and 1s
        h_A_tensor(i, j) = static_cast<half>(rand() % 2 * 1.0f);
      } else {
        // Initialize with random floats in the range [0, 1]
        h_A_tensor(i, j) = static_cast<half>( (rand() / double(RAND_MAX)) );
      }
    }
  }

  for (int i = 0; i < size<0>(h_B_tensor); ++i) {
    for (int j = 0; j < size<1>(h_B_tensor); ++j) {
      if (zo_init) {
        // Initialize with 0s and 1s
        h_B_tensor(i, j) = static_cast<half>(rand() % 2 * 1.0f);
      } else {
        // Initialize with random floats in the range [0, 1]
        h_B_tensor(i, j) = static_cast<half>( (rand() / double(RAND_MAX)) );
      }
    }
  }

  // int shuffle_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> shuffle_idx;
  for (int i = 0; i < reconn_sz; ++i) {
    shuffle_idx.push_back(i);
  }
  for (int i = 0; i < size<1>(h_R_4d); ++i) {
    std::shuffle(std::begin(shuffle_idx), std::end(shuffle_idx), std::mt19937{std::random_device{}()});
    for (int j = 0; j < reconn_sz; ++j) {
      // shuffle the indices to create a more complex pattern
      h_R_4d(make_coord(j, shuffle_idx[j]), i) = static_cast<half>(1.0f);
    }
  }

  #ifdef DEBUG
  if (verbose_level >= 2) {
    printf("A:\n");
    for (int i = 0; i < size<0>(h_A_tensor); ++i) {
      for (int j = 0; j < size<1>(h_A_tensor); ++j) {
        printf("%6.3f ", static_cast<float>(h_A_tensor(i, j)));
      }
      printf("\n");
    }

    printf("R:\n");
    for (int i = 0; i < size<0>(h_R_tensor); ++i) {
      for (int j = 0; j < size<1>(h_R_tensor); ++j) {
        printf("%6.3f ", static_cast<float>(h_R_tensor(i, j)));
      }
      printf("\n");
    }

    printf("B:\n");
    for (int i = 0; i < size<0>(h_B_tensor); ++i) {
      for (int j = 0; j < size<1>(h_B_tensor); ++j) {
        printf("%6.3f ", static_cast<float>(h_B_tensor(i, j)));
      }
      printf("\n");
    }
  }
  #endif // DEBUG

  thrust::device_vector<half> d_A = h_A;
  thrust::device_vector<half> d_B = h_B;
  thrust::device_vector<half> d_R = h_R;
  thrust::fill(thrust::device, d_C.begin(), d_C.end(), static_cast<half>(-1.0f));

  std::map<std::string, std::function<void()>> test_funcs;
  test_funcs["oft_tn"] = [&]() {
    oft_tn(m, n, k,
      d_A.data().get(), k,
      d_B.data().get(), k,
      d_R.data().get(), k,
      d_C.data().get(), n);
    CUTE_CHECK_LAST();
  };

  test_funcs["cpu_oft_tn"] = [&]() {
    cpu_oft_tn(
      h_A, h_R, h_B, h_C,
      m, GROUP_SIZE, n_groups, k, reconn_sz
    );
  };

  auto test_func = test_funcs["oft_tn"]; // default to the oft kernel

  #ifdef USE_CUBLAS
  cublasHandle_t cublas_handle;
  getCublasTensorOpHandle(&cublas_handle);
  test_funcs["cublas_AR_W"] = [&]() {
    cublas_oft(d_A, d_R, d_B, d_C, m, GROUP_SIZE, n_groups, k, reconn_sz, &cublas_handle, false); // AR_W
    GEMM_CHECK_CUDA(cudaDeviceSynchronize());
  };
  test_funcs["cublas_A_RW"] = [&]() {
    cublas_oft(d_A, d_R, d_B, d_C, m, GROUP_SIZE, n_groups, k, reconn_sz, &cublas_handle, true);  // A_RW
    GEMM_CHECK_CUDA(cudaDeviceSynchronize());
  };

  if (cublas_mode == "AR_W") {
    test_func = test_funcs["cublas_AR_W"];
  } else if (cublas_mode == "A_RW") {
    test_func = test_funcs["cublas_A_RW"];
  }

  if (correctness_check) {
    // check the correctness of the oft kernel against cublas
    thrust::fill(thrust::device, d_C.begin(), d_C.end(), static_cast<half>(-1.0f));
    test_func(); // run the oft kernel
    thrust::host_vector<half> h_C_result = d_C;

    // compute the two versions of reference results using cublas
    printf("Checking against AR_W reference result...\n");
    thrust::fill(thrust::device, d_C.begin(), d_C.end(), static_cast<half>(-1.0f));
    test_funcs["cublas_AR_W"]();
    thrust::host_vector<half> h_C_ref_AR_W = d_C;
    uint64_t error_count_AR_W = check_result(m, n, h_C_result, h_C_ref_AR_W, verbose_level >= 1);
    if(error_count_AR_W == 0) {
      printf("oft kernel result matches AR_W reference result!\n");
    } else {
      printf("oft kernel result does NOT match AR_W reference result for %lu/%lu entries\n", error_count_AR_W, h_C_result.size());
      // return 1;
    }

    printf("Checking against A_RW reference result...\n");
    thrust::fill(thrust::device, d_C.begin(), d_C.end(), static_cast<half>(-1.0f));
    test_funcs["cublas_A_RW"]();
    thrust::host_vector<half> h_C_ref_A_RW = d_C;
    uint64_t error_count_A_RW = check_result(m, n, h_C_result, h_C_ref_A_RW, verbose_level >= 1);
    if(error_count_A_RW == 0) {
      printf("oft kernel result matches A_RW reference result!\n");
    } else {
      printf("oft kernel result does NOT match A_RW reference result for %lu/%lu entries\n", error_count_A_RW, h_C_result.size());
      // return 1;
    }

    if (program.get<bool>("--correctness_cpu")) {
      // check the correctness of the oft kernel against CPU reference
      printf("Checking against CPU reference result...\n");
      thrust::fill(h_C.begin(), h_C.end(), static_cast<half>(-1.0f));
      test_funcs["cpu_oft_tn"](); // compute the CPU reference result
      uint64_t error_count_cpu = check_result(m, n, h_C_result, h_C, verbose_level >= 1);
      if(error_count_cpu == 0) {
        printf("oft kernel result matches CPU reference result!\n");
      } else {
        printf("oft kernel result does NOT match CPU reference result for %lu/%lu entries\n", error_count_cpu, h_C_result.size());
        // return 1;
      }
    }
  }
  #endif // USE_CUBLAS

  if (timing_iterations <= 0) {
    return 0;
  }

  double base_t_flops = (double)m*n*k*2e-12; // 2 flops per multiply-add
  printf("Base TFLOPS: %.5f\n", base_t_flops);
  double additional_t_AR_W = (double)n_groups*m*k*reconn_sz*2e-12; // 2 flops per multiply-add
  double additional_t_A_RW = (double)n*k*reconn_sz*2e-12; // 2 flops per multiply-add
  double t_flops_A_RW = base_t_flops + additional_t_A_RW;
  double t_flops_AR_W = base_t_flops + additional_t_AR_W;
  double t_flops_AR_W_sparse = base_t_flops * program.get<double>("--sparse_speedup") + additional_t_AR_W;
  printf("Total TFLOPS (AR_W): %.5f, (AR_W_sparse): %.5f, (A_RW): %.5f\n",
         t_flops_AR_W, t_flops_AR_W_sparse, t_flops_A_RW);
  test_func(); // run the test function once to ensure everything is set up correctly
  // Timing iterations
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    test_func();
  }
  double time = timer.seconds() / timing_iterations;
  double theoretical_speedup = t_flops_AR_W_sparse / t_flops_AR_W;
  printf("Theoretical speedup: %.2f\n", theoretical_speedup);
  printf("TFLOPS/s (AR_W): %.2f, (AR_W_sparse): %.2f, (A_RW): %.2f, Time: %.3f ms\n",
         t_flops_AR_W / time, t_flops_AR_W_sparse / time, t_flops_A_RW / time, time * 1000.0);
  return 0;
}

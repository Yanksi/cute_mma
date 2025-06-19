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
#include "cute_sparse_simple.hpp"
#ifdef USE_CUBLAS
#include "cublas_gemm.hpp"
#endif

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

#include <cutlass/util/print_error.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/helper_cuda.hpp>

#include <argparse/argparse.hpp>

#define mmax(a,b) ((a) > (b) ? (a) : (b))
#define mmin(a,b) ((a) < (b) ? (a) : (b))

#define GROUP_SIZE 256

namespace cute {
  template <typename TO, typename TR>
  struct Params {
    static_assert(sizeof(TO) == 0, "This struct should not be used");
  };

  template <>
  struct Params <half, half> {
    static const unsigned int bM = 256;
    static const unsigned int bN_group = 1;
    static const unsigned int bK_block = 2;
    static const unsigned int bP = 2;
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
  constexpr auto get_cp_width = []() {
    if constexpr (block_tiling) {
      return vec_width;
    } else {
      return elements_per_thread;
    }
  };
  constexpr int cp_width = get_cp_width();
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
  auto group_size = Int<GROUP_SIZE>{}; // Group size for the block tiling
  auto reconn_sz = _8{}; // hardcoded for now, can be made dynamic later
  auto bM = Int<CurrParams::bM>{};
  auto bN_group = Int<CurrParams::bN_group>{};
  auto bN = bN_group * group_size;
  auto bK_block = Int<CurrParams::bK_block>{};
  auto bK = bK_block * reconn_sz;
  auto blocks_tiler = make_shape(bM, bN_group, bK_block);                   // (BLK_M, BLK_N, BLK_K)
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
  TiledCopy copyR = cp_layout<uint128_t, half, true, CurrParams::block_tiling_copy>(bN, bK, size(warp_layout) * _32{});

  dim3 dimBlock(size(warp_layout) * _32{});
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  oft_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, blocks_tiler,
       A, A_layout, copyA,
       R, R_layout, copyR, group_size, reconn_sz,
       B, B_layout, copyB,
       C, C_layout, warp_layout, Int<CurrParams::bP>{});
}

int main(int argc, char** argv)
{
  argparse::ArgumentParser program(std::string("oft"));
  program.add_argument("-m", "--m")
    .help("Number of rows in matrix A")
    .default_value(8192)
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
  
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    return 1;
  }
  
  int m = program.get<int>("--m");
  int n = program.get<int>("--n");
  int k = program.get<int>("--k");
  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;

  int n_groups = n / GROUP_SIZE;

  thrust::host_vector<half> h_A(m * k);
  thrust::host_vector<half> h_B(n * k);
  thrust::host_vector<half> h_R(n_groups * 8 * k); // 8 is the hardcoded reconnection size
  thrust::host_vector<half> h_C(m * n);

  // initialize matrix with positive values to avoid cancellation errors
  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<half>( (rand() / double(RAND_MAX)) );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<half>( (rand() / double(RAND_MAX)) );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<half>(-1);
  // initialize Rs to be identity matrices for easier testing
  for (int i = 0; i < n_groups; ++i) {
    for (int j = 0; j < k / 8; ++j) {
      for (int u = 0; u < 8; ++u) {
        for (int v = 0; v < 8; ++v) {
          int curr_idx = (i * 8 + u) * k + (j * 8 + v);
          if (u == v) {
            h_R[curr_idx] = static_cast<half>(1.0);
          } else {
            h_R[curr_idx] = static_cast<half>(0.0);
          }
        }
      }
    }
  }

  thrust::device_vector<half> d_A = h_A;
  thrust::device_vector<half> d_B = h_B;
  thrust::device_vector<half> d_C = h_C;
  thrust::device_vector<half> d_R = h_R;
  oft_tn(m, n, k,
        d_A.data().get(), k,
        d_B.data().get(), k,
        d_R.data().get(), k,
        d_C.data().get(), n);
  CUTE_CHECK_LAST();
  
  // #ifdef USE_CUBLAS
  // program.add_argument("--cublas")
  //   .help("Benchmark cuBLAS")
  //   .default_value(false)
  //   .implicit_value(true);
  // #endif

  // try {
  //   program.parse_args(argc, argv);
  // } catch (const std::runtime_error& err) {
  //   std::cout << err.what() << std::endl;
  //   std::cout << program;
  //   return 1;
  // }

  // int m = program.get<int>("--m");
  // int n = program.get<int>("--n");
  // int k = program.get<int>("--k");
  // char transA = program.get<char>("--transA");
  // char transB = program.get<char>("--transB");
  // int timing_iterations = program.get<int>("--timing_iterations");

  // #ifdef USE_CUBLAS
  // bool cublas = program.get<bool>("--cublas");
  // #endif
  
  // cudaDeviceProp props;
  // cudaError_t error = cudaGetDeviceProperties(&props, 0);
  // if (error != cudaSuccess) {
  //   std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
  //   return -1;
  // }

  // if (props.major < 8) { 
  //   std::cout << "This example requires an Ampere GPU or newer (CC >= 80)" << std::endl;
  //   // Return 0 so tests pass if run on unsupported architectures or CUDA Toolkits.
  //   return 0;
  // }

  // std::cout << "M = " << m << std::endl;
  // std::cout << "N = " << n << std::endl;
  // std::cout << "K = " << k << std::endl;
  // std::cout << "C = A^" << transA << " B^" << transB << std::endl;

  // thrust::host_vector<CUTE_MMA_DTYPE_O> h_A(m*k);
  // thrust::host_vector<CUTE_MMA_DTYPE_O> h_B(n*k);
  // thrust::host_vector<CUTE_MMA_DTYPE_R> h_C(m*n);

  // // initialize matrix with positive values to avoid cancellation errors
  // for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<CUTE_MMA_DTYPE_O>( (rand() / double(RAND_MAX)) );
  // for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<CUTE_MMA_DTYPE_O>( (rand() / double(RAND_MAX)) );
  // for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<CUTE_MMA_DTYPE_R>(-1);

  // thrust::device_vector<CUTE_MMA_DTYPE_O> d_A = h_A;
  // thrust::device_vector<CUTE_MMA_DTYPE_O> d_B = h_B;
  // thrust::device_vector<CUTE_MMA_DTYPE_R> d_C = h_C;

  // double gflops = (2.0*m*n*k) * 1e-9;

  // GPU_Clock timer;

  // int ldA = 0, ldB = 0, ldC = m;

  // if (toupper(transA) == 'N') {
  //   ldA = m;
  // } else if (toupper(transA) == 'T') {
  //   ldA = k;
  // } else {
  //   assert(false);
  // }

  // if (toupper(transB) == 'N') {
  //   ldB = k;
  // } else if (toupper(transB) == 'T') {
  //   ldB = n;
  // } else {
  //   assert(false);
  // }

  // std::function<void()> test_func = [&]() {
  //   gemm(transA, transB, m, n, k,
  //        d_A.data().get(), ldA,
  //        d_B.data().get(), ldB,
  //        d_C.data().get(), ldC);
  // };

  // #ifdef USE_CUBLAS
  // cublasHandle_t handle;
  // if (cublas) {
  //   getCublasTensorOpHandle(&handle);
  //   test_func = [&]() {
  //     gemm_cublas(transA, transB, m, n, k,
  //                 d_A.data().get(), ldA,
  //                 d_B.data().get(), ldB,
  //                 d_C.data().get(), ldC, &handle);
  //     GEMM_CHECK_CUDA(cudaDeviceSynchronize());
  //   };
  // }
  // #endif

  // // Run once
  // #ifdef DEBUG
  // d_C = h_C;
  // test_func();
  // CUTE_CHECK_LAST();
  // thrust::host_vector<CUTE_MMA_DTYPE_R> kernel_result = d_C;
  // double* ref_C = new double[m*n];
  // for (int i = 0; i < m; ++i) {
  //   for (int j = 0; j < n; ++j) {
  //     double sum = 0;
  //     for (int l = 0; l < k; ++l) {
  //       double a = (transA == 'T' || transA == 't') ? h_A[i*ldA + l] : h_A[i + l*ldA];
  //       double b = (transB == 'T' || transB == 't') ? h_B[l*ldB + j] : h_B[l + j*ldB];
  //       sum += a * b;
  //     }
  //     ref_C[j*m + i] = sum;
  //   }
  // }
  // double max_error = 0;
  // for (int i = 0; i < m*n; ++i) {
  //   double cr = (double)kernel_result[i];
  //   double rr = ref_C[i];
  //   max_error = std::max(max_error, std::abs((cr - rr) / rr));
  // }
  // printf("Max error: %e\n", max_error);
  
  // #endif
  // d_C = h_C;
  // test_func(); // warmup
  // CUTE_CHECK_LAST();

  // // Timing iterations
  // timer.start();
  // for (int i = 0; i < timing_iterations; ++i) {
  //   test_func();
  // }
  // double cute_time = timer.seconds() / timing_iterations;
  // CUTE_CHECK_LAST();
  // printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

  return 0;
}

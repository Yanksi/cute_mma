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
#include "gemm_tc.hpp"
#include "gemm_config.hpp"
#include "cublas_gemm.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

#include <cutlass/util/print_error.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/helper_cuda.hpp>

#include <argparse/argparse.hpp>

#define mmax(a,b) ((a) > (b) ? (a) : (b))

template <typename copy_as_t, typename ele_t, bool k_major, bool block_tiling,
  typename _BM, typename _BK, typename _N_Threads>
constexpr auto cp_layout(_BM bm, _BK bk, _N_Threads total_threads) {
  constexpr int vec_width = sizeof(copy_as_t) / sizeof(ele_t);
  constexpr int total_elements = bm * bk;
  constexpr int elements_per_thread = total_elements / total_threads;
  CUTE_STATIC_ASSERT(total_elements % total_threads == 0);
  CUTE_STATIC_ASSERT(elements_per_thread % vec_width == 0);
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
    // As it not really possible to have copy width greater than bm, we can assert it
    CUTE_STATIC_ASSERT(cp_width < bm);
    constexpr int threads_along_m = bm / cp_width;
    constexpr int threads_along_k = total_threads / threads_along_m;
    return make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<copy_as_t>, ele_t>{},
                           Layout<Shape<Int<threads_along_m>, Int<threads_along_k>>>{},
                           Layout<Shape<Int<cp_width>, _1>>{});
  }
}

// Setup params for a NT GEMM
template <typename TA>
void gemm_nt(int m, int n, int k,
        TA const* A, int ldA,
        TA const* B, int ldB,
        TA      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<ParamNT::bM>{};
  auto bN = Int<ParamNT::bN>{};
  auto bK = Int<ParamNT::bK>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<ParamNT::bP>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK, bP));             // (m,k,p) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK, bP));             // (n,k,p) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)

  TiledMMA mmaC = make_tiled_mma(ParamNT::mma_atom{}, ParamNT::warp_layout{});  // 16x8x8 TiledMMA
  CUTE_STATIC_ASSERT(bM % tile_size<0>(mmaC) == 0);
  CUTE_STATIC_ASSERT(bN % tile_size<1>(mmaC) == 0);
  CUTE_STATIC_ASSERT(bK % tile_size<2>(mmaC) == 0);

  TiledCopy copyA = cp_layout<uint128_t, TA, false, ParamNT::block_tiling_copy>(bM, bK, size(mmaC));
  TiledCopy copyB = cp_layout<uint128_t, TA, false, ParamNT::block_tiling_copy>(bN, bK, size(mmaC));
  

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA,
       B, dB, sB, copyB,
       C, dC, sC, mmaC);
}

// // Setup params for a TN GEMM, K-Major inputs
template <typename TA>
void gemm_tn(int m, int n, int k,
        TA const* A, int ldA,
        TA const* B, int ldB,
        TA      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(ldC, Int<1>{});                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<ParamTN::bM>{};
  auto bN = Int<ParamTN::bN>{};
  auto bK = Int<ParamTN::bK>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<ParamTN::bP>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA_atom = make_layout(make_shape (bM, bK), LayoutRight{}); // (m,k) -> smem_idx; padded k-major
  auto sB_atom = make_layout(make_shape (bN, bK), LayoutRight{}); // (n,k) -> smem_idx; padded k-major
  auto sA = tile_to_shape(sA_atom, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(sB_atom, make_shape(bN, bK, bP));
  auto sC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx

  
  TiledMMA mmaC = make_tiled_mma(ParamTN::mma_atom{}, ParamTN::warp_layout{});  // 16x8x8 TiledMMA
  CUTE_STATIC_ASSERT(bM % tile_size<0>(mmaC) == 0);
  CUTE_STATIC_ASSERT(bN % tile_size<1>(mmaC) == 0);
  CUTE_STATIC_ASSERT(bK % tile_size<2>(mmaC) == 0);
  
  TiledCopy copyA = cp_layout<uint128_t, TA, true, ParamTN::block_tiling_copy>(bM, bK, size(mmaC));
  TiledCopy copyB = cp_layout<uint128_t, TA, true, ParamTN::block_tiling_copy>(bN, bK, size(mmaC));

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA,
       B, dB, sB, copyB,
       C, dC, sC, mmaC);
}

template <typename TA>
void gemm(char transA, char transB, int m, int n, int k,
     TA const* A, int ldA,
     TA const* B, int ldB,
     TA      * C, int ldC,
     cudaStream_t stream = 0)
{
  #ifdef LAYOUT_NT
  if (transA == 'N' && transB == 'T') {
    return gemm_nt(m, n, k, A, ldA, B, ldB, C, ldC, stream);
  }
  #endif
  #ifdef LAYOUT_TN
  if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, A, ldA, B, ldB, C, ldC, stream);
  }
  #endif
  assert(false && "Not implemented");
}


int main(int argc, char** argv)
{
  std::string running_type = getTypeName<DTYPE>();
  argparse::ArgumentParser program(std::string("gemm_") + running_type);
  program.add_argument("-m", "--m")
    .help("Number of rows in matrix A")
    .default_value(8192)
    .action([](const std::string& value) { return std::stoi(value); });
  program.add_argument("-n", "--n")
    .help("Number of columns in matrix B")
    .default_value(8192)
    .action([](const std::string& value) { return std::stoi(value); });
  program.add_argument("-k", "--k")
    .help("Number of columns in matrix A and rows in matrix B")
    .default_value(4096)
    .action([](const std::string& value) { return std::stoi(value); });
  program.add_argument("--transA")
    .help("Transpose matrix A")
    .default_value('T')
    .action([](const std::string& value) { return value[0]; });
  program.add_argument("--transB")
    .help("Transpose matrix B")
    .default_value('N')
    .action([](const std::string& value) { return value[0]; });
  program.add_argument("--timing_iterations")
    .help("Number of iterations to time")
    .default_value(100)
    .action([](const std::string& value) { return std::stoi(value); });
  program.add_argument("--correctness")
    .help("Check correctness")
    .default_value(false)
    .implicit_value(true);
  
  #ifdef USE_CUBLAS
  program.add_argument("--cublas")
    .help("Use cuBLAS")
    .default_value(false)
    .implicit_value(true);
  #endif

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
  char transA = program.get<char>("--transA");
  char transB = program.get<char>("--transB");
  int timing_iterations = program.get<int>("--timing_iterations");
  bool correctness = program.get<bool>("--correctness");

  #ifdef USE_CUBLAS
  bool cublas = program.get<bool>("--cublas");
  #endif
  
  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major < 8) { 
    std::cout << "This example requires an Ampere GPU or newer (CC >= 80)" << std::endl;
    // Return 0 so tests pass if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }

  #ifdef USE_CUBLAS
  cublasHandle_t handle;
  getCublasTensorOpHandle(&handle);
  #endif

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "C = A^" << transA << " B^" << transB << std::endl;

  thrust::host_vector<DTYPE> h_A(m*k);
  thrust::host_vector<DTYPE> h_B(n*k);
  thrust::host_vector<DTYPE> h_C(m*n);
  double* ref_C = (correctness) ? new double[m*n] : nullptr;

  // initialize matrix with positive values to avoid cancellation errors
  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<DTYPE>( (rand() / double(RAND_MAX)) );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<DTYPE>( (rand() / double(RAND_MAX)) );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<DTYPE>(-1);

  thrust::device_vector<DTYPE> d_A = h_A;
  thrust::device_vector<DTYPE> d_B = h_B;
  thrust::device_vector<DTYPE> d_C = h_C;

  double gflops = (2.0*m*n*k) * 1e-9;

  GPU_Clock timer;

  int ldA = 0, ldB = 0, ldC = m;

  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false);
  }

  // A simple CPU reference GEMM
  if (correctness) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        double sum = 0;
        for (int l = 0; l < k; ++l) {
          double a = (transA == 'T') ? h_A[i*ldA + l] : h_A[i + l*ldA];
          double b = (transB == 'T') ? h_B[l*ldB + j] : h_B[l + j*ldB];
          sum += a * b;
        }
        ref_C[j*n + i] = sum;
      }
    }
  }

  std::function<void()> test_func = [&]() {
    gemm(transA, transB, m, n, k,
         d_A.data().get(), ldA,
         d_B.data().get(), ldB,
         d_C.data().get(), ldC);
  };

  #ifdef USE_CUBLAS
  if (cublas) {
    test_func = [&]() {
      gemm_cublas(transA, transB, m, n, k,
                  d_A.data().get(), ldA,
                  d_B.data().get(), ldB,
                  d_C.data().get(), ldC, &handle);
      GEMM_CHECK_CUDA(cudaDeviceSynchronize());
    };
  }
  #endif

  // Run once
  d_C = h_C;
  test_func();
  CUTE_CHECK_LAST();
  thrust::host_vector<DTYPE> kernel_result = d_C;

  #ifdef USE_CUBLAS
  if (correctness) {
    double max_error = 0;
    for (int i = 0; i < m*n; ++i) {
      double cr = static_cast<double>(kernel_result[i]);
      double rr = ref_C[i];
      max_error = std::max(max_error, std::abs((cr - rr) / rr));
    }
    printf("Max error: %e\n", max_error);
  }
  #endif

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    test_func();
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

  return 0;
}
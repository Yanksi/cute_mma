#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <random>
// #include "cute_oft_simple.hpp"
#include "cute_oft_coop_pc.hpp"
#ifdef USE_CUBLAS
#include "cublas_oft.hpp"
#endif
#include "cpu_oft.hpp"
#include "z_curve.hpp"
#include <omp.h>

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

uint64_t check_result(
  int m, int n,
  thrust::host_vector<half>& h_C_result,
  thrust::host_vector<half>& h_C_ref,
  float * maximum_error,
  float error_threshold = 5e-3,
  bool verbose = false
) {
  using namespace cute;
  uint64_t error_count = 0;
  float _max_error = 0.0f;
  auto h_C_layout = make_layout(
    make_shape(m, n),
    LayoutRight{}
  );
  bool *correctness_mat = new bool[m * n];

  #pragma omp parallel for reduction(+:error_count) reduction(max:_max_error)
  for (int i = 0; i < h_C_result.size(); ++i) {
    float ref_val = static_cast<float>(h_C_ref[i]);
    float result_val = static_cast<float>(h_C_result[i]);
    float error = abs((ref_val - result_val) / ref_val);
    _max_error = max(_max_error, error);
    if (error > error_threshold) {
      auto coord = h_C_layout.get_hier_coord(i);
      correctness_mat[i] = false;
      error_count++;
    } else {
      correctness_mat[i] = true;
    }
  }
  
  *maximum_error = _max_error;
  if (verbose) {
    Tensor C_result = make_tensor(h_C_result.data(), h_C_layout);
    printf("Result Mat:\n");
    print_tensor(C_result); printf("\n");
    Tensor C_ref = make_tensor(h_C_ref.data(), h_C_layout);
    printf("Reference Mat:\n");
    print_tensor(C_ref); printf("\n");
    Tensor correctness_tensor = make_tensor(correctness_mat, h_C_layout);
    printf("Correctness Mat:\n");
    print_tensor(correctness_tensor); printf("\n");
  }
  delete[] correctness_mat;
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
  program.add_argument("-w", "--warmup_iterations")
    .help("Number of warmup iterations to run before timing")
    .default_value(10)
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
  int warmup_iterations = program.get<int>("--warmup_iterations");
  int timing_iterations = program.get<int>("--timing_iterations");

  #ifdef USE_CUBLAS
  std::string cublas_mode = program.get<std::string>("--cublas_mode");
  bool correctness_check = program.get<bool>("--correctness");
  #endif

  int group_size = CurrKernelParams::group_size; // group size for the block tiling
  int reconn_sz = CurrKernelParams::reconn_sz; // hardcoded reconnection size

  int n_groups = n / group_size; // number of groups for the block tiling
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
  Tensor h_R_4d = flat_divide(
    h_R_tensor,
    make_tile(
      make_layout(reconn_sz), // hardcoded reconnection size
      make_layout(reconn_sz)  // hardcoded reconnection size
    )
  );

  // set a time based random seed
  int random_seed = program.get<int>("--random_seed");
  int zo_init = program.get<bool>("--01init");

  #pragma omp parallel
  {
    // Each thread gets its own random number generator seeded differently
    std::mt19937 generator(random_seed + omp_get_thread_num());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    #pragma omp for
    for (int i = 0; i < h_A.size(); ++i) {
      if (zo_init) {
        // Initialize with 0s and 1s
        h_A[i] = static_cast<half>(static_cast<int>(distribution(generator) * 2));
      } else {
        // Initialize with random floats in the range [0, 1]
        h_A[i] = static_cast<half>(distribution(generator));
      }
    }

    #pragma omp for
    for (int i = 0; i < h_B.size(); ++i) {
      if (zo_init) {
        // Initialize with 0s and 1s
        h_B[i] = static_cast<half>(static_cast<int>(distribution(generator) * 2));
      } else {
        // Initialize with random floats in the range [0, 1]
        h_B[i] = static_cast<half>(distribution(generator));
      }
    }

    std::vector<int> shuffle_idx;
    for (int i = 0; i < reconn_sz; ++i) {
      shuffle_idx.push_back(i);
    }
    #pragma omp for
    for (int g = 0; g < size<1>(h_R_4d); ++g) {
      for (int k = 0; k < size<2>(h_R_4d); ++k) {
        std::shuffle(std::begin(shuffle_idx), std::end(shuffle_idx), generator);
        for (int j = 0; j < reconn_sz; ++j) {
          // shuffle the indices to create a more complex pattern
          h_R_4d(j, shuffle_idx[j], g, k) = static_cast<half>(1.0f);
        }
      }
    }
  }

  #ifdef DEBUG
  if (verbose_level >= 2) {
    printf("A:\n");
    print_tensor(h_A_tensor);print("\n");

    printf("R:\n");
    print_tensor(h_R_tensor);print("\n");

    printf("B:\n");
    print_tensor(h_B_tensor);print("\n");
  }
  #endif // DEBUG

  thrust::device_vector<half> d_A = h_A;
  thrust::device_vector<half> d_B = h_B;
  thrust::device_vector<half> d_R = h_R;
  thrust::fill(thrust::device, d_C.begin(), d_C.end(), static_cast<half>(-1.0f));

  std::map<std::string, std::function<void()>> test_funcs;
  test_funcs["oft_tn"] = [&]() {
    oft_tn<CurrKernelParams>(m, n, k,
      d_A.data().get(), k,
      d_B.data().get(), k,
      d_R.data().get(), k,
      d_C.data().get(), n);
    CUTE_CHECK_LAST();
  };

  test_funcs["cpu_oft_tn"] = [&]() {
    cpu_oft_tn(
      h_A, h_R, h_B, h_C,
      m, group_size, n_groups, k, reconn_sz
    );
  };

  auto test_func = test_funcs["oft_tn"]; // default to the oft kernel

  #ifdef USE_CUBLAS
  cublasHandle_t cublas_handle;
  getCublasTensorOpHandle(&cublas_handle);
  test_funcs["cublas_AR_W"] = [&]() {
    cublas_oft(d_A, d_R, d_B, d_C, m, group_size, n_groups, k, reconn_sz, &cublas_handle, false); // AR_W
    GEMM_CHECK_CUDA(cudaDeviceSynchronize());
  };
  test_funcs["cublas_A_RW"] = [&]() {
    cublas_oft(d_A, d_R, d_B, d_C, m, group_size, n_groups, k, reconn_sz, &cublas_handle, true);  // A_RW
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
    float maximum_error_AR_W = 0.0f;
    uint64_t error_count_AR_W = check_result(m, n, h_C_result, h_C_ref_AR_W, &maximum_error_AR_W, 5e-3, verbose_level >= 1);
    float error_rate_AR_W = static_cast<float>(error_count_AR_W) / h_C_result.size();
    if(error_count_AR_W == 0) {
      printf("oft kernel result matches AR_W reference result!\n");
    } else {
      printf("oft kernel result does NOT match AR_W reference result for %lu/%lu entries\n", error_count_AR_W, h_C_result.size());
      printf("Error rate: %.2f%%\n", error_rate_AR_W * 100.0);
      printf("Maximum error: %.5f\n", maximum_error_AR_W);
      // return 1;
    }
    printf("\n\n");

    printf("Checking against A_RW reference result...\n");
    thrust::fill(thrust::device, d_C.begin(), d_C.end(), static_cast<half>(-1.0f));
    test_funcs["cublas_A_RW"]();
    thrust::host_vector<half> h_C_ref_A_RW = d_C;
    float maximum_error_A_RW = 0.0f;
    uint64_t error_count_A_RW = check_result(m, n, h_C_result, h_C_ref_A_RW, &maximum_error_A_RW, 5e-3, verbose_level >= 1);
    float error_rate_A_RW = static_cast<float>(error_count_A_RW) / h_C_result.size();
    if(error_count_A_RW == 0) {
      printf("oft kernel result matches A_RW reference result!\n");
    } else {
      printf("oft kernel result does NOT match A_RW reference result for %lu/%lu entries\n", error_count_A_RW, h_C_result.size());
      printf("Error rate: %.2f%%\n", error_rate_A_RW * 100.0);
      printf("Maximum error: %.5f\n", maximum_error_A_RW);
      // return 1;
    }
    printf("\n\n");

    if (program.get<bool>("--correctness_cpu")) {
      // check the correctness of the oft kernel against CPU reference
      printf("Checking against CPU reference result...\n");
      thrust::fill(h_C.begin(), h_C.end(), static_cast<half>(-1.0f));
      test_funcs["cpu_oft_tn"](); // compute the CPU reference result
      float maximum_error_cpu = 0.0f;
      uint64_t error_count_cpu = check_result(m, n, h_C_result, h_C, &maximum_error_cpu, 5e-3, verbose_level >= 1);
      if(error_count_cpu == 0) {
        printf("oft kernel result matches CPU reference result!\n");
      } else {
        printf("oft kernel result does NOT match CPU reference result for %lu/%lu entries\n", error_count_cpu, h_C_result.size());
        printf("Error rate: %.2f%%\n", static_cast<float>(error_count_cpu) / h_C_result.size() * 100.0);
        printf("Maximum error: %.5f\n", maximum_error_cpu);
        // return 1;
      }
      printf("\n\n");
    }
  }
  #endif // USE_CUBLAS

  for (int i = 0; i < warmup_iterations; ++i) {
    test_func(); // run the warmup iterations
  }

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

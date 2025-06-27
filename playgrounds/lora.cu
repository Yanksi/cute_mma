#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <argparse/argparse.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cutlass/util/GPU_Clock.hpp>

#define GEMM_LIKELY(x) __builtin_expect(!!(x), 1)
#define GEMM_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define GEMM_CHECK_CUBLAS(call)                                                                   \
    do {                                                                                          \
        cublasStatus_t status = call;                                                             \
        if (GEMM_UNLIKELY(status != CUBLAS_STATUS_SUCCESS)) {                                     \
            throw std::runtime_error("CUBLAS call failed with status " + std::to_string(status)); \
        }                                                                                         \
    } while (0)

#define GEMM_CHECK_CUDA(call)                                                                   \
    do {                                                                                        \
        cudaError_t status = call;                                                              \
        if (GEMM_UNLIKELY(status != cudaSuccess)) {                                             \
            throw std::runtime_error("CUDA call failed with status " + std::to_string(status)); \
        }                                                                                       \
    } while (0)

void getCublasTensorOpHandle(cublasHandle_t* handle) {
    GEMM_CHECK_CUBLAS(cublasCreate(handle));
    GEMM_CHECK_CUBLAS(cublasSetMathMode(*handle, CUBLAS_TF32_TENSOR_OP_MATH));
}

void lora_cublas(
    cublasHandle_t& handle,
    const thrust::device_vector<half>& d_A,
    const thrust::device_vector<half>& d_B,
    const thrust::device_vector<half>& d_L1,
    const thrust::device_vector<half>& d_L2,
    thrust::device_vector<half>& d_C,
    int m, int n, int k, int r, bool lora_only = false) {
    
    half alpha = static_cast<half>(1.0f);
    half beta1 = static_cast<half>(0.0f);
    half beta2 = static_cast<half>(1.0f);

    thrust::device_vector<half> d_temp(m * r, 0.0f);
    // Perform the first multiplication with L1
    GEMM_CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, r, m, k,
                                  &alpha, d_L1.data().get(), k,
                                  d_A.data().get(), k,
                                  &beta1, d_temp.data().get(), r));
    // Perform the second multiplication with L2
    GEMM_CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, r,
                                  &alpha, d_L2.data().get(), r,
                                  d_temp.data().get(), r,
                                  &beta1, d_C.data().get(), n));
    if (!lora_only) {
        // Perform the matrix multiplication using cublasHgemm
        GEMM_CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                    &alpha, d_B.data().get(), k,
                                    d_A.data().get(), k,
                                    &beta2, d_C.data().get(), n));
    }
}

// Benchmarking the cublas hgemm function to get the speed of light of GPU
int main(int argc, char** argv) {
    argparse::ArgumentParser program("cublas_gemm");
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
    program.add_argument("-g", "--group_size")
        .help("Groups size of OFT")
        .default_value(256)
        .action([](const std::string& value) { return std::stoi(value); });
    program.add_argument("-r", "--reconn_sz")
        .help("Reconnection size of OFT")
        .default_value(8)
        .action([](const std::string& value) { return std::stoi(value); });
    program.add_argument("-rs", "--random_seed")
        .help("Random seed for the input matrices")
        .default_value(static_cast<int>(std::time(nullptr))) // Use current time as default seed
        .action([](const std::string& value) { return std::stoi(value); });
    program.add_argument("-l", "--lora_only")
        .help("Use Lora only, no cublas hgemm")
        .default_value(false)
        .implicit_value(true);
    

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
    int timing_iterations = program.get<int>("--timing_iterations");
    int group_size = program.get<int>("--group_size");
    int reconn_sz = program.get<int>("--reconn_sz");
    bool lora_only = program.get<bool>("--lora_only");

    double addition_parameter_ratio = (double)reconn_sz / (double)group_size;
    int lora_rank = (int)(addition_parameter_ratio * (n * k) / (n + k));

    std::cout << "M = " << m << std::endl;
    std::cout << "N = " << n << std::endl;
    std::cout << "K = " << k << std::endl;
    std::cout << "Adding parameter ratio = " << addition_parameter_ratio << std::endl;
    std::cout << "Lora rank = " << lora_rank << std::endl;
    std::cout << "Lora only = " << (lora_only ? "true" : "false") << std::endl;

    thrust::host_vector<half> h_A(m * k);
    thrust::host_vector<half> h_B(n * k);
    thrust::host_vector<half> h_L1(k * lora_rank);
    thrust::host_vector<half> h_L2(n * lora_rank);

    cublasHandle_t handle;
    getCublasTensorOpHandle(&handle);

    int random_seed = program.get<int>("--random_seed");
    std::srand(static_cast<unsigned int>(random_seed));

    // Initialize matrices A and B with random values
    for (int i = 0; i < h_A.size(); ++i) {
        h_A[i] = static_cast<half>(std::rand() / double(RAND_MAX));
    }
    for (int i = 0; i < h_B.size(); ++i) {
        h_B[i] = static_cast<half>(std::rand() / double(RAND_MAX));
    }
    // Initialize L1 and L2 matrices with random values
    for (int i = 0; i < h_L1.size(); ++i) {
        h_L1[i] = static_cast<half>(std::rand() / double(RAND_MAX));
    }
    for (int i = 0; i < h_L2.size(); ++i) {
        h_L2[i] = static_cast<half>(std::rand() / double(RAND_MAX));
    }

    // Allocate device memory
    thrust::device_vector<half> d_A = h_A;
    thrust::device_vector<half> d_B = h_B;
    thrust::device_vector<half> d_C(m * n, 0.0f);
    thrust::device_vector<half> d_L1 = h_L1;
    thrust::device_vector<half> d_L2 = h_L2;
    // Warp up the GPU
    lora_cublas(handle, d_A, d_B, d_L1, d_L2, d_C, m, n, k, lora_rank, lora_only);
    GEMM_CHECK_CUDA(cudaDeviceSynchronize());
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        lora_cublas(handle, d_A, d_B, d_L1, d_L2, d_C, m, n, k, lora_rank, lora_only);
        GEMM_CHECK_CUDA(cudaDeviceSynchronize());
    }
    double time = timer.seconds() / timing_iterations;
    double total_flops = 2.0 * m * (k * lora_rank + n * lora_rank); // 2 * m * k * r + 2 * m * n * r
    if (!lora_only) {
        total_flops += 2.0 * m * n * k; // Add the cublas hgemm part
    }
    double tflops = total_flops / (time * 1e12); // 2 * m * n * k for hgemm
    
    std::cout << "Average time for lora: " << time * 1000 << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;
    cublasDestroy(handle);
    return 0;
}
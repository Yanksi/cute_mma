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
    program.add_argument("-rs", "--random_seed")
        .help("Random seed for the input matrices")
        .default_value(static_cast<int>(std::time(nullptr))) // Use current time as default seed
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
    int timing_iterations = program.get<int>("--timing_iterations");

    std::cout << "M = " << m << std::endl;
    std::cout << "N = " << n << std::endl;
    std::cout << "K = " << k << std::endl;

    thrust::host_vector<half> h_A(m * k);
    thrust::host_vector<half> h_B(n * k);

    cublasHandle_t handle;
    getCublasTensorOpHandle(&handle);

    int random_seed = program.get<int>("--random_seed");
    std::srand(static_cast<unsigned int>(random_seed));

    // Initialize matrices A and B with random values
    for (int i = 0; i < m * k; ++i) {
        h_A[i] = static_cast<half>(std::rand() / double(RAND_MAX));
    }
    for (int i = 0; i < n * k; ++i) {
        h_B[i] = static_cast<half>(std::rand() / double(RAND_MAX));
    }

    // Allocate device memory
    thrust::device_vector<half> d_A = h_A;
    thrust::device_vector<half> d_B = h_B;
    thrust::device_vector<half> d_C(m * n, 0.0f);
    half alpha = static_cast<half>(1.0f);
    half beta = static_cast<half>(0.0f);
    GEMM_CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                &alpha, d_A.data().get(), k,
                d_B.data().get(), n,
                &beta, d_C.data().get(), m)); // Warm up the GPU
    GEMM_CHECK_CUDA(cudaDeviceSynchronize());
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        GEMM_CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                      &alpha, d_A.data().get(), k,
                                      d_B.data().get(), n,
                                      &beta, d_C.data().get(), m));
        GEMM_CHECK_CUDA(cudaDeviceSynchronize());
    }
    double time = timer.seconds() / timing_iterations;
    double tflops = (2.0 * m * n * k) / (time * 1e12); // 2 * m * n * k for hgemm
    std::cout << "Average time for cublasHgemm: " << time * 1000 << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;
    cublasDestroy(handle);
    return 0;
}
#include <cublas_v2.h>
#include <iostream>

int main() {
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS initialization failed!" << std::endl;
        return 1;
    }
    std::cout << "cuBLAS initialized successfully!" << std::endl;
    cublasDestroy(handle);
    return 0;
}

#include <cute/tensor.hpp>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>

#define GEMM_CHECK_CUBLAS(call)                                                                   \
    do {                                                                                          \
        cublasStatus_t status = call;                                                             \
        if (GEMM_UNLIKELY(status != CUBLAS_STATUS_SUCCESS)) {                                     \
            throw std::runtime_error("CUBLAS call failed with status " + std::to_string(status)); \
        }                                                                                         \
    } while (0)

void construct_block_diag(
  half const* R, int k,
  int reconn_sz, int group_id,
  half * R_diag
) {
  using namespace cute;
  half const* R_curr = R + (group_id * reconn_sz) * k;
  auto curr_group_layout = make_layout(
    make_shape(reconn_sz, k), LayoutRight{}
  );
  auto block_diag_mat_layout = make_layout(
    make_shape(k, k), LayoutRight{}
  );
  auto tile = make_tile(
    make_layout(reconn_sz),
    make_layout(reconn_sz)
  );
  auto curr_group_blocked = tiled_divide(curr_group_layout, tile);
  auto curr_mat_blocked = tiled_divide(block_diag_mat_layout, tile);
  Tensor curr_group_tensor = make_tensor(R_curr, curr_group_blocked)(_, 0, _);
  Tensor block_diag_tensor = make_tensor(R_diag, curr_mat_blocked);
  for (int i = 0; i < size<1>(curr_group_tensor); ++i) {
    copy(curr_group_tensor(_, i), block_diag_tensor(_, i, i));
  }
}


// cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle,
//                                   cublasOperation_t transa,
//                                   cublasOperation_t transb,
//                                   int m, int n, int k,
//                                   const __half           *alpha,
//                                   const __half           *A, int lda,
//                                   long long int          strideA,
//                                   const __half           *B, int ldb,
//                                   long long int          strideB,
//                                   const __half           *beta,
//                                   __half                 *C, int ldc,
//                                   long long int          strideC,
//                                   int batchCount)

void transform_weight(
  const thrust::device_vector<half> &d_B,
  const thrust::device_vector<half> &d_R,
        thrust::device_vector<half> &d_B_transformed,
  int group_sz, int k, int n_groups, int reconn_sz
  cublasHandle_t* handle
) {
  half alpha = 1.0;
  half beta = 0.0;
  for (int i = 0; i < n_groups; ++i) {
    GEMM_CHECK_CUBLAS(cublasHgemmStridedBatched(
      *handle, CUBLAS_OP_T, CUBLAS_OP_N,
      reconn_sz, group_size, reconn_sz, &alpha,
      d_R.data().get() + i * reconn_sz * k, k,
      reconn_sz,
      d_B.data().get() + i * group_sz * k, k,
      reconn_sz,
      &beta,
      d_B_transformed.data().get() + i * group_sz * k, k,
      reconn_sz,
      n_groups
    ));
  }
}

void reference_impl(
  const thrust::device_vector<half> &d_A,
  const thrust::device_vector<half> &d_R,
  const thrust::device_vector<half> &d_B,
        thrust::device_vector<half> &d_C,
  int m, int group_size, int n_groups, int k,
  int reconn_sz
  cublasHandle_t* handle
) {
  half alpha = 1.0;
  half beta = 0.0;
  int n = group_size * n_groups;
  thrust::device_vector<half> d_Bp(n * k);
  transform_weight(d_B, d_R, d_Bp, group_size, k, n_groups, reconn_sz, handle);
  GEMM_CHECK_CUBLAS(cublasHgemm(
      *handle, CUBLAS_OP_T, CUBLAS_OP_N,
      n, m, k, &alpha, d_Bp.data().get(), k, d_A.data().get(), k, &beta, d_C.data().get(), n)
    );
}

int main(int argc, char** argv) {
    int k = 32;
    int reconn_sz = 8;
    int n_groups = 2;

    thrust::host_vector<half> h_R(n_groups * reconn_sz * k);
    thrust::host_vector<half> h_R_block_diag(k * k);

    for (int j = 0; j < n_groups * reconn_sz * k; ++j) {
        h_R[j] = static_cast<half>( (rand() / double(RAND_MAX)) );
    }

    for (int j = 0; j < k * k; ++j) {
      h_R_block_diag[j] = static_cast<half>(0.0f);
    }

    int group_id = 1;
    construct_block_diag(
      h_R.data(), k,
      reconn_sz, group_id,
      h_R_block_diag.data()
    );

    for (int i = group_id * reconn_sz; i < (group_id + 1) * reconn_sz; ++i) {
        for (int j = 0; j < k; ++j) {
            printf("%.3f ", static_cast<float>(h_R[i * k + j]));
        }
        printf("\n");
    }

    for (int i = 0; i < k; ++i) {
      for (int j = 0; j < k; ++j) {
        printf("%.3f ", static_cast<float>(h_R_block_diag[i * k + j]));
      }
      printf("\n");
    }
}
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>

#define GEMM_UNLIKELY(x) __builtin_expect(!!(x), 0)
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
  int group_sz, int k, int n_groups, int reconn_sz,
  cublasHandle_t* handle
) {
  half alpha = 1.0;
  half beta = 0.0;
  for (int i = 0; i < n_groups; ++i) {
    GEMM_CHECK_CUBLAS(cublasHgemmStridedBatched(
      *handle, CUBLAS_OP_T, CUBLAS_OP_N,
      reconn_sz, group_sz, reconn_sz, &alpha,
      d_R.data().get() + i * reconn_sz * k, k,
      reconn_sz,
      d_B.data().get() + i * group_sz * k, k,
      reconn_sz,
      &beta,
      d_B_transformed.data().get() + i * group_sz * k, k,
      reconn_sz,
      k / reconn_sz
    ));
  }
}

void reference_impl(
  const thrust::device_vector<half> &d_A,
  const thrust::device_vector<half> &d_R,
  const thrust::device_vector<half> &d_B,
        thrust::device_vector<half> &d_C,
  int m, int group_size, int n_groups, int k,
  int reconn_sz,
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

void getCublasTensorOpHandle(cublasHandle_t* handle) {
    GEMM_CHECK_CUBLAS(cublasCreate(handle));
    GEMM_CHECK_CUBLAS(cublasSetMathMode(*handle, CUBLAS_TF32_TENSOR_OP_MATH));
}

int main(int argc, char** argv) {
  using namespace cute;
    int k = 32;
    int reconn_sz = 8;
    int n_groups = 8;
    int m = 2048;
    int group_sz = 16;
    int n = n_groups * group_sz;
    thrust::host_vector<half> h_A(m*k);
    thrust::host_vector<half> h_R(n_groups*reconn_sz*k, 0);
    thrust::host_vector<half> h_B(n*k);
    thrust::host_vector<half> h_C(m*n, -1);
    
    cublasHandle_t cublas_handle;
    getCublasTensorOpHandle(&cublas_handle);

    for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<half>( (rand() / double(RAND_MAX)) );
    for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<half>( (rand() / double(RAND_MAX)) );
    Tensor h_R_tensor = make_tensor(
      h_R.data(),
      make_layout(
        make_shape(n_groups * reconn_sz, k),
        LayoutRight{}
      )
    );

    Tensor h_R_4d = zipped_divide(
      h_R_tensor,
      make_tile(
        make_layout(reconn_sz),
        make_layout(reconn_sz)
      )
    );

    Tensor h_B_tensor = make_tensor(
      h_B.data(),
      make_layout(
        make_shape(n, k),
        LayoutRight{}
      )
    );

    Tensor h_B_4d = zipped_divide(
      h_B_tensor,
      make_tile(
        make_layout(group_sz),
        make_layout(reconn_sz)
      )
    );

    for (int i = 0; i < size<1>(h_R_4d); ++i) {
      h_R_4d(make_coord(0, 0), i) = static_cast<half>(2.0f);
      for (int j = 1; j < reconn_sz; ++j) {
        h_R_4d(make_coord(j, j), i) = static_cast<half>(1.0f);
        h_R_4d(make_coord(j, 0), i) = static_cast<half>(1.0f);
      }
    }

    // for (int i = 0; i < 2 * reconn_sz; ++i) {
    //   for (int j = 0; j < k; ++j) {
    //     printf("%.3f ", static_cast<float>(h_R_tensor(i, j)));
    //   }
    //   print("\n");
    // }

    // return 0;

    thrust::device_vector<half> d_A = h_A;
    thrust::device_vector<half> d_R = h_R;
    thrust::device_vector<half> d_B = h_B;
    thrust::device_vector<half> d_B_transformed(n * k);

    transform_weight(d_B, d_R, d_B_transformed, group_sz, k, n_groups, reconn_sz, &cublas_handle);

    thrust::host_vector<half> h_B_transformed = d_B_transformed;
    Tensor d_B_transformed_tensor = make_tensor(
      h_B_transformed.data(),
      h_B_4d.layout()
    );

    for (int i = 0; i < size<1>(d_B_transformed_tensor); ++i) {
      for (int j = 0; j < group_sz; ++j) {
        for (int k = 0; k < reconn_sz; ++k) {
          // Check the transformation
          float val = static_cast<float>(d_B_transformed_tensor(make_coord(j, k), i));
          float expected_val = static_cast<float>(h_B_4d(make_coord(j, k), i)) + static_cast<float>(h_B_4d(make_coord(j, 0), i));
          if (std::abs(val - expected_val) > 1e-3) {
            std::cerr << "Transformation mismatch at group " << i << ", j=" << j << ", k=" << k
                      << ": got " << val << ", expected " << expected_val << std::endl;
            return 1;
          }
        }
      }
    }
}
#pragma once
// S2-Basis: Tensor-Handles, cuBLAS-Singleton, Dims.
#include "common.h"
#include <cublas_v2.h>
#include <vector>

inline cublasHandle_t cublas_handle() {
    static cublasHandle_t h = nullptr;
    if (!h) {
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
            std::printf("cublasCreate FAIL\n"); exit(1);
        }
    }
    return h;
}

// Row-major C(M,N) = A(M,K) @ B(K,N), bf16 IO, fp32 accumulate.
inline void gemm_nn(int M, int N, int K, const bf16* A, int lda,
                    const bf16* B, int ldb, bf16* C, int ldc) {
    // C_row(M,N) == Ct(N,M) col; Ct = Bt @ At (beide OP_N auf Aliasen)
    float al = 1.0f, be = 0.0f;
    cublasStatus_t s = cublasGemmEx(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al,
        B, CUDA_R_16BF, N, A, CUDA_R_16BF, K, &be, C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::printf("gemm_nn FAIL %d M=%d N=%d K=%d\n", s, M, N, K); exit(1);
    }
}

struct Buf {
    void* p = nullptr;
    size_t n = 0;  // Elemente
    template <typename T> T* as() { return (T*)p; }
};

inline Buf alloc_gpu(size_t bytes) {
    Buf b;
    CUDA_CHECK(cudaMalloc(&b.p, bytes));
    b.n = bytes;
    return b;
}

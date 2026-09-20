#pragma once
// Lineare Schichten in bf16 (Tensor Cores), grads: dX bf16, dW fp32.
// Gewichtslayout wie nn.Linear: W(N,K) row-major ([out,in]).
// Y(M,N) = X(M,K) @ W^T ; dX = dY @ W ; dW = dY^T @ X
#include "util.h"

// Y(M,N) bf16 = X(M,K) bf16 @ W(N,K)^T
// Y^T(N,M) = W(N,K) @ X^T(K,M): A=W OP_T lda=K, B=X OP_N lda=K.
inline void linear_fwd(int M, int N, int K, const bf16* X, const bf16* W,
                       bf16* Y) {
    float al = 1.0f, be = 0.0f;
    cublasStatus_t s = cublasGemmEx(
        cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &al,
        W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &be, Y, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) { std::printf("linear_fwd FAIL\n"); exit(1); }
}

// dX(M,K) bf16 = dY(M,N) @ W(N,K) (+ beta*dX bei beta != 0)
inline void linear_dX(int M, int N, int K, const bf16* dY, const bf16* W,
                      bf16* dX, float beta = 0.0f) {
    // dXt(K,M) = Wct(K,N) @ dYt(N,M)
    float al = 1.0f;
    cublasStatus_t s = cublasGemmEx(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &al,
        W, CUDA_R_16BF, K, dY, CUDA_R_16BF, N, &beta, dX, CUDA_R_16BF, K,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) { std::printf("linear_dX FAIL\n"); exit(1); }
}

// dW(N,K) fp32 = dY(M,N)^T @ X(M,K) (+ beta*dW bei beta != 0, z.B. Up-Akku)
inline void linear_dW(int M, int N, int K, const bf16* dY, const bf16* X,
                      float* dW, float beta = 0.0f) {
    // dWt(K,N) = Xc(K,M) @ dYt^T(M,N)
    float al = 1.0f;
    cublasStatus_t s = cublasGemmEx(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, K, N, M, &al,
        X, CUDA_R_16BF, K, dY, CUDA_R_16BF, N, &beta, dW, CUDA_R_32F, K,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) { std::printf("linear_dW FAIL\n"); exit(1); }
}

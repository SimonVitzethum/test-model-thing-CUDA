#pragma once
// S2-Basis: Tensor-Handles, cuBLAS-Singleton, Dims.
#include "common.h"
#include <cublas_v2.h>
#include <vector>

#include <cstdlib>
#include <cstring>

// How the host thread waits for the GPU (set before the first CUDA call).
// TMT_CUDA_WAIT=yield (default): polls but yields the core to other processes;
// same speed as spin. block: sleeps until the GPU is done, ~1/3 of a core
// instead of a full one, measured ~5% slower. spin: CUDA's busy wait.
static const int tmt_cuda_wait_mode = [] {
    const char* v = getenv("TMT_CUDA_WAIT");
    unsigned flag = cudaDeviceScheduleYield;
    if (v && !strcmp(v, "spin")) flag = cudaDeviceScheduleSpin;
    else if (v && !strcmp(v, "block")) flag = cudaDeviceScheduleBlockingSync;
    cudaSetDeviceFlags(flag);
    return (int)flag;
}();

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

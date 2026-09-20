// Roofline-Vermessung: DRAM-Bandbreite (Streaming, >L2) + bf16-GEMM-Dach.
// Aufruf: ./roofline   (nutzt cublas direkt, keine Modellabhängigkeit)
#include "common.h"
#include <cublas_v2.h>
#include <vector>

int main() {
    cublasHandle_t h;
    cublasCreate(&h);
    // ---- 1. DRAM-Bandbreite: 1 GB Stream-Triad (read+read+write) ----
    {
        const long N = 1L << 28;  // 256M floats = 1 GB
        float *a, *b, *c;
        cudaMalloc(&a, N * 4); cudaMalloc(&b, N * 4); cudaMalloc(&c, N * 4);
        // Triad-Kernel inline via Thrust? per Hand:
        // einfacher Memcpy-Durchsatz-Test (read+write = 2x):
        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < 10; ++i)
            cudaMemcpyAsync(c, a, N * 4, cudaMemcpyDeviceToDevice);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0;
        cudaEventElapsedTime(&ms, s, e);
        ms /= 10;
        double gbs = (2.0 * N * 4) / (ms / 1e3) / 1e9;
        std::printf("DRAM memcpy: %.0f GB/s (read+write)\n", gbs);
        cudaFree(a); cudaFree(b); cudaFree(c);
    }
    // ---- 2. bf16-GEMM-Dach: Sweep über (M,N,K) ----
    {
        using bf16 = __nv_bfloat16;
        struct Case { int M, N, K; };
        std::vector<Case> cs = {{512, 512, 512}, {2048, 1280, 1280},
                                {8192, 1280, 1280}, {8192, 8192, 8192},
                                {16384, 4096, 4096}, {32768, 2048, 2048}};
        for (auto [M, N, K] : cs) {
            bf16 *A, *B, *C;
            cudaMalloc(&A, (size_t)M * K * 2);
            cudaMalloc(&B, (size_t)K * N * 2);
            cudaMalloc(&C, (size_t)M * N * 2);
            float al = 1, be = 0;
            // Warmup (cublas autotune einpendeln)
            for (int i = 0; i < 5; ++i)
                cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al, B,
                             CUDA_R_16BF, N, A, CUDA_R_16BF, K, &be, C,
                             CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cudaEvent_t s, e;
            cudaEventCreate(&s); cudaEventCreate(&e);
            const int IT = 20;
            cudaEventRecord(s);
            for (int i = 0; i < IT; ++i)
                cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al, B,
                             CUDA_R_16BF, N, A, CUDA_R_16BF, K, &be, C,
                             CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cudaEventRecord(e);
            cudaEventSynchronize(e);
            float ms = 0;
            cudaEventElapsedTime(&ms, s, e);
            ms /= IT;
            double tf = 2.0 * M * N * K / (ms / 1e3) / 1e12;
            std::printf("GEMM row(%d,%d)x(%d,%d): %.3f ms -> %.0f TFLOPS\n",
                        M, K, K, N, ms, tf);
            cudaFree(A); cudaFree(B); cudaFree(C);
        }
    }
    return 0;
}

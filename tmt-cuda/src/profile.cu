// Profil: MoE bei Realgrößen -> effektive TFLOPS vs. Dach (49 TFLOPS).
// Arena vorallokiert (wie im Training); misst steady-state ohne malloc.
#include "util.h"
#include "linalg.cu"
#include "moe.cu"
#include <cstdio>

static double now_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
}

int main() {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    int N = 2048, E = 8, K = 2, D = 1280;
    bf16 *X, *Y, *dY, *dX, *Wr, *Wx[16];
    float *dWr, *dWx[16];
    cudaMalloc(&X, (size_t)N * D * 2);
    cudaMalloc(&Y, (size_t)N * D * 2);
    cudaMalloc(&dY, (size_t)N * D * 2);
    cudaMalloc(&dX, (size_t)N * D * 2);
    cudaMalloc(&Wr, (size_t)E * D * 2);
    cudaMalloc(&dWr, (size_t)E * D * 4);
    for (int x = 0; x < E; ++x) {
        cudaMalloc(&Wx[x], (size_t)D * D * 2);
        cudaMalloc(&dWx[x], (size_t)D * D * 4);
    }
    MoeKeep keep;
    MoeWs ws;
    moe_keep_alloc(keep, N, E, K, D);
    moe_ws_alloc(ws, N, E, K, D);
    const int IT = 60;
    const int WARM = 15;
    // Warmup (cublas-Autotune einpendeln lassen)
    for (int i = 0; i < WARM; ++i) {
        moe_forward(X, Wr, Wx, Y, keep, ws, N, E, K, D);
        moe_backward(dY, Wx, Wr, dWr, dWx, dX, keep, ws, 0.01f, 0.001f);
    }
    double gflop_f =
        (2.0 * N * D * E + 2.0 * N * K * D * D) / 1e9;
    cudaEventRecord(s);
    for (int i = 0; i < IT; ++i)
        moe_forward(X, Wr, Wx, Y, keep, ws, N, E, K, D);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    double ms = now_ms(s, e) / IT;
    printf("MoE fwd  N=%d E=%d K=%d D=%d: %.2f ms | %.1f TFLOPS (%.0f%% von 49)\n",
           N, E, K, D, ms, gflop_f / (ms / 1e3) / 1e3,
           gflop_f / (ms / 1e3) / 10 / 49);
    cudaEventRecord(s);
    for (int i = 0; i < IT; ++i) {
        moe_forward(X, Wr, Wx, Y, keep, ws, N, E, K, D);
        moe_backward(dY, Wx, Wr, dWr, dWx, dX, keep, ws, 0.01f, 0.001f);
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    ms = now_ms(s, e) / IT;
    double gflop_b = gflop_f * 2.2;
    printf("MoE fwd+bwd: %.2f ms | %.1f TFLOPS (%.0f%% von 49)\n", ms,
           (gflop_f + gflop_b) / (ms / 1e3) / 1e3,
           (gflop_f + gflop_b) / (ms / 1e3) / 10 / 49);
    return 0;
}

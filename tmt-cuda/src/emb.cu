#pragma once
// Embedding-Gather (bf16) + Atomic-Scatter der fp32-Grade.
// Tabelle: W(256,D) bf16 compute-Kopie + Wm(256,D) fp32 master (Adam).
// Grade: dWm(256,D) fp32, per Batch genullt, atomar aufsummiert.
#include "common.h"

__global__ void emb_gather_kernel(const bf16* W, const int* ids, bf16* Out,
                                  int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    bf16* d = Out + (long)i * D;
    if (ids[i] < 0) { for (int dd = 0; dd < D; ++dd) d[dd] = f2bf(0.f); return; }  // ignored target
    const bf16* s = W + (long)ids[i] * D;
    for (int dd = 0; dd < D; ++dd) d[dd] = s[dd];
}

__global__ void emb_scatter_kernel(const float* dOut, const int* ids,
                                   float* dW, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float* d = dW + (long)ids[i] * D;
    const float* s = dOut + (long)i * D;
    for (int dd = 0; dd < D; ++dd) atomicAdd(&d[dd], s[dd]);
}

inline void emb_forward(const bf16* W, const int* ids, bf16* Out, int N,
                        int D) {
    emb_gather_kernel<<<(N + 255) / 256, 256>>>(W, ids, Out, N, D);
}

inline void emb_backward(const float* dOut, const int* ids, float* dW, int N,
                         int D) {
    emb_scatter_kernel<<<(N + 255) / 256, 256>>>(dOut, ids, dW, N, D);
}

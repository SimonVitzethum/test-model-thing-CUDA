#pragma once
// LayerNorm mit Gamma/Beta, bf16 IO, fp32-Statistiken.
// y = (x-mean)*rstd*gamma + beta. Backward Standard-Formel.
#include "common.h"

// Forward: ein Block pro Zeile. Speichert mean/rstd + xhat? xhat wird für
// Backward aus x,mean,rstd rekomputiert (kein Extraspeicher).
__global__ void ln_fwd_kernel(const bf16* X, const float* gamma,
                              const float* beta, bf16* Y, float* mean,
                              float* rstd, int N, int D) {
    int r = blockIdx.x;
    if (r >= N) return;
    __shared__ float buf[256];
    float sum = 0, sq = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = bf2f(X[(long)r * D + d]);
        sum += v; sq += v * v;
    }
    buf[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    __shared__ float m_, rs_;
    if (threadIdx.x == 0) {
        // Quadratsumme: zweiter Reduce-Durchgang über sq fehlt hier bewusst
        // NICHT — sq wurde oben pro Thread akkumuliert, jetzt reduzieren:
        m_ = buf[0] / D;
    }
    __syncthreads();
    buf[threadIdx.x] = sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float v = buf[0] / D - m_ * m_;
        mean[r] = m_;
        rs_ = rsqrtf(v + 1e-5f);
        rstd[r] = rs_;
    }
    __syncthreads();
    float mm = m_, rr = rs_;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float xh = (bf2f(X[(long)r * D + d]) - mm) * rr;
        Y[(long)r * D + d] = f2bf(xh * gamma[d] + beta[d]);
    }
}

// Backward: dX, dGamma, dBeta aus dY. dGamma/dBeta via Block-Atomics.
__global__ void ln_bwd_kernel(const bf16* X, const bf16* dY,
                              const float* gamma, const float* mean,
                              const float* rstd, bf16* dX, float* dGamma,
                              float* dBeta, int N, int D) {
    int r = blockIdx.x;
    if (r >= N) return;
    __shared__ float buf[256];
    float m = mean[r], rr = rstd[r];
    // Reduktion 1: sum(dy*gamma), Reduktion 2: sum(dy*gamma*xhat)
    float s1 = 0, s2 = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float dy = bf2f(dY[(long)r * D + d]) * gamma[d];
        float xh = (bf2f(X[(long)r * D + d]) - m) * rr;
        s1 += dy; s2 += dy * xh;
        atomicAdd(&dBeta[d], bf2f(dY[(long)r * D + d]));
        atomicAdd(&dGamma[d], bf2f(dY[(long)r * D + d]) * xh);
    }
    buf[threadIdx.x] = s1;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    __shared__ float t1;
    if (threadIdx.x == 0) t1 = buf[0];
    __syncthreads();
    buf[threadIdx.x] = s2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    float a = t1 / D, b = buf[0] / D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float dy = bf2f(dY[(long)r * D + d]) * gamma[d];
        float xh = (bf2f(X[(long)r * D + d]) - m) * rr;
        dX[(long)r * D + d] = f2bf(rr * (dy - a - xh * b));
    }
}

inline void layernorm_fwd(const bf16* X, const float* gamma,
                          const float* beta, bf16* Y, float* mean,
                          float* rstd, int N, int D) {
    ln_fwd_kernel<<<N, 256>>>(X, gamma, beta, Y, mean, rstd, N, D);
}

inline void layernorm_bwd(const bf16* X, const bf16* dY, const float* gamma,
                          const float* mean, const float* rstd, bf16* dX,
                          float* dGamma, float* dBeta, int N, int D) {
    CUDA_CHECK(cudaMemset(dGamma, 0, (size_t)D * 4));
    CUDA_CHECK(cudaMemset(dBeta, 0, (size_t)D * 4));
    ln_bwd_kernel<<<N, 256>>>(X, dY, gamma, mean, rstd, dX, dGamma, dBeta,
                              N, D);
}

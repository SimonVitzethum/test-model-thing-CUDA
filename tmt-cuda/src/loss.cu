#pragma once
// Losses (alle Mittel über N = B*T Positionen):
// - CE über 256 Bytes (fused, fp32-Probs für Backward)
// - Stop-BCE mit pos_weight (Logits in)
// - Latent-MSE gegen EMA-Target (dX-Anteil)
// - Variance-Hinge (Skalar; dX über Fenster-Varianz)
// Gibt Total-Loss (Host-float) + CE-Mittel (für BPC) zurück.
#include "common.h"

// CE: max-sub, exp-sum, -logit_t+logsum. Speichert probs fp32 (N,256).
__global__ void ce_fwd_kernel(const bf16* logits, const int* tgt,
                              float* probs, float* loss_out, int N) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    const bf16* L = logits + (long)r * 256;
    float mx = bf2f(L[0]);
    for (int i = 1; i < 256; ++i) mx = fmaxf(mx, bf2f(L[i]));
    float se = 0;
    for (int i = 0; i < 256; ++i) se += expf(bf2f(L[i]) - mx);
    float lse = logf(se) + mx;
    for (int i = 0; i < 256; ++i)
        probs[(long)r * 256 + i] = expf(bf2f(L[i]) - mx) / se;
    loss_out[r] = lse - bf2f(L[tgt[r]]);
}

// dLogits = w*(p - onehot)/N (bf16)
__global__ void ce_bwd_kernel(const float* probs, const int* tgt,
                              bf16* dLogits, float w, int N) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    for (int i = 0; i < 256; ++i) {
        float p = probs[(long)r * 256 + i] - (i == tgt[r] ? 1.0f : 0.0f);
        dLogits[(long)r * 256 + i] = f2bf(w * p / N);
    }
}

// Stop-BCE mit pos_weight. Gibt mittleren Loss zurück (Device->Host via sum).
__global__ void stop_fwd_kernel(const bf16* s, const int* end, float* loss_out,
                                float pos_w, int N) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    float z = bf2f(s[r]);
    float t = end[r] ? 1.0f : 0.0f;
    // -[pw*t*log(s) + (1-t)*log(1-s)], stabil
    float l = -(pos_w * t * (-log1pf(expf(-z))) +
                (1.0f - t) * (-z - log1pf(expf(-z))));
    loss_out[r] = l;
}

// dStop = w*(sigmoid(z)-t)*w(t)/N, w(t) = pos_w für t=1 sonst 1
__global__ void stop_bwd_kernel(const bf16* s, const int* end, bf16* ds,
                                float pos_w, float w, int N) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    float z = bf2f(s[r]);
    float t = end[r] ? 1.0f : 0.0f;
    float g = sigmoid_f(z) - t;
    if (t > 0.5f) g *= pos_w;
    ds[r] = f2bf(w * g / N);
}

// Latent-MSE-Anteil an dX: dX += 2*(x-tgt)/M * w (fp32 dX)
__global__ void latent_bwd_kernel(const float* x, const float* tgt,
                                  float* dX, float w, long M) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    dX[i] += 2.0f * (x[i] - tgt[i]) / M * w;
}

// Varianz-Hinge: loss = max(0, 1-sqrt(var+eps)); dX += w*d(var)/... 
// d(var)/dx_i = 2*(x_i-mean)/M ; dL/dvar = -0.5/sqrt(var+eps) falls aktiv.
__global__ void var_bwd_kernel(const float* x, float* dX, float mean,
                               float scl, long M) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    dX[i] += scl * 2.0f * (x[i] - mean) / M;
}

#pragma once
// Fused AdamW: ein Kernel pro Parameter (alle groß -> Launch egal).
// Master+m+v fp32, work bf16-Arbeitskopie, grad fp32.
#include "common.h"

__global__ void adam_one_kernel(float* master, float* m, float* v,
                                bf16* work, const float* grad, long n,
                                float lr, float b1, float b2, float eps,
                                float wd, float bc1, float bc2) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float nm = b1 * m[i] + (1.0f - b1) * g;
    float nv = b2 * v[i] + (1.0f - b2) * g * g;
    m[i] = nm; v[i] = nv;
    float w = master[i] - lr * ((nm / bc1) / (sqrtf(nv / bc2) + eps) +
                                wd * master[i]);
    master[i] = w;
    work[i] = f2bf(w);
}

inline void adam_step_one(float* master, float* m, float* v, bf16* work,
                          const float* grad, long n, float lr, float b1,
                          float b2, float eps, float wd, int step) {
    float bc1 = 1.0f - powf(b1, step);
    float bc2 = 1.0f - powf(b2, step);
    adam_one_kernel<<<(n + 255) / 256, 256>>>(
        master, m, v, work, grad, n, lr, b1, b2, eps, wd, bc1, bc2);
}

// bf16-Grad -> fp32 aufaddieren (für bf16 dX-Pfade).
__global__ void cast_add_kernel(const bf16* s, float* d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] += bf2f(s[i]);
}

// ---------- Multi-tensor optimizer ----------
// All parameters are split once into chunks of up to MT_CHUNK elements; one
// launch then covers every parameter (norm, AdamW, zeroing) instead of one
// launch plus a host synchronization per parameter.
constexpr int MT_CHUNK = 4096;
struct MTChunk { int param; int len; long start; };
struct MTParams {  // device arrays, one entry per parameter
    float **master = nullptr, **m = nullptr, **v = nullptr, **grad = nullptr;
    bf16** work = nullptr;
    unsigned char* flags = nullptr;  // bit 0: in the gradient norm, bit 1: updated by AdamW
    MTChunk* chunks = nullptr;
    int nchunks = 0;
    double* sumsq = nullptr;  // global squared gradient norm
};

__global__ void mt_sumsq_kernel(MTParams P) {
    const MTChunk c = P.chunks[blockIdx.x];
    __shared__ double buf[256];
    double s = 0;
    if (P.flags[c.param] & 1) {
        const float* g = P.grad[c.param] + c.start;
        for (int i = threadIdx.x; i < c.len; i += blockDim.x) s += (double)g[i] * g[i];
    }
    buf[threadIdx.x] = s;
    __syncthreads();
    for (int t = blockDim.x / 2; t > 0; t >>= 1) {
        if (threadIdx.x < t) buf[threadIdx.x] += buf[threadIdx.x + t];
        __syncthreads();
    }
    if (threadIdx.x == 0 && buf[0] != 0) atomicAdd(P.sumsq, buf[0]);
}

// AdamW with global-norm clipping computed on the device. A non-finite norm
// skips the update entirely (the host then refuses the step).
__global__ void mt_adam_kernel(MTParams P, float clip, float lr, float b1, float b2, float eps,
                               float wd, float bc1, float bc2) {
    const MTChunk c = P.chunks[blockIdx.x];
    if (!(P.flags[c.param] & 2)) return;
    double s = *P.sumsq;
    if (!isfinite(s)) return;
    float scale = clip > 0 ? fminf(1.f, clip / fmaxf((float)sqrt(s), 1e-12f)) : 1.f;
    float *master = P.master[c.param] + c.start, *m = P.m[c.param] + c.start, *v = P.v[c.param] + c.start;
    const float* grad = P.grad[c.param] + c.start;
    bf16* work = P.work[c.param] + c.start;
    for (int i = threadIdx.x; i < c.len; i += blockDim.x) {
        float g = grad[i] * scale;
        float nm = b1 * m[i] + (1.0f - b1) * g;
        float nv = b2 * v[i] + (1.0f - b2) * g * g;
        m[i] = nm; v[i] = nv;
        float w = master[i] - lr * ((nm / bc1) / (sqrtf(nv / bc2) + eps) + wd * master[i]);
        master[i] = w;
        work[i] = f2bf(w);
    }
}

__global__ void mt_zero_kernel(MTParams P) {
    const MTChunk c = P.chunks[blockIdx.x];
    float* g = P.grad[c.param] + c.start;
    for (int i = threadIdx.x; i < c.len; i += blockDim.x) g[i] = 0.f;
}

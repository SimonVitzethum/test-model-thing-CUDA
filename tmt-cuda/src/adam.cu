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

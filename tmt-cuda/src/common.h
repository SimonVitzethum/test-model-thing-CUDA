#pragma once
// Gemeinsame Typen/Helfer. Compute = BF16, Mathe = FP32, Master = FP32.
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

using bf16 = __nv_bfloat16;

__host__ __device__ __forceinline__ float bf2f(bf16 x) {
    return __bfloat162float(x);
}
__host__ __device__ __forceinline__ bf16 f2bf(float x) {
    return __float2bfloat16_rn(x);
}
__host__ __device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}
__host__ __device__ __forceinline__ float silu_f(float x) {
    return x * sigmoid_f(x);
}
__host__ __device__ __forceinline__ float silu_bwd(float x, float dy) {
    float s = sigmoid_f(x);
    return dy * s * (1.0f + x * (1.0f - s));
}

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::printf("CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

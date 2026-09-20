#pragma once
// Shared CUDA operators, configuration and owned parameter storage:
// Byte-Shards, Fenster-TBPTT, CE+Stop+Latent+Var, AdamW, Checkpoints.
// Aufruf: ./train data ckpt [key=val ...]
//   dim layers experts topk batch seqlen lr warmup decaysteps minlr
//   aux zloss latent ce var stop stopposw ematau gradclip maxcarry
//   steps saveevery seed
#include "util.h"
#include "linalg.cu"
#include "moe.cu"
#include "norm.cu"
#include "cell.cu"
#include "emb.cu"
#include "loss.cu"
#include "adam.cu"
#include "mla.cu"
#include <cmath>
#include <cstring>
#include <ctime>

#include "config.h"
#include <stdexcept>

// Per-model ownership: independent models never share parameter indices.
struct DeviceMemory {
    std::vector<void*> allocations;
    DeviceMemory() = default;
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    template<class T> void allocate(T*& ptr, size_t bytes) {
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        allocations.push_back(ptr);
    }
    ~DeviceMemory() { for (void* ptr : allocations) cudaFree(ptr); }
};
struct Par {
    float *master, *m, *v, *grad;
    bf16* work;
    long n;
};
struct ParameterStore {
    DeviceMemory memory;
    std::vector<Par> values;
    Par& at(size_t i) { return values.at(i); }
    size_t add(long n) {
        if (n <= 0 || n > INT_MAX) throw std::runtime_error("parameter too large");
        Par p{}; p.n = n;
        memory.allocate(p.master, n * 4);
        memory.allocate(p.m, n * 4);
        memory.allocate(p.v, n * 4);
        memory.allocate(p.grad, n * 4);
        memory.allocate(p.work, n * 2);
        CUDA_CHECK(cudaMemset(p.m, 0, n * 4));
        CUDA_CHECK(cudaMemset(p.v, 0, n * 4));
        CUDA_CHECK(cudaMemset(p.grad, 0, n * 4));
        values.push_back(p);
        return values.size() - 1;
    }
};

static void host_init(float* h, long n, float a, float b, unsigned& s) {
    for (long i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        h[i] = a + (b - a) * ((s >> 9) * (1.0f / 8388608.0f));
    }
}
static void host_normal(float* h, long n, float std, unsigned& s) {
    for (long i = 0; i < n; ) {
        s = s * 1664525u + 1013904223u;
        float u1 = ((s >> 9) + 1) * (1.0f / 8388609.0f);
        s = s * 1664525u + 1013904223u;
        float u2 = ((s >> 9) + 1) * (1.0f / 8388609.0f);
        float r = sqrtf(-2 * logf(u1)), t = 6.2831853f * u2;
        h[i++] = r * cosf(t) * std;
        if (i < n) h[i++] = r * sinf(t) * std;
    }
}

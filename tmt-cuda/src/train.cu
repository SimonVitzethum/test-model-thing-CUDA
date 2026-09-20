#pragma once
// S2-Trainings-Loop (Recurrence + MoE, noch ohne MLA):
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
#include <cmath>
#include <cstring>
#include <ctime>

// ---------- Config ----------
struct Cfg {
    int dim = 1280, layers = 32, experts = 8, topk = 2;
    int batch = 16, seqlen = 128;
    float lr = 5e-4f; int warmup = 200, decaysteps = 8000; float minlr = 0.1f;
    float aux = 0.01f, zloss = 0.001f, latent = 1.f, ce = 1.f, var = 1.f;
    float stop = 1.f, stopposw = 20.f, ematau = 0.99f, gradclip = 1.f;
    int maxcarry = 2048, steps = 0, saveevery = 500, seed = 0;
};

static void set_cfg(Cfg& c, const char* k, const char* v) {
#define I(f) if (!strcmp(k, #f)) { c.f = atoi(v); return; }
#define F(f) if (!strcmp(k, #f)) { c.f = (float)atof(v); return; }
    I(dim) I(layers) I(experts) I(topk) I(batch) I(seqlen)
    F(lr) I(warmup) I(decaysteps) F(minlr)
    F(aux) F(zloss) F(latent) F(ce) F(var) F(stop) F(stopposw) F(ematau)
    F(gradclip) I(maxcarry) I(steps) I(saveevery) I(seed)
    std::printf("unbekannt: %s\n", k); exit(1);
#undef I
#undef F
}

// ---------- Parameter (per Index, nie Pointer halten!) ----------
struct Par {
    float *master, *m, *v, *grad;
    bf16* work;
    long n;
};
static std::vector<Par> PARS;
inline Par& P(size_t i) { return PARS[i]; }
static size_t new_par(long n) {
    Par p;
    p.n = n;
    CUDA_CHECK(cudaMalloc(&p.master, n * 4));
    CUDA_CHECK(cudaMalloc(&p.m, n * 4));
    CUDA_CHECK(cudaMalloc(&p.v, n * 4));
    CUDA_CHECK(cudaMalloc(&p.grad, n * 4));
    CUDA_CHECK(cudaMalloc(&p.work, n * 2));
    CUDA_CHECK(cudaMemset(p.m, 0, n * 4));
    CUDA_CHECK(cudaMemset(p.v, 0, n * 4));
    CUDA_CHECK(cudaMemset(p.grad, 0, n * 4));
    PARS.push_back(p);
    return PARS.size() - 1;
}

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

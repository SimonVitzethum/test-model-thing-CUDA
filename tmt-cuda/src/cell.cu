#pragma once
// Fused persistente Recurrent-Cell (S1, Forward).
//
// a = sigmoid(decay + gate*x); state = a*state + (1-a)*x        (persistent über T, Register)
// y = silu(LN(state)) + x                 (Residual wie TMT-Layer)
//
// Layout (B,T,D), D-kontinuierlich -> coalesced. Ein Thread = ein (b,d),
// loopt über die ganze Sequenz: genau 3 Launches pro Fenster (state,
// stats, out) statt ~T*L*10.
#include "common.h"

// Options shared by the recurrence kernels (passed by value).
// Document reset: at a position whose input byte equals docsep, a_t = 0, so
// s_t = x_t. Carry, gradient flow and traces are cut by the same recurrence.
struct CellOpt {
    const int* ids = nullptr;  // (B,T) input bytes, needed for docsep
    int docsep = -1;           // -1: no document reset
    float gamma = 1.f;         // trace decay per byte (1: exact traces)
    float *trDec = nullptr, *trGate = nullptr;  // (B,D) traces, in/out
    float *lam = nullptr, *prod = nullptr;      // (B,D) outputs for emb_trace
    float *logDec = nullptr, *logGate = nullptr, *logEmb = nullptr;  // trace part of the gradient
};
__device__ __forceinline__ float cell_a(const CellOpt& o, float decay, const float* gate,
                                        int d, float x, int b, int T, int t) {
    if (o.docsep >= 0 && o.ids[b * T + t] == o.docsep) return 0.f;
    return sigmoid_f(decay + (gate ? gate[d] * x : 0.f));
}

// Pass 1: Zeit-Loop, schreibt States (fp32) für die Norm.
// carry (B,D) oder nullptr (= Nullstart).
__global__ void state_pass_kernel(const bf16* __restrict__ X,
                                  float* __restrict__ S,
                                  const float* __restrict__ decay,
                                  const float* __restrict__ carry,
                                  int B, int T, int D, const float* gate, CellOpt o) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || d >= D) return;
    float state = carry ? carry[(long)b * D + d] : 0.0f;
    for (int t = 0; t < T; ++t) {
        int idx = (b * T + t) * D + d;
        float x = bf2f(X[idx]);
        float dec = cell_a(o, decay[d], gate, d, x, b, T, t);
        state = dec * state + (1.f - dec) * x;
        S[idx] = state;
    }
}

// Pass 2: ein Block pro (b,t) reduziert Mittelwert + rstd über D.
__global__ void stats_kernel(const float* __restrict__ S,
                             float* __restrict__ mean,
                             float* __restrict__ rstd,
                             int B, int T, int D) {
    int row = blockIdx.x;               // row = b*T + t
    if (row >= B * T) return;
    __shared__ float buf[256];
    __shared__ float total_sum;
    float sum = 0.0f, sq = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = S[row * D + d];
        sum += v; sq += v * v;
    }
    buf[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) total_sum = buf[0];
    __syncthreads();
    buf[threadIdx.x] = sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = total_sum / D;
        float v = buf[0] / D - m * m;
        mean[row] = m;
        rstd[row] = rsqrtf(v + 1e-5f);
    }
}

// Pass 3: Norm + SiLU + Residual, wieder persistent über T.
__global__ void out_pass_kernel(const bf16* __restrict__ X,
                                const float* __restrict__ S,
                                const float* __restrict__ mean,
                                const float* __restrict__ rstd,
                                bf16* __restrict__ Y,
                                int B, int T, int D) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || d >= D) return;
    for (int t = 0; t < T; ++t) {
        int row = b * T + t;
        int idx = row * D + d;
        float h = (S[idx] - mean[row]) * rstd[row];
        Y[idx] = f2bf(silu_f(h) + bf2f(X[idx]));
    }
}

// Nur State-Loop (S2-Harness nutzt danach layernorm_fwd aus norm.cu).
// carry (B,D) fp32 oder nullptr.
inline void state_forward(const bf16* X, float* S, const float* decay,
                          const float* carry, int B, int T, int D,
                          const float* gate = nullptr, CellOpt o = CellOpt()) {
    const int TPB = 256;
    dim3 grid(B, (D + TPB - 1) / TPB);
    state_pass_kernel<<<grid, TPB>>>(X, S, decay, carry, B, T, D, gate, o);
}

void cell_forward(const bf16* X, float* S, const float* decay,
                  float* mean, float* rstd, bf16* Y,
                  int B, int T, int D, cudaStream_t stream = 0) {
    const int TPB = 256;
    dim3 grid1(B, (D + TPB - 1) / TPB);
    state_pass_kernel<<<grid1, TPB, 0, stream>>>(X, S, decay, nullptr, B, T,
                                                 D, nullptr, CellOpt());
    stats_kernel<<<B * T, TPB, 0, stream>>>(S, mean, rstd, B, T, D);
    out_pass_kernel<<<grid1, TPB, 0, stream>>>(X, S, mean, rstd, Y, B, T, D);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        std::printf("LAUNCH-FAIL: %s (grid1=%d,%d stats=%d)\n",
                    cudaGetErrorString(le), grid1.x, grid1.y, B * T);
        exit(3);
    }
}

// Exact within-window derivative. The incoming carry is constant for TBPTT,
// but contributes to the decay/gate derivatives at the first position.
//
// Hybrid traces (optional, o.trDec/o.trGate): e = ds_(t0-1)/dθ for the
// per-channel parameters is carried across windows. After the backward loop,
// `future` is exactly λ = dL/ds_(t0-1), so λ·e_in adds the credit of all bytes
// before the window. The trace is then advanced in closed form:
//   e_out = P·e_in + Σ_t (Π_{k>t} γa_k)·local_t,   P = Π_t γa_t
// i.e. e_t = γ·a_t·e_(t-1) + local_t per byte, independent of the window length.
__global__ void state_bwd_kernel(const bf16* dS, const float* S,
                                 const float* decay, float* dX, float* dDec,
                                 int B, int T, int D, const bf16* X,
                                 const float* initial, const float* gate, float* dGate,
                                 CellOpt o) {
    int b = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || d >= D) return;
    float future = 0, gd = 0, gg = 0;
    float suffix = 1, ld = 0, lg = 0;  // trace advance (suffix products)
    for (int t = T - 1; t >= 0; --t) {
        long idx = ((long)b * T + t) * D + d;
        float x = bf2f(X[idx]);
        float a = cell_a(o, decay[d], gate, d, x, b, T, t);
        float total = bf2f(dS[idx]) + future;
        float prev = t ? S[idx - D] : (initial ? initial[(long)b * D + d] : 0.f);
        float local = (prev - x) * a * (1.f - a);
        float gz = total * local;
        dX[idx] = total * (1.f - a) + (gate ? gz * gate[d] : 0.f);
        gd += gz; gg += gz * x;
        future = total * a;
        ld += suffix * local; lg += suffix * local * x;
        suffix *= o.gamma * a;
    }
    long bd = (long)b * D + d;
    if (o.trDec) {
        float e = o.trDec[bd], part = future * e;
        gd += part;
        if (o.logDec) atomicAdd(&o.logDec[d], part);
        o.trDec[bd] = suffix * e + ld;
    }
    if (o.trGate && gate) {
        float e = o.trGate[bd], part = future * e;
        gg += part;
        if (o.logGate) atomicAdd(&o.logGate[d], part);
        o.trGate[bd] = suffix * e + lg;
    }
    if (o.lam) { o.lam[bd] = future; o.prod[bd] = suffix; }
    atomicAdd(&dDec[d], gd);
    if (dGate && gate) atomicAdd(&dGate[d], gg);
}

inline void cell_backward(const bf16* dS, const float* S,
                          const float* decay, float* dX, float* dDec,
                          int B, int T, int D, const bf16* X,
                          const float* initial, const float* gate, float* dGate,
                          CellOpt o = CellOpt()) {
    dim3 grid(B, (D + 255) / 256);
    CUDA_CHECK(cudaMemset(dDec, 0, (size_t)D * 4));
    if (dGate) CUDA_CHECK(cudaMemset(dGate, 0, (size_t)D * 4));
    state_bwd_kernel<<<grid, 256>>>(dS, S, decay, dX, dDec, B, T, D,
                                   X, initial, gate, dGate, o);
}

// Embedding trace, e[b][r][d] = ds_d/dEmb[r][d], for a layer whose input
// contains the embedding row of each byte. Uses λ and P from state_bwd_kernel.
// One thread owns column d of stream b, so trace updates need no atomics.
__global__ void emb_trace_kernel(const bf16* X, const float* S, const float* initial,
                                 const float* decay, const float* gate, float* trEmb,
                                 float* dEmb, int B, int T, int D, CellOpt o) {
    int b = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || d >= D) return;
    long bd = (long)b * D + d;
    float l = o.lam[bd], P = o.prod[bd];
    float* e = trEmb + (long)b * 256 * D + d;
    for (int r = 0; r < 256; ++r) {
        float v = e[(long)r * D];
        if (v != 0.f) {
            atomicAdd(&dEmb[(long)r * D + d], l * v);
            if (o.logEmb) atomicAdd(&o.logEmb[(long)r * D + d], l * v);
        }
        e[(long)r * D] = P * v;
    }
    float suffix = 1;
    for (int t = T - 1; t >= 0; --t) {
        long idx = ((long)b * T + t) * D + d;
        float x = bf2f(X[idx]);
        float a = cell_a(o, decay[d], gate, d, x, b, T, t);
        float prev = t ? S[idx - D] : initial[bd];
        float k = (1.f - a) + (gate ? (prev - x) * a * (1.f - a) * gate[d] : 0.f);
        e[(long)o.ids[b * T + t] * D] += suffix * k;
        suffix *= o.gamma * a;
    }
}

inline void emb_trace(const bf16* X, const float* S, const float* initial,
                      const float* decay, const float* gate, float* trEmb,
                      float* dEmb, int B, int T, int D, CellOpt o) {
    dim3 grid(B, (D + 255) / 256);
    emb_trace_kernel<<<grid, 256>>>(X, S, initial, decay, gate, trEmb, dEmb, B, T, D, o);
}

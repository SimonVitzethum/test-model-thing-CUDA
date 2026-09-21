#pragma once
// Fact memory: cross-attention from the residual stream onto the bytes of
// retrieved facts (knowledge-graph memory, mem=1).
//
// Memory encoding (shared by all memory layers), per stream b and slot j:
//   e_j = E[byte_j] + Eprev[byte_(j-1)] + P[j]      (E = the model's byte embedding)
//   s_j = a_j s_(j-1) + (1 - a_j) e_j,  a_j = sigmoid(decay + gate * e_j)
//   m_j = e_j + s_j
// The previous-byte term lets attention find "the byte after the one just
// copied" (induction-style copying); the causal recurrence (the model's own
// gated cell, half-lives 1..64 bytes) tells each slot what precedes it, e.g.
// that "Asia" follows "continent: ". Slots with byte < 0 are padding and only
// occur after the valid prefix.
//
// Per memory layer, with x the residual stream (N = B*T rows):
//   h = LN(x); Q = h Wq^T; K = m Wk^T; V = m Wv^T
//   o = softmax(Q K^T / sqrt(dh)) V   over the valid slots of the same stream
//   x += o Wo^T                       (Wo starts at zero: memory off == base model)
// A stream without valid slots gets o = 0, so no memory is an exact no-op.
#include "common.h"
#include "linalg.cu"
#include "norm.cu"
#include "cell.cu"

struct MemLayer {
    int use = 0;
    size_t gamma, beta, wq, wk, wv, wo;  // parameter indices
    bf16 *Xsnap = nullptr, *Hn = nullptr, *Q = nullptr, *K = nullptr, *V = nullptr, *O = nullptr;
    float *mean = nullptr, *rstd = nullptr, *P = nullptr;  // P: (B,H,T,M) attention
};
struct MemShared {
    size_t eprev, pos;          // parameter indices (256,D), (M,D)
    size_t decay, gate;         // parameter indices (D), (D): encoder recurrence
    size_t rq = 0, rk = 0;      // stage-2 retrieval heads (mem_rdim, D); gradients from the host
    int* ids = nullptr;         // (B,M) memory bytes, -1 = padding (device)
    bf16* enc0 = nullptr;       // (B*M,D) e: byte/prev-byte/position encoding
    float* state = nullptr;     // (B*M,D) s: recurrence over the memory bytes
    float* dEnc0 = nullptr;     // (B*M,D) gradient through the recurrence
    bf16* enc = nullptr;        // (B*M,D) m = e + s
    bf16 *dO = nullptr, *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dHn = nullptr;
    bf16 *dXln = nullptr, *dEncB = nullptr;
    float *dS = nullptr, *dEnc = nullptr;  // (B,H,T,M), (B*M,D)
};

__global__ void mem_encode_kernel(const bf16* E, const bf16* Eprev, const bf16* P,
                                  const int* ids, bf16* enc, int B, int M, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)B * M * D) return;
    int d = i % D; long bj = i / D; int j = bj % M; int b = bj / M;
    int byte = ids[b * M + j];
    if (byte < 0) { enc[i] = f2bf(0.f); return; }
    int prev = j ? ids[b * M + j - 1] : -1;
    float v = bf2f(E[(long)byte * D + d]) + bf2f(P[(long)j * D + d]);
    if (prev >= 0) v += bf2f(Eprev[(long)prev * D + d]);
    enc[i] = f2bf(v);
}

// Scatter the encoding gradient into E, Eprev and P (padding skipped).
__global__ void mem_encode_bwd_kernel(const float* dEnc, const int* ids, float* dE,
                                      float* dEprev, float* dP, int B, int M, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)B * M * D) return;
    int d = i % D; long bj = i / D; int j = bj % M; int b = bj / M;
    int byte = ids[b * M + j];
    if (byte < 0) return;
    float g = dEnc[i];
    atomicAdd(&dE[(long)byte * D + d], g);
    atomicAdd(&dP[(long)j * D + d], g);
    int prev = j ? ids[b * M + j - 1] : -1;
    if (prev >= 0) atomicAdd(&dEprev[(long)prev * D + d], g);
}

// One thread per (b, h, t): masked softmax over the M slots and o = P V.
__global__ void mem_attn_fwd_kernel(const bf16* Q, const bf16* K, const bf16* V,
                                    const int* ids, float* P, bf16* O,
                                    int B, int T, int M, int H, int dh, float scale) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)B * H * T) return;
    int t = i % T; long bh = i / T; int h = bh % H; int b = bh / H;
    int HD = H * dh;
    const bf16* q = Q + ((long)b * T + t) * HD + h * dh;
    float* p = P + i * M;
    float mx = -1e30f;
    for (int j = 0; j < M; ++j) {
        if (ids[b * M + j] < 0) { p[j] = -1e30f; continue; }
        const bf16* k = K + ((long)b * M + j) * HD + h * dh;
        float s = 0;
        for (int d = 0; d < dh; ++d) s += bf2f(q[d]) * bf2f(k[d]);
        p[j] = s * scale; mx = fmaxf(mx, p[j]);
    }
    float sum = 0;
    for (int j = 0; j < M; ++j) { p[j] = p[j] > -1e29f ? expf(p[j] - mx) : 0.f; sum += p[j]; }
    float inv = sum > 0 ? 1.f / sum : 0.f;
    for (int j = 0; j < M; ++j) p[j] *= inv;
    bf16* o = O + ((long)b * T + t) * HD + h * dh;
    for (int d = 0; d < dh; ++d) {
        float acc = 0;
        for (int j = 0; j < M; ++j) if (p[j] != 0.f) acc += p[j] * bf2f(V[((long)b * M + j) * HD + h * dh + d]);
        o[d] = f2bf(acc);
    }
}

// Backward part 1, one thread per (b, h, t): dS = P (dP - <P, dP>), dQ.
__global__ void mem_attn_bwd_q_kernel(const bf16* dO, const bf16* K, const bf16* V,
                                      const float* P, float* dS, bf16* dQ,
                                      int B, int T, int M, int H, int dh, float scale) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)B * H * T) return;
    int t = i % T; long bh = i / T; int h = bh % H; int b = bh / H;
    int HD = H * dh;
    const bf16* g = dO + ((long)b * T + t) * HD + h * dh;
    const float* p = P + i * M;
    float* s = dS + i * M;
    float dot = 0;
    for (int j = 0; j < M; ++j) {
        float dp = 0;
        if (p[j] != 0.f) {
            const bf16* v = V + ((long)b * M + j) * HD + h * dh;
            for (int d = 0; d < dh; ++d) dp += bf2f(g[d]) * bf2f(v[d]);
        }
        s[j] = dp; dot += p[j] * dp;
    }
    for (int j = 0; j < M; ++j) s[j] = p[j] * (s[j] - dot);
    bf16* dq = dQ + ((long)b * T + t) * HD + h * dh;
    for (int d = 0; d < dh; ++d) {
        float acc = 0;
        for (int j = 0; j < M; ++j) if (s[j] != 0.f) acc += s[j] * bf2f(K[((long)b * M + j) * HD + h * dh + d]);
        dq[d] = f2bf(acc * scale);
    }
}

// Backward part 2, one thread per (b, h, j): dK_j = scale Σ_t dS q_t, dV_j = Σ_t P dO_t.
__global__ void mem_attn_bwd_kv_kernel(const bf16* Q, const bf16* dO, const float* P,
                                       const float* dS, bf16* dK, bf16* dV,
                                       int B, int T, int M, int H, int dh, float scale) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)B * H * M) return;
    int j = i % M; long bh = i / M; int h = bh % H; int b = bh / H;
    int HD = H * dh;
    bf16* dk = dK + ((long)b * M + j) * HD + h * dh;
    bf16* dv = dV + ((long)b * M + j) * HD + h * dh;
    for (int d = 0; d < dh; ++d) {
        float ak = 0, av = 0;
        for (int t = 0; t < T; ++t) {
            long pt = (bh * T + t) * M + j;
            long qt = ((long)b * T + t) * HD + h * dh + d;
            ak += dS[pt] * bf2f(Q[qt]);
            av += P[pt] * bf2f(dO[qt]);
        }
        dk[d] = f2bf(ak * scale); dv[d] = f2bf(av);
    }
}

__global__ void mem_add_bf16_kernel(bf16* a, const bf16* b, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = f2bf(bf2f(a[i]) + bf2f(b[i]));
}
__global__ void mem_cast_add_kernel(const bf16* s, float* d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] += bf2f(s[i]);
}

static inline int mem_blocks(long n) { return (int)((n + 255) / 256); }

__global__ void mem_sum_kernel(const bf16* e, const float* s, bf16* out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f2bf(bf2f(e[i]) + s[i]);
}
__global__ void mem_f32_to_bf16_kernel(const float* s, bf16* d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = f2bf(s[i]);
}
__global__ void mem_add_f32_kernel(float* a, const float* b, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

inline void mem_encode(const bf16* E, const bf16* Eprev, const bf16* P, const float* decay,
                       const float* gate, MemShared& s, int B, int M, int D) {
    long n = (long)B * M * D;
    mem_encode_kernel<<<mem_blocks(n), 256>>>(E, Eprev, P, s.ids, s.enc0, B, M, D);
    state_forward(s.enc0, s.state, decay, nullptr, B, M, D, gate);
    mem_sum_kernel<<<mem_blocks(n), 256>>>(s.enc0, s.state, s.enc, n);
}

// x (N,D) is updated in place: x += o Wo^T. Wq/Wk/Wv/Wo are bf16 work copies.
inline void mem_forward(bf16* X, MemLayer& L, MemShared& s, const float* gamma, const float* beta,
                        const bf16* Wq, const bf16* Wk, const bf16* Wv, const bf16* Wo, bf16* Y,
                        int B, int T, int M, int H, int dh, int D) {
    int N = B * T, HD = H * dh;
    long ND = (long)N * D;
    CUDA_CHECK(cudaMemcpy(L.Xsnap, X, ND * 2, cudaMemcpyDeviceToDevice));
    layernorm_fwd(X, gamma, beta, L.Hn, L.mean, L.rstd, N, D);
    linear_fwd(N, HD, D, L.Hn, Wq, L.Q);
    linear_fwd(B * M, HD, D, s.enc, Wk, L.K);
    linear_fwd(B * M, HD, D, s.enc, Wv, L.V);
    mem_attn_fwd_kernel<<<mem_blocks((long)B * H * T), 256>>>(L.Q, L.K, L.V, s.ids, L.P, L.O,
                                                              B, T, M, H, dh, 1.f / sqrtf((float)dh));
    linear_fwd(N, D, HD, L.O, Wo, Y);
    mem_add_bf16_kernel<<<mem_blocks(ND), 256>>>(X, Y, ND);
}

// dX (N,D) holds dL/dx_out and is updated in place to dL/dx_in (residual kept).
// Gradients of this layer's parameters are written (not accumulated); the
// encoding gradient is accumulated into s.dEnc (fp32).
inline void mem_backward(bf16* dX, MemLayer& L, MemShared& s, const float* gamma,
                         const bf16* Wq, const bf16* Wk, const bf16* Wv, const bf16* Wo,
                         float* gGamma, float* gBeta, float* gWq, float* gWk, float* gWv, float* gWo,
                         int B, int T, int M, int H, int dh, int D) {
    int N = B * T, HD = H * dh;
    long ND = (long)N * D;
    float scale = 1.f / sqrtf((float)dh);
    linear_dW(N, D, HD, dX, L.O, gWo);
    linear_dX(N, D, HD, dX, Wo, s.dO);
    mem_attn_bwd_q_kernel<<<mem_blocks((long)B * H * T), 256>>>(s.dO, L.K, L.V, L.P, s.dS, s.dQ,
                                                                B, T, M, H, dh, scale);
    mem_attn_bwd_kv_kernel<<<mem_blocks((long)B * H * M), 256>>>(L.Q, s.dO, L.P, s.dS, s.dK, s.dV,
                                                                 B, T, M, H, dh, scale);
    linear_dW(N, HD, D, s.dQ, L.Hn, gWq);
    linear_dX(N, HD, D, s.dQ, Wq, s.dHn);
    linear_dW(B * M, HD, D, s.dK, s.enc, gWk);
    linear_dW(B * M, HD, D, s.dV, s.enc, gWv);
    linear_dX(B * M, HD, D, s.dK, Wk, s.dEncB);
    linear_dX(B * M, HD, D, s.dV, Wv, s.dEncB, 1.f);
    mem_cast_add_kernel<<<mem_blocks((long)B * M * D), 256>>>(s.dEncB, s.dEnc, (long)B * M * D);
    layernorm_bwd(L.Xsnap, s.dHn, gamma, L.mean, L.rstd, s.dXln, gGamma, gBeta, N, D);
    mem_add_bf16_kernel<<<mem_blocks(ND), 256>>>(dX, s.dXln, ND);
}

// s.dEnc holds dL/dm (all memory layers). m = e + s, so e gets dL/dm directly
// plus the gradient through the recurrence; then it is scattered into E/Eprev/P.
inline void mem_encode_backward(MemShared& s, const float* decay, const float* gate, float* dDecay,
                                float* dGate, float* dE, float* dEprev, float* dP, int B, int M, int D) {
    long n = (long)B * M * D;
    mem_f32_to_bf16_kernel<<<mem_blocks(n), 256>>>(s.dEnc, s.dEncB, n);
    cell_backward(s.dEncB, s.state, decay, s.dEnc0, dDecay, B, M, D, s.enc0, nullptr, gate, dGate);
    mem_add_f32_kernel<<<mem_blocks(n), 256>>>(s.dEnc, s.dEnc0, n);
    mem_encode_bwd_kernel<<<mem_blocks(n), 256>>>(s.dEnc, s.ids, dE, dEprev, dP, B, M, D);
}

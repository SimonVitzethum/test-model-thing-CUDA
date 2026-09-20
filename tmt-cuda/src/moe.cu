#pragma once
// MoE Top-k mit Token-Dispatch (nur selektierte Experten rechnen).
// N = B*T Positionen, E Experten, k Top-k. Matmulen via cuBLAS (linalg.cu).
// Router-Logits fp32 (direkt aus GEMM, keine Rundung), Rest bf16/fp32 gemischt.
#include "util.h"
#include "linalg.cu"
// Fused Router+Topk: ein Block pro Zeile (64 Threads), 8 Experten-Dots
// parallel (je 1 Thread), dann Softmax/Topk. Volle Occupancy statt
// 2048 Einzel-Threads. Wr (20KB) L2-resident.
__global__ void router_topk_kernel(const bf16* X, const bf16* Wr,
                                   float* logits, float* probs, int* idx,
                                   float* w, int N, int E, int K, int D) {
    int r = blockIdx.x;
    if (r >= N) return;
    int te = threadIdx.x;  // 0..63
    int e = te / 8, c = te % 8;  // 8 Threads pro Experte, je 1/8 des Dots
    __shared__ float lp[16];
    const __nv_bfloat162* x =
        (const __nv_bfloat162*)(X + (long)r * D);
    float acc = 0;
    if (e < E) {
        const __nv_bfloat162* we =
            (const __nv_bfloat162*)(Wr + (long)e * D);
        int D2 = D / 2;
        for (int d = c; d < D2; d += 8) {
            float2 a = __bfloat1622float2(x[d]);
            float2 b = __bfloat1622float2(we[d]);
            acc += a.x * b.x + a.y * b.y;
        }
        if ((D & 1) && c == 0)
            acc += bf2f(X[(long)r * D + D - 1]) * bf2f(Wr[(long)e * D + D - 1]);
        unsigned mask = 0xFFu << (e * 8);
        for (int o = 4; o > 0; o >>= 1)
            acc += __shfl_down_sync(mask, acc, o);
        if (c == 0) {
            lp[e] = acc;
            logits[r * E + e] = acc;
        }
    }
    __syncthreads();
    if (te < E) {
        float mx = lp[0];
        for (int e = 1; e < E; ++e) mx = fmaxf(mx, lp[e]);
        float se = 0;
        for (int e = 0; e < E; ++e) se += expf(lp[e] - mx);
        // Softmax schreiben: jeder Thread seinen Experten (koalesziert? E
        // konsekutiv pro Zeile -> benachbarte Threads = benachbarte e ✓)
        probs[r * E + te] = expf(lp[te] - mx) / se;
    }
    __syncthreads();
    // Top-k sequenziell in Thread 0..K (K klein, trivial)
    if (te == 0) {
        for (int j = 0; j < K; ++j) {
            int best = -1;
            float bv = -1e30f;
            for (int e = 0; e < E; ++e) {
                bool used = false;
                for (int q = 0; q < j; ++q)
                    if (idx[r * K + q] == e) used = true;
                if (!used && probs[r * E + e] > bv) { bv = probs[r * E + e]; best = e; }
            }
            idx[r * K + j] = best;
        }
        float z = 0;
        for (int j = 0; j < K; ++j) z += probs[r * E + idx[r * K + j]];
        for (int j = 0; j < K; ++j) w[r * K + j] = probs[r * E + idx[r * K + j]] / z;
    }
}

// Top-k + renormierte Gewichte aus fp32-Logits(N,E). E klein -> Scan.
// (Nur noch für Referenz/Tests; Produktion nutzt router_topk_kernel.)
__global__ void topk_kernel(const float* logits, int* idx, float* w,
                            float* probs, int N, int E, int K) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    const float* L = logits + (long)r * E;
    float se = 0;
    float mx = L[0];
    for (int e = 1; e < E; ++e) mx = fmaxf(mx, L[e]);
    for (int e = 0; e < E; ++e) se += expf(L[e] - mx);
    for (int e = 0; e < E; ++e) probs[r * E + e] = expf(L[e] - mx) / se;
    for (int j = 0; j < K; ++j) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < E; ++e) {
            bool used = false;
            for (int q = 0; q < j; ++q)
                if (idx[r * K + q] == e) used = true;
            if (!used && probs[r * E + e] > bv) { bv = probs[r * E + e]; best = e; }
        }
        idx[r * K + j] = best;
    }
    float z = 0;
    for (int j = 0; j < K; ++j) z += probs[r * E + idx[r * K + j]];
    for (int j = 0; j < K; ++j) w[r * K + j] = probs[r * E + idx[r * K + j]] / z;
}

__global__ void count_kernel(const int* idx, int* counts, int N, int K) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    for (int j = 0; j < K; ++j) atomicAdd(&counts[idx[r * K + j]], 1);
}

__global__ void fill_kernel(const int* idx, const float* w, int* cursor,
                            int* perm, float* slotw, int* slot_of,
                            int N, int K) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    for (int j = 0; j < K; ++j) {
        int e = idx[r * K + j];
        int pos = atomicAdd(&cursor[e], 1);
        perm[pos] = r;
        slotw[pos] = w[r * K + j];
        slot_of[r * K + j] = pos;
    }
}

__global__ void gather_kernel(const bf16* X, const int* perm, bf16* Xg,
                              int TK, int D) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= TK) return;
    int r = perm[pos];
    const bf16* s = X + (long)r * D;
    bf16* d = Xg + (long)pos * D;
    for (int dd = 0; dd < D; ++dd) d[dd] = s[dd];
}

// Slot-basiert (für Padding-Layout): Xg[slot_of[r][j]] = X[r].
// Elementweise: ein Thread pro Element, Warps immer in einer Zeile
// (D=Vielfaches von 32 vorausgesetzt, sonst Tail-Guard).
__global__ void gather_slot_kernel(const bf16* X, const int* slot_of,
                                   bf16* Xg, int N, int K, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)N * K * D;
    if (i >= n) return;
    long dd = i % D;
    long tmp = i / D;
    int j = tmp % K, r = tmp / K;
    long pos = slot_of[r * K + j];
    Xg[pos * D + dd] = X[(long)r * D + dd];
}

// y[r] = beta*y[r] + sum_j slotw*silu(Yg_pre[slot]), ein Thread pro Element.
// Mit beta=1 spart sich der Caller den separaten Residual-Add-Pass.
__global__ void combine_kernel(const bf16* Yg, const float* slotw,
                               const int* slot_of, bf16* Y, int N, int K,
                               int D, float beta = 0.0f) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)N * D;
    if (i >= n) return;
    long dd = i % D;
    long r = i / D;
    float acc = (beta != 0.0f) ? beta * bf2f(Y[i]) : 0.0f;
    for (int j = 0; j < K; ++j) {
        int pos = slot_of[r * K + j];
        acc += slotw[pos] * silu_f(bf2f(Yg[(long)pos * D + dd]));
    }
    Y[i] = f2bf(acc);
}

// dYg_pre[slot] = w*dy*silu'(pre); s_j = dot(dY[r], silu(pre)).
// Elementweise für dYg; s_j-Reduktion separat (klein).
__global__ void combine_bwd_kernel(const bf16* dY, const bf16* Yg,
                                   const float* slotw, const int* slot_of,
                                   bf16* dYg, float* s_j, int N, int K,
                                   int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)N * K * D;
    if (i >= n) return;
    long dd = i % D;
    long tmp = i / D;
    int j = tmp % K, r = tmp / K;
    int pos = slot_of[r * K + j];
    float dy = bf2f(dY[(long)r * D + dd]);
    float pre = bf2f(Yg[(long)pos * D + dd]);
    dYg[(long)pos * D + dd] = f2bf(silu_bwd(pre, slotw[pos] * dy));
}
// s_j[r][j] = dot(dY[r], silu(Yg[slot])): ein Block pro (r,j), Reduce über D.
__global__ void sdot_kernel(const bf16* dY, const bf16* Yg,
                            const int* slot_of, float* s_j, int N, int K,
                            int D) {
    int rj = blockIdx.x;
    if (rj >= N * K) return;
    int r = rj / K, j = rj % K;
    int pos = slot_of[r * K + j];
    __shared__ float buf[256];
    float acc = 0;
    for (int dd = threadIdx.x; dd < D; dd += blockDim.x)
        acc += bf2f(dY[(long)r * D + dd]) *
               silu_f(bf2f(Yg[(long)pos * D + dd]));
    buf[threadIdx.x] = acc;
    __syncthreads();
    for (int t = 128; t > 0; t >>= 1) {
        if (threadIdx.x < t) buf[threadIdx.x] += buf[threadIdx.x + t];
        __syncthreads();
    }
    if (threadIdx.x == 0) s_j[rj] = buf[0];
}

// Router-Backward durch Softmax+Topk+Renorm (s_j = dL/dw_j).
__global__ void router_bwd_kernel(const float* probs, const int* idx,
                                  const float* s_j, float* dlogits,
                                  int N, int E, int K, const float* logits,
                                  const int* counts, float aux, float zcoef) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    float Z = 0, sp = 0;
    for (int j = 0; j < K; ++j) {
        int e = idx[r * K + j];
        float p = probs[r * E + e];
        Z += p; sp += s_j[r * K + j] * p;
    }
    float Z2 = Z * Z + 1e-12f;
    for (int e = 0; e < E; ++e) dlogits[r * E + e] = 0;
    for (int j = 0; j < K; ++j) {
        int e = idx[r * K + j];
        dlogits[r * E + e] = (s_j[r * K + j] * Z - sp) / Z2;
    }
    float acc = 0;
    for (int e = 0; e < E; ++e) acc += dlogits[r * E + e] * probs[r * E + e];
    for (int e = 0; e < E; ++e)
        dlogits[r * E + e] = probs[r * E + e] * (dlogits[r * E + e] - acc);
    float expected = 0, mx = logits[r * E], sum = 0;
    for (int e = 0; e < E; ++e) {
        expected += probs[r * E + e] * counts[e] / (float)(N * K);
        mx = fmaxf(mx, logits[r * E + e]);
    }
    for (int e = 0; e < E; ++e) sum += expf(logits[r * E + e] - mx);
    float lse = mx + logf(sum);
    for (int e = 0; e < E; ++e) {
        float p = probs[r * E + e];
        dlogits[r * E + e] += aux * E / N * p * (counts[e] / (float)(N * K) - expected)
                             + zcoef * 2.f / N * lse * p;
    }
}

// dX[r] += sum_j dXg[slot] (bf16, elementweise, keine Atomics;
// reihenfolgenunabhängig durch Read-Modify-Write pro Element).
__global__ void scatter_add_kernel(const bf16* dXg, const int* slot_of,
                                   bf16* dX, int N, int K, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)N * D;
    if (i >= n) return;
    long dd = i % D;
    long r = i / D;
    float acc = bf2f(dX[i]);
    for (int j = 0; j < K; ++j)
        acc += bf2f(dXg[(long)slot_of[r * K + j] * D + dd]);
    dX[i] = f2bf(acc);
}

// fp32 -> bf16 Cast.
__global__ void to_bf16_kernel(const float* s, bf16* d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = f2bf(s[i]);
}

// Switch-Aux: E*sum(mean_p*frac); fragt counts + summierte probs.
__global__ void aux_sum_kernel(const float* probs, float* sum_p,
                               int N, int E) {
    // ein Block, E Threads (keine Transzendenten hier)
    int e = threadIdx.x;
    if (e >= E) return;
    float acc = 0;
    for (int r = 0; r < N; ++r) acc += probs[(long)r * E + e];
    sum_p[e] = acc;
}

// z-Loss: mean(lse^2) über Zeilen, ein Thread pro Zeile + Atomik.
__global__ void zloss_kernel(const float* logits, float* out, int N, int E) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    float mx = logits[r * E];
    for (int j = 1; j < E; ++j) mx = fmaxf(mx, logits[r * E + j]);
    float sum = 0;
    for (int j = 0; j < E; ++j) sum += expf(logits[r * E + j] - mx);
    float lse = mx + logf(sum);
    atomicAdd(out, lse * lse / N);
}

// bf16 -> fp32 Cast (für Router-dW in fp32).
__global__ void to_f32_kernel(const bf16* s, float* d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = bf2f(s[i]);
}

__global__ void dense_activation_kernel(const bf16* pre, const bf16* grad,
                                         bf16* out, long n, float beta = 0.0f) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float base = (beta != 0.0f) ? beta * bf2f(out[i]) : 0.0f;
    out[i] = f2bf(base + (grad ? silu_bwd(bf2f(pre[i]), bf2f(grad[i])) : silu_f(bf2f(pre[i]))));
}

struct MoeKeep {  // pro Layer, persistent fwd->bwd (vorallokiert)
    float zloss = 0;
    float* logits = nullptr;  // (N,E) fp32 (z-Term im Backward)
    float* probs = nullptr;   // (N,E)
    int* idx = nullptr;       // (N,K)
    float* slotw = nullptr;   // (Tk)
    int* slot_of = nullptr;   // (N,K)
    int* counts = nullptr;    // (E, Device-Kopie für Router-Grad)
    int* perm = nullptr;      // (Tk) für Xg-Recompute im Backward
    bf16* H = nullptr;        // (N,D) MoE-Input (Router-dW via Cast)
    bf16* Yg = nullptr;       // (Tk,D) Experten-Pre-Akt (Combine-Bwd)
    float* Xf = nullptr;      // (N,D) fp32-Input (shared transient OK? nein:
                              //  pro Layer nötig -> Keep. Ersatz: H+Cast.)
    int hcnt[16] = {0};       // Host: Tokens pro Experte
    int hoff[16] = {0};       // Host: Offsets (echt, für Aux/Statistiken)
    int phoff[16] = {0};      // Host: Offsets mit Padding (GEMM-Layout)
    int N = 0, E = 0, K = 0, D = 0, Tk = 0, Ptk = 0;
};
struct MoeWs {  // shared transient (ein Satz reicht, Layer laufen sequenziell)
    float* w = nullptr;       // (N,K) Gewichte (nur fwd)
    int* offsets = nullptr;   // (E)
    int* cursor = nullptr;    // (E)
    float* sum_p = nullptr;   // (E+1)
    bf16* Xg = nullptr;       // (Tk,D) gruppiert (fwd+bwd recompute)
    bf16* dYg = nullptr;      // (Tk,D)
    float* s_j = nullptr;     // (N,K)
    float* dlogits = nullptr; // (N,E)
    bf16* dlog_b = nullptr;   // (N,E)
    bf16* dXg = nullptr;      // (Tk,D)
    float* Xf = nullptr;      // (N,D) fp32-Cast von H (Router-dW)
};

inline void moe_keep_alloc(MoeKeep& k, int N, int E, int K, int D) {
    int Tk = N * K + E * 128;  // + Padding-Slack (128er-Blöcke pro Experte)
    CUDA_CHECK(cudaMalloc(&k.logits, (size_t)N * E * 4));
    CUDA_CHECK(cudaMalloc(&k.probs, (size_t)N * E * 4));
    CUDA_CHECK(cudaMalloc(&k.idx, (size_t)N * K * 4));
    CUDA_CHECK(cudaMalloc(&k.slotw, (size_t)Tk * 4));
    CUDA_CHECK(cudaMalloc(&k.slot_of, (size_t)N * K * 4));
    CUDA_CHECK(cudaMalloc(&k.counts, (size_t)E * 4));
    CUDA_CHECK(cudaMalloc(&k.perm, (size_t)Tk * 4));
    CUDA_CHECK(cudaMalloc(&k.H, (size_t)N * D * 2));
    CUDA_CHECK(cudaMalloc(&k.Yg, (size_t)Tk * D * 2));
}
inline void moe_keep_free(MoeKeep& k) {
    for (void* p : {(void*)k.logits, (void*)k.probs, (void*)k.idx,
                    (void*)k.slotw, (void*)k.slot_of, (void*)k.counts,
                    (void*)k.perm, (void*)k.H, (void*)k.Yg})
        if (p) cudaFree(p);
    k = MoeKeep();
}
inline void moe_ws_alloc(MoeWs& w, int N, int E, int K, int D) {
    int Tk = N * K + E * 128;
    CUDA_CHECK(cudaMalloc(&w.w, (size_t)N * K * 4));
    CUDA_CHECK(cudaMalloc(&w.offsets, (size_t)E * 4));
    CUDA_CHECK(cudaMalloc(&w.cursor, (size_t)E * 4));
    CUDA_CHECK(cudaMalloc(&w.sum_p, (size_t)(E + 1) * 4));
    CUDA_CHECK(cudaMalloc(&w.Xg, (size_t)Tk * D * 2));
    CUDA_CHECK(cudaMalloc(&w.dYg, (size_t)Tk * D * 2));
    CUDA_CHECK(cudaMalloc(&w.s_j, (size_t)N * K * 4));
    CUDA_CHECK(cudaMalloc(&w.dlogits, (size_t)N * E * 4));
    CUDA_CHECK(cudaMalloc(&w.dlog_b, (size_t)N * E * 2));
    CUDA_CHECK(cudaMalloc(&w.dXg, (size_t)Tk * D * 2));
    CUDA_CHECK(cudaMalloc(&w.Xf, (size_t)N * D * 4));
}
inline void moe_ws_free(MoeWs& w) {
    for (void* p : {(void*)w.w, (void*)w.offsets, (void*)w.cursor,
                    (void*)w.sum_p, (void*)w.Xg, (void*)w.dYg, (void*)w.s_j,
                    (void*)w.dlogits, (void*)w.dlog_b, (void*)w.dXg,
                    (void*)w.Xf})
        if (p) cudaFree(p);
    w = MoeWs();
}

// Forward: X(N,D) bf16 -> Y(N,D) bf16. Keine Allokation (Arena).
// Gibt Switch-Aux (Host) zurück.
inline float moe_forward(const bf16* X, const bf16* Wrouter, bf16** Wexp,
                         bf16* Y, MoeKeep& k, MoeWs& w, int N, int E, int K,
                         int D, float beta = 0.0f) {
    k.N = N; k.E = E; k.K = K; k.D = D;
    const int TPB = 256;
    int blocks = (N + TPB - 1) / TPB;
    bool prof = getenv("TMT_PROFILE") != nullptr;
    struct timespec t0, t1;
    double acc[6] = {0};
    auto tick = [&](int i) {
        if (!prof) return;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        acc[i] += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        t0 = t1;
    };
    auto sync_tick = [&](int i) {
        if (!prof) return;
        CUDA_CHECK(cudaDeviceSynchronize());
        tick(i);
    };
    if (prof) clock_gettime(CLOCK_MONOTONIC, &t0);
    long n = (long)N * D;
    CUDA_CHECK(cudaMemcpy(k.H, X, n * 2, cudaMemcpyDeviceToDevice));
    if (E == 1) {
        linear_fwd(N, D, D, X, Wexp[0], k.Yg);
        dense_activation_kernel<<<(n + 255) / 256, 256>>>(k.Yg, nullptr, Y, n,
                                                          beta);
        k.Tk = N;
        return 0;
    }
    // Router fused (statt m=8-GEMM + Topk): ein Block (64 Thr.) pro Zeile.
    router_topk_kernel<<<N, 64>>>(X, Wrouter, k.logits, k.probs, k.idx,
                                  w.w, N, E, K, D);
    CUDA_CHECK(cudaMemset(k.counts, 0, (size_t)E * 4));  // Arena persistiert!
    count_kernel<<<blocks, TPB>>>(k.idx, k.counts, N, K);
    sync_tick(1);
    // Offsets per Host (E klein, kein Thrust nötig); 1 Sync pro Call.
    CUDA_CHECK(cudaMemcpy(k.hcnt, k.counts, (size_t)E * 4, cudaMemcpyDeviceToHost));
    int tk = 0, ptk = 0;
    for (int e = 0; e < E; ++e) { k.hoff[e] = tk; tk += k.hcnt[e]; }
    // Padding auf 128er-Blöcke: stabile GEMM-Formen -> cuBLAS-Cache-Hits.
    // (Pad-Zeilen werden genullt, combine/scatter lesen nur echte Slots.)
    for (int e = 0; e < E; ++e) {
        k.phoff[e] = ptk;
        ptk += (k.hcnt[e] + 127) / 128 * 128;
    }
    k.Tk = tk;
    k.Ptk = ptk;
    CUDA_CHECK(cudaMemcpy(w.cursor, k.phoff, (size_t)E * 4, cudaMemcpyHostToDevice));
    fill_kernel<<<blocks, TPB>>>(k.idx, w.w, w.cursor, k.perm, k.slotw,
                                 k.slot_of, N, K);
    CUDA_CHECK(cudaMemset(w.Xg, 0, (size_t)(ptk > 0 ? ptk : 1) * D * 2));
    sync_tick(2);
    if (tk > 0) {
        long gblocks = ((long)N * K * D + TPB - 1) / TPB;
        gather_slot_kernel<<<gblocks, TPB>>>(X, k.slot_of, w.Xg, N, K, D);
        sync_tick(3);
        for (int e = 0; e < E; ++e) {
            int pm = (k.hcnt[e] + 127) / 128 * 128;
            if (pm == 0) continue;
            linear_fwd(pm, D, D, w.Xg + (long)k.phoff[e] * D, Wexp[e],
                       k.Yg + (long)k.phoff[e] * D);
        }
        sync_tick(4);
    }
    {
        long cblocks = ((long)N * D + TPB - 1) / TPB;
        combine_kernel<<<cblocks, TPB>>>(k.Yg, k.slotw, k.slot_of, Y, N, K, D,
                                         beta);
    }
    aux_sum_kernel<<<1, E>>>(k.probs, w.sum_p, N, E);
    CUDA_CHECK(cudaMemset(w.sum_p + E, 0, 4));
    zloss_kernel<<<(N + TPB - 1) / TPB, TPB>>>(k.logits, w.sum_p + E, N, E);
    sync_tick(5);
    // Switch-Aux auf Host (gleicher Sync wie oben nutzbar, hier separat)
    float hsum[17];
    CUDA_CHECK(cudaMemcpy(hsum, w.sum_p, (size_t)(E + 1) * 4, cudaMemcpyDeviceToHost));
    k.zloss = hsum[E];
    float aux = 0;
    for (int e = 0; e < E; ++e)
        aux += (hsum[e] / N) * (k.hcnt[e] / (float)(N * K));
    if (prof) {
        std::printf("  [moe-fwd] router=%.3f dispatch=%.3f gather=%.3f experts=%.3f tail=%.3f sum=%.3f\n",
                    acc[1], acc[2], acc[3], acc[4], acc[5],
                    acc[1] + acc[2] + acc[3] + acc[4] + acc[5]);
        fflush(stdout);
    }
    return E * aux;
}

// Backward: dY(N,D) bf16 -> dX(N,D) bf16 (genullt+gesetzt),
// dWexp[E] fp32, dWrouter(E,D) fp32. Keine Allokation.
inline void moe_backward(const bf16* dY, bf16** Wexp,
                         const bf16* Wrouter, float* dWrouter,
                         float** dWexp, bf16* dX, MoeKeep& k, MoeWs& w,
                         float aux = 0.f, float zcoef = 0.f) {
    int N = k.N, E = k.E, K = k.K, D = k.D, tk = k.Tk;
    const int TPB = 256;
    int blocks = (N + TPB - 1) / TPB;
    CUDA_CHECK(cudaMemset(dX, 0, (size_t)N * D * 2));
    CUDA_CHECK(cudaMemset(dWrouter, 0, (size_t)E * D * 4));
    if (E == 1) {
        long n = (long)N * D;
        dense_activation_kernel<<<(n + 255) / 256, 256>>>(k.Yg, dY, w.dXg, n);
        linear_dW(N, D, D, w.dXg, k.H, dWexp[0]);
        linear_dX(N, D, D, w.dXg, Wexp[0], dX);
        return;
    }
    if (tk == 0) return;
    // Xg-Recompute aus Keep.H (Pads nullen für saubere dW)
    {
        CUDA_CHECK(cudaMemset(w.Xg, 0, (size_t)(k.Ptk > 0 ? k.Ptk : 1) * D * 2));
        int gblocks = (N * K + TPB - 1) / TPB;
        gather_slot_kernel<<<gblocks, TPB>>>(k.H, k.slot_of, w.Xg, N, K, D);
    }
    {
        long n = (long)N * K * D;
        combine_bwd_kernel<<<(n + TPB - 1) / TPB, TPB>>>(
            dY, k.Yg, k.slotw, k.slot_of, w.dYg, w.s_j, N, K, D);
        sdot_kernel<<<N * K, TPB>>>(dY, k.Yg, k.slot_of, w.s_j, N, K, D);
    }
    // Router: dlogits(N,E) fp32 -> dWrouter(E,D) fp32 via GEMM
    router_bwd_kernel<<<blocks, TPB>>>(k.probs, k.idx, w.s_j, w.dlogits, N, E, K,
                                       k.logits, k.counts, aux, zcoef);
    // Router: dlogits(N,E) -> dWrouter(E,D) via H (bf16, kein Xf-Cast).
    // dW = dlog^T @ H (linear_dW-Muster).
    {
        long nn = (long)N * E;
        to_bf16_kernel<<<(nn + TPB - 1) / TPB, TPB>>>(w.dlogits, w.dlog_b, nn);
    }
    linear_dW(N, E, D, w.dlog_b, k.H, dWrouter);
    // Router-Inputgrad: dX += dlogits_b @ Wrouter (beta=1, dlog_b schon da)
    {
        float al = 1.0f, be = 1.0f;
        cublasStatus_t s2 = cublasGemmEx(
            cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, D, N, E, &al,
            Wrouter, CUDA_R_16BF, D, w.dlog_b, CUDA_R_16BF, E, &be, dX,
            CUDA_R_16BF, D,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (s2 != CUBLAS_STATUS_SUCCESS) { std::printf("router dX FAIL\n"); exit(1); }
    }
    // Experten pro Gruppe (gepaddete Formen -> stabile cuBLAS-Kernels)
    CUDA_CHECK(cudaMemset(w.dYg, 0, (size_t)(k.Ptk > 0 ? k.Ptk : 1) * D * 2));
    for (int e = 0; e < E; ++e) {
        int pm = (k.hcnt[e] + 127) / 128 * 128;
        if (pm == 0) continue;
        long off = (long)k.phoff[e] * D;
        linear_dW(pm, D, D, w.dYg + off, w.Xg + off, dWexp[e]);
        linear_dX(pm, D, D, w.dYg + off, Wexp[e], w.dXg + off);
    }
    {
        long n = (long)N * D;
        scatter_add_kernel<<<(n + TPB - 1) / TPB, TPB>>>(w.dXg, k.slot_of, dX,
                                                        N, K, D);
    }
}

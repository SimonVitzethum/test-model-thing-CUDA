#pragma once
// MoE Top-k mit Token-Dispatch (nur selektierte Experten rechnen).
// N = B*T Positionen, E Experten, k Top-k. Matmulen via cuBLAS (linalg.cu).
// Router-Logits fp32 (direkt aus GEMM, keine Rundung), Rest bf16/fp32 gemischt.
#include "util.h"
#include "linalg.cu"

// Top-k + renormierte Gewichte aus fp32-Logits(N,E). E klein -> Scan.
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

// y[r] = sum_j slotw*silu(Yg_pre[slot]), ein Thread pro r (kein Race).
// Yg hält Pre-Aktivierungen (kein Extraspeicher für silu nötig).
__global__ void combine_kernel(const bf16* Yg, const float* slotw,
                               const int* slot_of, bf16* Y, int N, int K,
                               int D) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    for (int dd = 0; dd < D; ++dd) {
        float acc = 0;
        for (int j = 0; j < K; ++j) {
            int pos = slot_of[r * K + j];
            acc += slotw[pos] * silu_f(bf2f(Yg[(long)pos * D + dd]));
        }
        Y[(long)r * D + dd] = f2bf(acc);
    }
}

// dYg_pre[slot] = w*dy*silu'(pre); s_j = dot(dY[r], silu(pre)).
__global__ void combine_bwd_kernel(const bf16* dY, const bf16* Yg,
                                   const float* slotw, const int* slot_of,
                                   bf16* dYg, float* s_j, int N, int K,
                                   int D) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    for (int j = 0; j < K; ++j) {
        int pos = slot_of[r * K + j];
        float dot = 0;
        for (int dd = 0; dd < D; ++dd) {
            float dy = bf2f(dY[(long)r * D + dd]);
            float pre = bf2f(Yg[(long)pos * D + dd]);
            dot += dy * silu_f(pre);
            dYg[(long)pos * D + dd] = f2bf(silu_bwd(pre, slotw[pos] * dy));
        }
        s_j[r * K + j] = dot;
    }
}

// Router-Backward durch Softmax+Topk+Renorm (s_j = dL/dw_j).
__global__ void router_bwd_kernel(const float* probs, const int* idx,
                                  const float* s_j, float* dlogits,
                                  int N, int E, int K) {
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
}

// dX[r] += sum_j dXg[slot] (bf16, ein Thread pro r, keine Atomics;
// reihenfolgenunabhängig durch Read-Modify-Write).
__global__ void scatter_add_kernel(const bf16* dXg, const int* slot_of,
                                   bf16* dX, int N, int K, int D) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    for (int dd = 0; dd < D; ++dd) {
        float acc = bf2f(dX[(long)r * D + dd]);
        for (int j = 0; j < K; ++j)
            acc += bf2f(dXg[(long)slot_of[r * K + j] * D + dd]);
        dX[(long)r * D + dd] = f2bf(acc);
    }
}

// fp32 -> bf16 Cast.
__global__ void to_bf16_kernel(const float* s, bf16* d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = f2bf(s[i]);
}

// Switch-Aux: E*sum(mean_p*frac); fragt counts + summierte probs.
__global__ void aux_sum_kernel(const float* probs, float* sum_p,
                               int N, int E) {
    // ein Block, E Threads
    int e = threadIdx.x;
    if (e >= E) return;
    float acc = 0;
    for (int r = 0; r < N; ++r) acc += probs[(long)r * E + e];
    sum_p[e] = acc;
}

// bf16 -> fp32 Cast (für Router-dW in fp32).
__global__ void to_f32_kernel(const bf16* s, float* d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = bf2f(s[i]);
}

struct MoeCache {
    float* logits = nullptr; // (N,E) fp32
    int* idx = nullptr;      // (N,K)
    float* w = nullptr;      // (N,K)
    float* probs = nullptr;  // (N,E)
    int* perm = nullptr;     // (Tk)
    float* slotw = nullptr;  // (Tk)
    int* slot_of = nullptr;  // (N,K)
    int* counts = nullptr;   // (E)
    int* offsets = nullptr;  // (E)
    bf16* Xg = nullptr;      // (Tk,D)
    bf16* Yg = nullptr;      // (Tk,D)
    float* Xf = nullptr;     // (N,D) fp32-Kopie des Inputs (Router-dW)
    float* sum_p = nullptr;  // (E) für Aux
    int hcnt[16] = {0};      // Host: Tokens pro Experte
    int hoff[16] = {0};      // Host: Offsets
    int N = 0, E = 0, K = 0, D = 0, Tk = 0;
};

inline void moe_cache_free(MoeCache& c) {
    for (void* p : {(void*)c.logits, (void*)c.idx, (void*)c.w, (void*)c.probs,
                    (void*)c.perm, (void*)c.slotw, (void*)c.slot_of,
                    (void*)c.counts, (void*)c.offsets, (void*)c.Xg,
                    (void*)c.Yg, (void*)c.Xf, (void*)c.sum_p})
        if (p) cudaFree(p);
    c = MoeCache();
}

// Forward: X(N,D) bf16 -> Y(N,D) bf16. Router W(E,D) bf16, Experten Wx(D,D).
// Gibt Switch-Aux (Host) zurück.
inline float moe_forward(const bf16* X, const bf16* Wrouter, bf16** Wexp,
                         bf16* Y, MoeCache& c, int N, int E, int K, int D) {
    c.N = N; c.E = E; c.K = K; c.D = D;
    CUDA_CHECK(cudaMalloc(&c.logits, (size_t)N * E * 4));
    CUDA_CHECK(cudaMalloc(&c.idx, (size_t)N * K * 4));
    CUDA_CHECK(cudaMalloc(&c.w, (size_t)N * K * 4));
    CUDA_CHECK(cudaMalloc(&c.probs, (size_t)N * E * 4));
    CUDA_CHECK(cudaMalloc(&c.slot_of, (size_t)N * K * 4));
    CUDA_CHECK(cudaMalloc(&c.counts, (size_t)E * 4));
    CUDA_CHECK(cudaMalloc(&c.offsets, (size_t)E * 4));
    CUDA_CHECK(cudaMalloc(&c.sum_p, (size_t)E * 4));
    CUDA_CHECK(cudaMemset(c.counts, 0, (size_t)E * 4));
    const int TPB = 256;
    int blocks = (N + TPB - 1) / TPB;
    // Router: logits^T(E,N) = Wrouter(E,D) @ Xt(D,N), fp32 out
    {
        float al = 1, be = 0;
        cublasStatus_t s = cublasGemmEx(
            cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, E, N, D, &al,
            Wrouter, CUDA_R_16BF, E, X, CUDA_R_16BF, D, &be,
            c.logits, CUDA_R_32F, E,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (s != CUBLAS_STATUS_SUCCESS) { std::printf("router GEMM FAIL\n"); exit(1); }
    }
    topk_kernel<<<blocks, TPB>>>(c.logits, c.idx, c.w, c.probs, N, E, K);
    count_kernel<<<blocks, TPB>>>(c.idx, c.counts, N, K);
    // Offsets per Host (E klein, kein Thrust nötig)
    CUDA_CHECK(cudaMemcpy(c.hcnt, c.counts, (size_t)E * 4, cudaMemcpyDeviceToHost));
    int tk = 0;
    for (int e = 0; e < E; ++e) { c.hoff[e] = tk; tk += c.hcnt[e]; }
    c.Tk = tk;
    CUDA_CHECK(cudaMemcpy(c.offsets, c.hoff, (size_t)E * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&c.perm, (size_t)(tk > 0 ? tk : 1) * 4));
    CUDA_CHECK(cudaMalloc(&c.slotw, (size_t)(tk > 0 ? tk : 1) * 4));
    CUDA_CHECK(cudaMalloc(&c.Xg, (size_t)(tk > 0 ? tk : 1) * D * 2));
    CUDA_CHECK(cudaMalloc(&c.Yg, (size_t)(tk > 0 ? tk : 1) * D * 2));
    // cursor = Kopie der Offsets
    int* cursor;
    CUDA_CHECK(cudaMalloc(&cursor, (size_t)E * 4));
    CUDA_CHECK(cudaMemcpy(cursor, c.hoff, (size_t)E * 4, cudaMemcpyHostToDevice));
    fill_kernel<<<blocks, TPB>>>(c.idx, c.w, cursor, c.perm, c.slotw,
                                 c.slot_of, N, K);
    CUDA_CHECK(cudaFree(cursor));
    // fp32-Inputkopie für Router-dW
    CUDA_CHECK(cudaMalloc(&c.Xf, (size_t)N * D * 4));
    {
        long n = (long)N * D;
        to_f32_kernel<<<(n + TPB - 1) / TPB, TPB>>>(X, c.Xf, n);
    }
    if (tk > 0) {
        int gblocks = (tk + TPB - 1) / TPB;
        gather_kernel<<<gblocks, TPB>>>(X, c.perm, c.Xg, tk, D);
        for (int e = 0; e < E; ++e) {
            if (c.hcnt[e] == 0) continue;
            linear_fwd(c.hcnt[e], D, D, c.Xg + (long)c.hoff[e] * D, Wexp[e],
                       c.Yg + (long)c.hoff[e] * D);
        }
    }
    combine_kernel<<<blocks, TPB>>>(c.Yg, c.slotw, c.slot_of, Y, N, K, D);
    aux_sum_kernel<<<1, E>>>(c.probs, c.sum_p, N, E);
    // Switch-Aux auf Host
    float hsum[16];
    CUDA_CHECK(cudaMemcpy(hsum, c.sum_p, (size_t)E * 4, cudaMemcpyDeviceToHost));
    float aux = 0;
    for (int e = 0; e < E; ++e)
        aux += (hsum[e] / N) * (c.hcnt[e] / (float)N);
    return E * aux;
}

// Backward: dY(N,D) bf16 -> dX(N,D) bf16 (wird genullt+gesetzt),
// dWexp[E] fp32, dWrouter(E,D) fp32.
inline void moe_backward(const bf16* dY, bf16** Wexp,
                         const bf16* Wrouter, float* dWrouter,
                         float** dWexp, bf16* dX, MoeCache& c) {
    int N = c.N, E = c.E, K = c.K, D = c.D, tk = c.Tk;
    const int TPB = 256;
    int blocks = (N + TPB - 1) / TPB;
    CUDA_CHECK(cudaMemset(dX, 0, (size_t)N * D * 2));
    CUDA_CHECK(cudaMemset(dWrouter, 0, (size_t)E * D * 4));
    if (tk == 0) return;
    bf16* dYg;
    float* s_j;
    CUDA_CHECK(cudaMalloc(&dYg, (size_t)tk * D * 2));
    CUDA_CHECK(cudaMalloc(&s_j, (size_t)N * K * 4));
    combine_bwd_kernel<<<blocks, TPB>>>(dY, c.Yg, c.slotw, c.slot_of, dYg,
                                        s_j, N, K, D);
    // Router: dlogits(N,E) fp32 -> dWrouter(E,D) fp32 via GEMM
    float* dlogits;
    CUDA_CHECK(cudaMalloc(&dlogits, (size_t)N * E * 4));
    router_bwd_kernel<<<blocks, TPB>>>(c.probs, c.idx, s_j, dlogits, N, E, K);
    {
        // dWt(D,E) = Xf_c(D,N) @ dlogits(N,E): m=D,n=E,k=N
        float al = 1, be = 0;
        cublasStatus_t s = cublasGemmEx(
            cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, D, E, N, &al,
            c.Xf, CUDA_R_32F, D, dlogits, CUDA_R_32F, N, &be,
            dWrouter, CUDA_R_32F, D,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (s != CUBLAS_STATUS_SUCCESS) { std::printf("router dW FAIL\n"); exit(1); }
    }
    // Router-Inputgrad: dX += dlogits_b @ Wrouter (beta=1 auf Scatter)
    {
        bf16* dlog_b;
        CUDA_CHECK(cudaMalloc(&dlog_b, (size_t)N * E * 2));
        long n = (long)N * E;
        to_bf16_kernel<<<(n + TPB - 1) / TPB, TPB>>>(dlogits, dlog_b, n);
        // dX(N,D) += dlog_b(N,E) @ Wrouter(E,D): m=N? Als Col: Ct(D,N) =
        // Wr(D,E) @ dlogt(E,N): m=D, n=N, k=E, A=Wr lda=D, B=dlog_b ldb=E.
        float al = 1.0f, be = 1.0f;
        cublasStatus_t s2 = cublasGemmEx(
            cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, D, N, E, &al,
            Wrouter, CUDA_R_16BF, D, dlog_b, CUDA_R_16BF, E, &be, dX,
            CUDA_R_16BF, D,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (s2 != CUBLAS_STATUS_SUCCESS) { std::printf("router dX FAIL\n"); exit(1); }
        CUDA_CHECK(cudaFree(dlog_b));
    }
    CUDA_CHECK(cudaFree(dlogits));
    // Experten pro Gruppe
    bf16* dXg;
    CUDA_CHECK(cudaMalloc(&dXg, (size_t)tk * D * 2));
    for (int e = 0; e < E; ++e) {
        if (c.hcnt[e] == 0) continue;
        long off = (long)c.hoff[e] * D;
        linear_dW(c.hcnt[e], D, D, dYg + off, c.Xg + off, dWexp[e]);
        linear_dX(c.hcnt[e], D, D, dYg + off, Wexp[e], dXg + off);
    }
    scatter_add_kernel<<<blocks, TPB>>>(dXg, c.slot_of, dX, N, K, D);
    CUDA_CHECK(cudaFree(dYg));
    CUDA_CHECK(cudaFree(s_j));
    CUDA_CHECK(cudaFree(dXg));
}

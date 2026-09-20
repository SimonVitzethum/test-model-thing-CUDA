#pragma once
// S3: MLA nach DeepSeek-Prinzip (ohne Absorptions-Trick, dafür exakt und
// verifizierbar): KV als komprimierter Latent im Cache, inhaltsabhängige Attention
// über einen begrenzten Cache, entkoppeltes RoPE.
// Dims: d Modell, h Köpfe, dh Kopf-Dim, L Latent-Rank, R RoPE-Dim.
// Params/Layer: Wq(d,h*(dh+R)), Wdkv(d,L), Wkr(d,R),
//               Wuk(L,h*dh), Wuv(L,h*dh), Wo(h*dh,d).
// Cache/Stream: lat(Cmax,L) + kr(Cmax,R), bf16.
#include "util.h"
#include "linalg.cu"

struct MlaCache {
    bf16* lat = nullptr;  // (Cmax,L)
    bf16* kr = nullptr;   // (Cmax,R)
    long* head = nullptr; // Host: Schreibposition (1 long, gemappt? Host ok)
    long head_h = 0;
    int Cmax = 0, L = 0, R = 0;
};

// RoPE (NeoX-Paare) vorwärts wie rückwärts (neg=true dreht zurück).
// x: (N,H,F) mit F>=R -> nur erste R Dims rotiert, Rest kopiert.
__global__ void rope_kernel(const bf16* x, bf16* y, const long* pos,
                            int N, int H, int F, int R, float theta,
                            bool neg) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)N * H * F;
    if (i >= n) return;
    long f = i % F;
    long tmp = i / F;
    int r = tmp / H;
    float v = bf2f(x[i]);
    if (f < R) {
        int p2 = f / 2;
        float ang = (float)pos[r] /
                    powf(theta, (2 * p2) / (float)R);
        if (neg) ang = -ang;
        float co = cosf(ang), si = sinf(ang);
        // Partner: f^1
        long j = i ^ 1;
        float u = bf2f(x[j]);
        float sgn = (f % 2 == 0) ? -1.0f : 1.0f;
        // (x_even, x_odd) -> (x_e*cos - x_o*sin, x_o*cos + x_e*sin)
        // sgn-Formel: gerade: v*co + sgn*u*si mit sgn=-1 -> v*co-u*si ✓
        //             ungerade: sgn=+1 -> u ist even-Partner: v*co+u*si ✓
        y[i] = f2bf(v * co + sgn * u * si);
    } else {
        y[i] = x[i];
    }
}

// Cache-Write: lat_win(N,L)+kr_win(N,R) -> Ring ab head (pro Stream ein
// Cache; diese Funktion bedient genau einen Stream).
__global__ void cache_write_kernel(const bf16* latw, const bf16* krw,
                                   bf16* lat, bf16* kr, long head, int N,
                                   int Cmax, int L, int R) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)N * (L + R);
    if (i >= n) return;
    long row = i / (L + R), f = i % (L + R);
    long dst = (head + row) % Cmax;
    if (f < L)
        lat[dst * L + f] = latw[row * L + f];
    else
        kr[dst * R + (f - L)] = krw[row * R + (f - L)];
}

// Q aus q_flat splitten: qc(N,H,dh), qr(N,H,R). q_flat: Köpfe konkateniert.
__global__ void qsplit_kernel(const bf16* q, bf16* qc, bf16* qr, int N,
                              int H, int dh, int R) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)N * H * (dh + R);
    if (i >= n) return;
    long fr = i % (dh + R);
    long tmp = i / (dh + R);
    int h = tmp % H;
    int r = tmp / H;
    if (fr < dh)
        qc[((long)r * H + h) * dh + fr] = q[i];
    else
        qr[((long)r * H + h) * R + (fr - dh)] = q[i];
}

// Umgekehrt joinen (Backward): dqc/dqr -> dq_flat.
__global__ void qjoin_kernel(const bf16* dqc, const bf16* dqr, bf16* dq,
                             int N, int H, int dh, int R) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)N * H * (dh + R);
    if (i >= n) return;
    long fr = i % (dh + R);
    long tmp = i / (dh + R);
    int h = tmp % H;
    int r = tmp / H;
    if (fr < dh)
        dq[i] = dqc[((long)r * H + h) * dh + fr];
    else
        dq[i] = dqr[((long)r * H + h) * R + (fr - dh)];
}

// Online-Softmax-Update für einen Cache-Chunk (ein Query-Block):
// S(Qc,Cc) fp32, P=exp(S-mnew); O=O*alpha+P@Vc; l,m aktualisiert.
// Generisch über (Qc,Cc): ein Block pro Query-Zeile.
__global__ void online_update_kernel(const float* S, const bf16* Vc,
                                     float* O, float* m, float* l, int Qc,
                                     int Cc, int dh, float scale) {
    int q = blockIdx.x;
    if (q >= Qc) return;
    (void)scale;
    const float* s = S + (long)q * Cc;
    float mx = s[0];
    for (int c = 1; c < Cc; ++c) mx = fmaxf(mx, s[c]);
    float m_old = m[q];
    float m_new = fmaxf(m_old, mx);
    float alpha = expf(m_old - m_new);
    float l_new = l[q] * alpha;
    float* o = O + (long)q * dh;
    for (int d = threadIdx.x; d < dh; d += blockDim.x) {
        float acc = o[d] * alpha;
        for (int c = 0; c < Cc; ++c)
            acc += expf(s[c] - m_new) * bf2f(Vc[(long)c * dh + d]);
        o[d] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ps = 0;
        for (int c = 0; c < Cc; ++c) ps += expf(s[c] - m_new);
        l[q] = l_new + ps;
        m[q] = m_new;
    }
}

// (N,H,F) -> (B,H,T,F): Köpfe/Streams head-major umsortieren.
__global__ void to_bhnt_kernel(const bf16* s, bf16* d, int B, int H, int T,
                               int F) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)B * H * T * F;
    if (i >= n) return;
    long f = i % F;
    long tmp = i / F;
    int t = tmp % T, h = (tmp / T) % H, b = tmp / (T * H);
    long r = (long)b * T + t;  // Zeile in (N=B*T,H,F)
    d[i] = s[(r * H + h) * F + f];
}

// Umgekehrt (B,H,T,F) -> (N,H,F).
__global__ void to_nh_kernel(const bf16* s, bf16* d, int B, int H, int T,
                             int F) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)B * H * T * F;
    if (i >= n) return;
    long f = i % F;
    long tmp = i / F;
    int h = tmp % H;
    long r = tmp / H;  // r = b*T + t
    d[(r * H + h) * F + f] = s[i];
}

// Cache-Gather eines Chunks (wrap-sicher): lat/kr -> chunk (Cc,L)/(Cc,R).
__global__ void cache_gather_kernel(const bf16* lat, const bf16* kr,
                                    bf16* clat, bf16* ckr, long head,
                                    int c0, int Cc, int Cmax, int L, int R) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)Cc * (L + R);
    if (i >= n) return;
    long f = i % (L + R);
    long c = i / (L + R);
    long slot = (head + c0 + c) % Cmax;
    if (f < L)
        clat[c * L + f] = lat[slot * L + f];
    else
        ckr[c * R + (f - L)] = kr[slot * R + (f - L)];
}

// Softmax Jacobian needs the dot product over ALL key chunks, dO dot O.
// A chunk-local sum would give incorrect gradients whenever Clen > Cc.
__global__ void softmax_bwd_kernel(const float* S, const float* dP,
                                   const float* LSE, bf16* dS, bf16* P,
                                   int rows, int Cc, const float* dO,
                                   const bf16* O, int H, int T, int dh, float scale) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int t = row % T, h = (row / T) % H, b = row / (T * H);
    long out = ((long)b * T + t) * H * dh + h * dh;
    float dot = 0;
    for (int d = 0; d < dh; ++d) dot += dO[(long)row * dh + d] * bf2f(O[out + d]);
    for (int c = threadIdx.x; c < Cc; c += blockDim.x) {
        long i = (long)row * Cc + c;
        float p = expf(S[i] - LSE[row]);
        P[i] = f2bf(p);
        dS[i] = f2bf(scale * p * (dP[i] - dot));
    }
}

// Batched Online-Update über (BH*T) Query-Zeilen, ein Block pro Zeile.
__global__ void online_update_batched_kernel(
    const float* S, const bf16* Vc, float* O, float* m, float* l, int T,
    int Cc, int dh, long BH) {
    long qb = blockIdx.x;
    if (qb >= BH * T) return;
    long bh = qb / T;
    int q = qb % T;
    const float* s = S + (bh * T + q) * Cc;
    float mx = s[0];
    for (int c = 1; c < Cc; ++c) mx = fmaxf(mx, s[c]);
    float m_old = m[qb];
    float m_new = fmaxf(m_old, mx);
    float alpha = expf(m_old - m_new);
    float l_new = l[qb] * alpha;
    const bf16* vc = Vc + bh * Cc * dh;
    float* o = O + (long)qb * dh;
    for (int d = threadIdx.x; d < dh; d += blockDim.x) {
        float acc = o[d] * alpha;
        for (int c = 0; c < Cc; ++c)
            acc += expf(s[c] - m_new) * bf2f(vc[(long)c * dh + d]);
        o[d] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ps = 0;
        for (int c = 0; c < Cc; ++c) ps += expf(s[c] - m_new);
        l[qb] = l_new + ps;
        m[qb] = m_new;
    }
}

// O /= l über (rows=BHT, dh)
__global__ void norm_out_kernel(float* O, const float* l, long rows, int dh) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * dh) return;
    O[i] /= l[i / dh];
}

// Softmax pro (b,h,t)-Zeile über Cc + Blend-Faktoren:
// P=exp(S-mnew) in BF16 (GEMM-Regel), m/l/alpha/LSE fp32.
__global__ void softmax_scale_kernel(const float* S, bf16* P, float* m,
                                     float* l, float* alpha, float* LSE,
                                     long rows, int Cc, bool last) {
    long r = blockIdx.x;
    if (r >= rows) return;
    const float* s = S + r * Cc;
    float mx = s[0];
    for (int c = 1; c < Cc; ++c) mx = fmaxf(mx, s[c]);
    float m_old = m[r];
    float m_new = fmaxf(m_old, mx);
    float al = expf(m_old - m_new);
    float ps = 0;
    for (int c = threadIdx.x; c < Cc; c += blockDim.x) {
        float p = expf(s[c] - m_new);
        P[r * Cc + c] = f2bf(p);
        ps += p;
    }
    __shared__ float buf[256];
    buf[threadIdx.x] = ps;
    __syncthreads();
    for (int t = 128; t > 0; t >>= 1) {
        if (threadIdx.x < t) buf[threadIdx.x] += buf[threadIdx.x + t];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        l[r] = l[r] * al + buf[0];
        m[r] = m_new;
        alpha[r] = al;
        if (last) LSE[r] = logf(l[r]) + m_new;
    }
}

// O *= alpha pro Zeile (O: rows*dh fp32, alpha: rows)
__global__ void scale_rows_kernel(float* O, const float* alpha, long rows,
                                  int dh) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * dh) return;
    O[i] *= alpha[i / dh];
}

// Maskierter Scatter Chunk-Grad -> Fenster-Grad (nur Slots im Fenster).
// dC (B,Cc,F) fp32-Chunk -> dW (N=B*T,F): slot s <-> pos base0+s.
__global__ void masked_scatter_kernel(const float* dC, float* dW, long base0,
                                      long H0, int B, int T, int Cc, int F,
                                      int c0) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)B * Cc * F;
    if (i >= n) return;
    long f = i % F;
    long tmp = i / F;
    int c = tmp % Cc, b = tmp / Cc;
    long pos = base0 + c0 + c;
    long w0 = base0 + H0, w1 = w0 + T;
    if (pos < w0 || pos >= w1) return;
    int t = (int)(pos - w0);
    // dW liegt als (N,F) mit N=B*T Zeilen r=b*T+t
    dW[((long)b * T + t) * F + f] += dC[((long)b * Cc + c) * F + f];
}

struct MlaP {  // Param-Indizes (Host)
    size_t q, dkv, kr, uk, uv, o, gamma, beta;
};
struct MlaC {  // Ring-Cache pro Layer (alle Streams)
    bf16 *lat = nullptr, *kr = nullptr;  // (B,Cmax,L/R)
    long head = 0, base0 = 0;            // Host, lockstep über Streams
    int B = 0, Cmax = 0, L = 0, R = 0;
};
struct MlaW {  // Workspace (einmal max. allokiert, pro Layer wiederverwendet)
    bf16 *qf = nullptr;                               // (N,H*(dh+R))
    bf16 *qcnh = nullptr, *qrnh = nullptr, *qrb = nullptr;  // (N,H,*)
    bf16 *latw = nullptr, *krw = nullptr, *krb = nullptr;   // (N,L/R)
    bf16 *qcb = nullptr, *qrb2 = nullptr;                 // (B,H,T,*)
    bf16 *Kc = nullptr, *Vc = nullptr;    // (B,H,Cc,dh) Köpfe-major
    bf16 *Kcf = nullptr, *Vcf = nullptr;  // (B*Cc,H*dh) flach (GEMM-out)
    bf16 *krck = nullptr;                 // (B,H,Cc,R) repliziert
    bf16 *clat = nullptr, *ckr = nullptr;  // (B,Cc,L/R) Chunk-Gather
    float *S = nullptr, *dP = nullptr;
    bf16 *P = nullptr, *dS = nullptr;  // Probs/dS in bf16 (GEMM-Regel!)
    float *O = nullptr, *m = nullptr, *l = nullptr, *al = nullptr;
    float *dKc = nullptr, *dVc = nullptr;  // (B,H,Cc,dh)
    float *dKrH = nullptr;                 // (B,H,Cc,R) rope dk pro Kopf
    float *dQc = nullptr, *dQr = nullptr;  // (B,H,T,*) akkumuliert fp32
    float* dOf = nullptr;                  // (B,H,T,dh) fp32
    bf16* dOb = nullptr;                   // (B,H,T,dh) bf16-Cast davon
    float *dLatc = nullptr;  // (B*Cc,L) Chunk-Gradient
    float *dKrc = nullptr;   // (B*Cc,R) Chunk-Gradient (köpfe-summiert)
    float *dLat = nullptr, *dKr = nullptr;  // (N,L/R) Fenster fp32
    bf16 *dLatb = nullptr, *dKrb = nullptr;
    bf16 *dOflat = nullptr;  // (N,H*dh) bf16
    bf16* dQflat = nullptr;  // (N,H*(dh+R))
    bf16* dKVb = nullptr;    // (B*Cc,H*dh) bf16-Cast für dW-Up
    long* pos = nullptr;  // (N) Positionen (ungenutzt, pos liegt im Harness)
};

inline void mla_cache_free(MlaC& c) {
    if (c.lat) cudaFree(c.lat);
    if (c.kr) cudaFree(c.kr);
    c = MlaC();
}

// Forward: Xq(N,D) -> Y(N,D) (residual-fertig, Caller addiert).
// Schreibt lat/kr-Fenster in Ring, liest vollen Cache (chunked).
// Gibt Switch-ähnlichen Aux nicht zurück (kein MoE). Speichert qc/qr/LSE/O.

// O fp32 (B,H,T,dh) -> Oflat bf16 (N=B*T,H*dh).
__global__ void o_to_flat_kernel(const float* s, bf16* d, int B, int H,
                                 int T, int dh) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)B * H * T * dh;
    if (i >= n) return;
    long f = i % dh;
    long tmp = i / dh;
    int t = tmp % T, h = (tmp / T) % H, b = tmp / (T * H);
    d[((long)b * T + t) * H * dh + h * dh + f] = f2bf(s[i]);
}

// Strided-Batched-Helfer. Row-Matrizen werden transponiert gelesen:
// S[b,h] (T,Cc) via St(Cc,T) = Kc_row(Cc,k) @ Qc_row^T(k,T), Batch=B*H.
// Kc row (Cc,k): OP_T lda=k. Qc row (T,k): als (k,T) col-view OP_N lda=k.
inline void scores_gemm(const bf16* Kc, const bf16* Qc, float* S, int B,
                        int H, int T, int Cc, int k, float beta, float scale = 1.0f) {
    float al = scale;
    cublasStatus_t s = cublasGemmStridedBatchedEx(
        cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, Cc, T, k, &al, Kc,
        CUDA_R_16BF, k, (long long)Cc * k, Qc, CUDA_R_16BF, k,
        (long long)T * k, &beta, S, CUDA_R_32F, Cc,
        (long long)T * Cc, B * H, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) { std::printf("scores GEMM FAIL\n"); exit(1); }
}
// O[b,h] (T,dh) = P[b,h](T,Cc) @ Vc[b,h](Cc,dh), via Ot=Vc^T@P^T.
// Vc row (Cc,dh): OP_N lda=dh liest Vc^T. P row (T,Cc): OP_N lda=Cc
// liest P^T. (Elementweise verifiziert.)
inline void pv_gemm(const bf16* P, const bf16* Vc, float* O, int B, int H,
                    int T, int Cc, int dh, float beta) {
    float al = 1.0f;
    cublasStatus_t s = cublasGemmStridedBatchedEx(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, dh, T, Cc, &al, Vc,
        CUDA_R_16BF, dh, (long long)Cc * dh, P, CUDA_R_16BF, Cc,
        (long long)T * Cc, &beta, O, CUDA_R_32F, dh,
        (long long)T * dh, B * H, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) { std::printf("pv GEMM FAIL\n"); exit(1); }
}
// dP[b,h] (T,Cc) = dO[b,h](T,dh) @ Vc[b,h]^T, via dPt(Cc,T)=Vc@ dOt.
inline void dp_gemm(const bf16* dO, const bf16* Vc, float* dP, int B, int H,
                    int T, int Cc, int dh) {
    float al = 1.0f, be = 0.0f;
    // dPt(Cc,T) = Vc(Cc,dh) @ dOt(dh,T): A=Vc row (Cc,dh)==col (dh,Cc)??
    // Korrekt: dPt(Cc,T): A muss (Cc,dh) col-major sein. Vc row (Cc,dh)
    // == col (dh,Cc) -> OP_T gibt (Cc,dh) ✓ lda=dh.
    // B=dO row (T,dh)==col (dh,T) OP_N (dh,T)?? brauchen (dh,T): OP_N
    // mit ldb=dh ✓. C=dPt col (Cc,T) ldc=Cc ✓.
    cublasStatus_t s = cublasGemmStridedBatchedEx(
        cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, Cc, T, dh, &al, Vc,
        CUDA_R_16BF, dh, (long long)Cc * dh, dO, CUDA_R_16BF, dh,
        (long long)T * dh, &be, dP, CUDA_R_32F, Cc,
        (long long)T * Cc, B * H, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) { std::printf("dp GEMM FAIL\n"); exit(1); }
}
// dQ[b,h] (T,k) += dS[b,h](T,Cc) @ K[b,h](Cc,k), via dQt=Kct@dS.
inline void dq_gemm(const bf16* dS, const bf16* Kc, float* dQ, int B, int H,
                    int T, int Cc, int k) {
    float al = 1.0f, be = 1.0f;
    // dQt(k,T) = Kct(k,Cc) @ dS(T,Cc): A=Kc row (Cc,k)==col (k,Cc)
    // OP_N lda=k; B=dS row (T,Cc)==col (Cc,T) OP_T ldb=Cc; C=dQt ldc=k.
    cublasStatus_t s = cublasGemmStridedBatchedEx(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, k, T, Cc, &al, Kc,
        CUDA_R_16BF, k, (long long)Cc * k, dS, CUDA_R_16BF, Cc,
        (long long)T * Cc, &be, dQ, CUDA_R_32F, k,
        (long long)T * k, B * H, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) { std::printf("dq GEMM FAIL\n"); exit(1); }
}
// dKc[b,h] (Cc,k) = dS^T @ Qc: dKct(k,Cc) = Qct(k,T) @ dS(T,Cc).
inline void dk_gemm(const bf16* dS, const bf16* Qc, float* dKc, int B,
                    int H, int T, int Cc, int k) {
    float al = 1.0f, be = 0.0f;
    cublasStatus_t s = cublasGemmStridedBatchedEx(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, k, Cc, T, &al, Qc,
        CUDA_R_16BF, k, (long long)T * k, dS, CUDA_R_16BF, Cc,
        (long long)T * Cc, &be, dKc, CUDA_R_32F, k,
        (long long)Cc * k, B * H, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) { std::printf("dk GEMM FAIL\n"); exit(1); }
}
// dVc[b,h] (Cc,dh) = P^T @ dO: dVct(dh,Cc) = dOt(dh,T) @ P(T,Cc).
inline void dv_gemm(const bf16* P, const bf16* dO, float* dVc, int B,
                    int H, int T, int Cc, int dh) {
    float al = 1.0f, be = 0.0f;
    // A=dO row (T,dh)==col (dh,T) OP_N lda=dh; B=P row (T,Cc)==col (Cc,T)
    // OP_T ldb=Cc; C=dVct col (dh,Cc) ldc=dh.
    cublasStatus_t s = cublasGemmStridedBatchedEx(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, dh, Cc, T, &al, dO,
        CUDA_R_16BF, dh, (long long)T * dh, P, CUDA_R_16BF, Cc,
        (long long)T * Cc, &be, dVc, CUDA_R_32F, dh,
        (long long)Cc * dh, B * H, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) { std::printf("dv GEMM FAIL\n"); exit(1); }
}

// Kausal-Maske auf Scores: key-Pos > query-Pos -> -inf.
__global__ void causal_mask_kernel(float* S, long key_base, int c0,
                                   long q_base, int B, int H, int T, int Cc) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)B * H * T * Cc;
    if (i >= n) return;
    long c = i % Cc;
    long tmp = i / Cc;
    int t = tmp % T;
    if (key_base + c0 + c > q_base + t) S[i] = -1e30f;
}

__global__ void fill_f32_kernel(float* a, float v, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = v;
}

// kr-Chunk (B,Cc,R) -> pro Kopf repliziert (B,H,Cc,R).
__global__ void rep_hr_kernel(const bf16* s, bf16* d, int B, int H, int Cc,
                              int R) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)B * H * Cc * R;
    if (i >= n) return;
    long f = i % R;
    long tmp = i / R;
    int c = tmp % Cc;
    int b = tmp / (Cc * H);
    d[i] = s[((long)b * Cc + c) * R + f];
}

// (B,H,Cc,F) Köpfe konkatenieren -> (B*Cc,H*F) für dLat-GEMMs.
__global__ void bhn_to_flat_kernel(const float* s, float* d, int B, int H,
                                   int Cc, int F) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)B * Cc * H * F;
    if (i >= n) return;
    long f = i % F;
    long tmp = i / F;
    int h = tmp % H;
    long bc = tmp / H;  // b*Cc + c
    d[i] = s[((bc * H + h)) * F + f];
}

// Pro-Kopf (Cc,F)-Blöcke aus (B,Cc,H*F): für Up-Projektion Alternative.
// (Ungenutzt wenn Up direkt pro Kopf via linear_fwd auf Ranges läuft.)

// ---- Keep (pro Layer, persistent bis Backward) ----
struct MlaKeep {
    bf16 *qc = nullptr, *qr = nullptr;  // (B,H,T,dh/R)
    bf16* Oflat = nullptr;              // (N,H*dh)
    bf16* Xq = nullptr;                 // (N,D) Layer-Input (für dW)
    float* LSE = nullptr;               // (B,H,T)
};

// Rope-Backward: Rotation um -angle(pos), pos = base+c0+c.
// s/d: (B*Cc,R) fp32.
__global__ void rope_bwd_kernel(const float* s, float* d, long base, int c0,
                                int B, int Cc, int R, float theta) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)B * Cc * R;
    if (i >= n) return;
    long f = i % R;
    long tmp = i / R;
    int c = tmp % Cc;
    float ang = -(float)(base + c0 + c) / powf(theta, (2 * (f / 2)) / (float)R);
    float co = cosf(ang), si = sinf(ang);
    long j = i ^ 1;
    float u = s[j];
    float sgn = (f % 2 == 0) ? -1.0f : 1.0f;
    d[i] = s[i] * co + sgn * u * si;
}

// fp32 (B,H,T,F) -> bf16 flach (B*T=N,H*F), Köpfe konkateniert.
__global__ void dq_join_kernel(const float* dQc, const float* dQr, bf16* dq,
                               int B, int H, int T, int dh, int R,
                               const long* pos, float theta) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)B * T * H * (dh + R);
    if (i >= n) return;
    long f = i % (dh + R);
    long tmp = i / (dh + R);
    int h = tmp % H;
    long r = tmp / H;  // r = b*T + t
    long b = r / T, t = r % T;
    if (f < dh)
        dq[i] = f2bf(dQc[((long)b * H + h) * T * dh + t * dh + f]);
    else {
        int j = f - dh;
        long base = ((long)b * H + h) * T * R + t * R;
        float angle = pos[r] / powf(theta, (2 * (j / 2)) / (float)R);
        float sign = j % 2 == 0 ? 1.f : -1.f; // inverse RoPE
        dq[i] = f2bf(dQr[base + j] * cosf(angle) + sign * dQr[base + (j ^ 1)] * sinf(angle));
    }
}

// fp32 -> bf16 plain.
__global__ void mla_f2b_kernel(const float* s, bf16* d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = f2bf(s[i]);
}

// Evict only the oldest prefix when a new window would exceed capacity.
__global__ void compact_cache_kernel(bf16* data, int B, int capacity, int F, int drop, int keep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * F) return;
    int b = i / F, f = i % F;
    for (int c = 0; c < keep; ++c)
        data[((long)b * capacity + c) * F + f] = data[((long)b * capacity + c + drop) * F + f];
}

// Forward: Xq(N,D) -> Y(N,D). Schreibt Cache, liest vollen Cache chunked.
// hpos: (N,) Positionen (Device).
inline void mla_forward(
    const bf16* Xq, int N, const bf16* Wq, const bf16* Wdkv,
    const bf16* Wkr, const bf16* Wuk, const bf16* Wuv, const bf16* Wo,
    MlaC& C, MlaKeep& Kp, MlaW& W, bf16* Y, const long* hpos,
    int B, int T, int H, int dh, int L, int R, int Cc, int Cmax,
    float theta, float scale, int D) {
    const int TPB = 256;
    long BH = (long)B * H, BHT = BH * T;
    if (C.head + T > Cmax) {
        int drop = C.head + T - Cmax, keep = C.head - drop;
        compact_cache_kernel<<<(B * L + 255) / 256, 256>>>(C.lat, B, Cmax, L, drop, keep);
        compact_cache_kernel<<<(B * R + 255) / 256, 256>>>(C.kr, B, Cmax, R, drop, keep);
        C.base0 += drop; C.head = keep;
    }
    long H0win = C.head;  // Fenster-Start im Cache (linear, kein Wrap)
    // Xq für Backward sichern (dW braucht Input)
    CUDA_CHECK(cudaMemcpy(Kp.Xq, Xq, (long)N * D * 2,
                          cudaMemcpyDeviceToDevice));
    // Q/Lat/Kr GEMMs
    linear_fwd(N, H * (dh + R), D, Xq, Wq, W.qf);
    linear_fwd(N, L, D, Xq, Wdkv, W.latw);
    linear_fwd(N, R, D, Xq, Wkr, W.krw);
    {
        long n = (long)N * H * (dh + R);
        qsplit_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.qf, W.qcnh, W.qrnh, N,
                                                   H, dh, R);
    }
    {
        long n = (long)N * H * R;
        rope_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.qrnh, W.qrb, hpos, N, H,
                                                 R, R, theta, false);
    }
    {
        long n = (long)N * R;
        rope_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.krw, W.krb, hpos, N, 1,
                                                 R, R, theta, false);
    }
    // Cache-Write pro Stream (B kleine Launches)
    for (int b = 0; b < B; ++b) {
        long n = (long)T * (L + R);
        cache_write_kernel<<<(n + TPB - 1) / TPB, TPB>>>(
            W.latw + (long)b * T * L, W.krb + (long)b * T * R,
            C.lat + (long)b * Cmax * L, C.kr + (long)b * Cmax * R, C.head,
            T, Cmax, L, R);
    }
    long Clen = C.head + T;  // Cache-Länge nach Write
    // Layouts für Batched-GEMMs
    {
        long n = BH * T * dh;
        to_bhnt_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.qcnh, W.qcb, B, H, T,
                                                    dh);
    }
    {
        long n = BH * T * R;
        to_bhnt_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.qrb, W.qrb2, B, H, T,
                                                    R);
    }
    // Akkumulatoren
    {
        long n = BHT;
        fill_f32_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.m, -1e30f, n);
        CUDA_CHECK(cudaMemset(W.l, 0, n * 4));
        CUDA_CHECK(cudaMemset(W.O, 0, n * dh * 4));
    }
    for (long c0 = 0; c0 < Clen; c0 += Cc) {
        int cce = (int)((Clen - c0) < Cc ? (Clen - c0) : Cc);
        // Gather Chunk (alle Streams, 1 Launch)
        {
            long n = (long)B * cce * (L + R);
            // cache_gather ist single-stream -> pro Stream (B Launches)
            for (int b = 0; b < B; ++b)
                cache_gather_kernel<<<(n / B + TPB - 1) / TPB, TPB>>>(
                    C.lat + (long)b * Cmax * L, C.kr + (long)b * Cmax * R,
                    W.clat + (long)b * cce * L, W.ckr + (long)b * cce * R,
                    0, (int)c0, cce, Cmax, L, R);
        }
        // Prefix eviction keeps the retained cache contiguous; base0 tracks absolute positions.
        // Up-Projektionen flach: (B*cce,L)->(B*cce,H*dh), je 1 GEMM
        linear_fwd(B * cce, H * dh, L, W.clat, Wuk, W.Kcf);
        linear_fwd(B * cce, H * dh, L, W.clat, Wuv, W.Vcf);
        // -> Köpfe-major (B,H,cce,dh)
        {
            long n = BH * cce * dh;
            to_bhnt_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.Kcf, W.Kc, B, H,
                                                        cce, dh);
            to_bhnt_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.Vcf, W.Vc, B, H,
                                                        cce, dh);
        }
        // kr replizieren (B,cce,R)->(B,H,cce,R)
        {
            long n = BH * cce * R;
            rep_hr_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.ckr, W.krck, B, H,
                                                       cce, R);
        }
        // Scores content (beta=0) + rope (beta=1), Scale als alpha.
        scores_gemm(W.Kc, W.qcb, W.S, B, H, T, cce, dh, 0.0f, scale);
        scores_gemm(W.krck, W.qrb2, W.S, B, H, T, cce, R, 1.0f, scale);
        // Kausal-Maske (key-Pos > query-Pos -> -inf)
        {
            long n = BH * T * cce;
            causal_mask_kernel<<<(n + TPB - 1) / TPB, TPB>>>(
                W.S, C.base0, (int)c0, C.base0 + H0win, B, H, T, cce);
        }
        // All matrices in this chunk use the actual cce stride.
        CUDA_CHECK(cudaMemset(W.P, 0, BH * T * cce * 2));
        bool last = (c0 + Cc >= Clen);
        {
            long rows = BHT;
            softmax_scale_kernel<<<rows, TPB>>>(W.S, W.P, W.m, W.l, W.al,
                                                Kp.LSE, rows, cce, last);
            scale_rows_kernel<<<(BHT * dh + TPB - 1) / TPB, TPB>>>(
                W.O, W.al, BHT, dh);
        }
        pv_gemm(W.P, W.Vc, W.O, B, H, T, cce, dh, 1.0f);
    }
    // O normieren, Keep kopieren, flach + bf16, Wo-Projektion
    {
        long n = BHT * dh;
        norm_out_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.O, W.l, BHT, dh);
    }
    {
        long n = BH * T * dh;
        CUDA_CHECK(cudaMemcpy(Kp.qc, W.qcb, n * 2, cudaMemcpyDeviceToDevice));
    }
    {
        long n = BH * T * R;
        CUDA_CHECK(cudaMemcpy(Kp.qr, W.qrb2, n * 2, cudaMemcpyDeviceToDevice));
    }
    // O fp32 (B,H,T,dh) -> Oflat bf16 (N,H*dh)
    {
        long n = BH * T * dh;
        o_to_flat_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.O, Kp.Oflat, B, H,
                                                      T, dh);
    }
    linear_fwd(N, D, H * dh, Kp.Oflat, Wo, Y);
    C.head = H0win + T;
}

// dOflat (N,H*dh) bf16 -> dOf (B,H,T,dh) fp32.
__global__ void do_to_bhnt_kernel(const bf16* s, float* d, int B, int H,
                                  int T, int dh) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)B * H * T * dh;
    if (i >= n) return;
    long f = i % dh;
    long tmp = i / dh;
    int t = tmp % T, h = (tmp / T) % H, b = tmp / (T * H);
    d[i] = bf2f(s[((long)b * T + t) * H * dh + h * dh + f]);
}

__global__ void add_bf16_kernel_mla(bf16* a, const bf16* b, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = f2bf(bf2f(a[i]) + bf2f(b[i]));
}

// Tail nullen: Spalten [cce,Cc) in (rows,Cc)-Matrix.
__global__ void zero_tail_kernel(float* A, long rows, int cce, int Cc) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = rows * (Cc - cce);
    if (i >= n || cce >= Cc) return;
    long c = i % (Cc - cce);
    long r = i / (Cc - cce);
    A[r * Cc + cce + c] = 0.0f;
}

// Köpfe summieren: (B,H,Cc,F) -> (B*Cc,F).
__global__ void sum_heads_kernel(const float* s, float* d, int B, int H,
                                 int Cc, int F) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)B * Cc * F;
    if (i >= n) return;
    long f = i % F;
    long tmp = i / F;
    int c = tmp % Cc, b = tmp / Cc;
    float acc = 0;
    for (int h = 0; h < H; ++h)
        acc += s[((long)b * H + h) * Cc * F + c * F + f];
    d[((long)b * Cc + c) * F + f] = acc;
}

// fp32 (B,H,T,F) -> bf16 (N=B*T,H*F) flach (für lineare Bwd-GEMMs).
__global__ void bhnt_to_flat_bf16_kernel(const float* s, bf16* d, int B,
                                         int H, int T, int F) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)B * H * T * F;
    if (i >= n) return;
    long f = i % F;
    long tmp = i / F;
    int t = tmp % T, h = (tmp / T) % H, b = tmp / (T * H);
    d[((long)b * T + t) * H * F + h * F + f] = f2bf(s[i]);
}

// fp32 (B,H,T,F) buf -> bf16 (B,H,T,F).
__global__ void f32_to_bf16_kernel(const float* s, bf16* d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = f2bf(s[i]);
}

// (N,H,F) bf16 -> (B,H,T,F) bf16 (für dQ-RoPE-Pfad).
__global__ void nh_to_bhnt_bf16_kernel(const bf16* s, bf16* d, int B, int H,
                                       int T, int F) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)B * H * T * F;
    if (i >= n) return;
    long f = i % F;
    long tmp = i / F;
    int t = tmp % T, h = (tmp / T) % H, b = tmp / (T * H);
    long r = (long)b * T + t;
    d[i] = s[(r * H + h) * F + f];
}

// dLat-Chunk (B*Cc,L) = dKb(B*Cc,H*dh) @ Wuk^T, beta akkumuliert.
// Alles bf16 außer Akkumulation (fp32) — keine Mixed-Dtypes in cuBLAS.
inline void dlat_gemm(const bf16* dK, const bf16* Wu, float* dLat, int BCc,
                      int Hdh, int L, float beta) {
    float al = 1.0f;
    // W is row-major (Hdh,L), its column-major view is (L,Hdh).
    // dLat^T(L,BCc) = W_view(L,Hdh) @ dK^T(Hdh,BCc).
    // dK mem (BCc,Hdh) row == (Hdh,BCc) col: OP_N lda=Hdh.
    cublasStatus_t s = cublasGemmEx(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, L, BCc, Hdh, &al, Wu,
        CUDA_R_16BF, L, dK, CUDA_R_16BF, Hdh, &beta, dLat, CUDA_R_32F, L,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) { std::printf("dlat GEMM FAIL\n"); exit(1); }
}

// Backward: dY(N,D) -> dX(N,D); dW* fp32 (Caller nullt vorher).
// Nutzt Keep aus Forward (qc,qr,Oflat,LSE). Cache bleibt (kein Grad in
// Vergangenheit); Fenster-Grade fließen durch late/kr-Lineare.
inline void mla_backward(
    const bf16* dY, int N, const bf16* Wq, const bf16* Wdkv,
    const bf16* Wkr, const bf16* Wuk, const bf16* Wuv, const bf16* Wo,
    float* dWq, float* dWdkev, float* dWkr, float* dWuk, float* dWuv,
    float* dWo, const MlaC& C, const MlaKeep& Kp, MlaW& W, bf16* dX,
    const bf16* Xq, const long* hpos, int B, int T, int H, int dh, int L,
    int R, int Cc, float theta, float scale, int D) {
    const int TPB = 256;
    long BH = (long)B * H, BHT = BH * T;
    // Out-Proj
    linear_dW(N, D, H * dh, dY, Kp.Oflat, dWo);
    linear_dX(N, D, H * dh, dY, Wo, W.dOflat);
    // dO -> fp32 BHNT, dann bf16-Cast (GEMM-Regel)
    {
        long n = BH * T * dh;
        do_to_bhnt_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.dOflat, W.dOf, B,
                                                       H, T, dh);
        mla_f2b_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.dOf, W.dOb, n);
    }
    CUDA_CHECK(cudaMemset(W.dQc, 0, BH * T * dh * 4));
    CUDA_CHECK(cudaMemset(W.dQr, 0, BH * T * R * 4));
    CUDA_CHECK(cudaMemset(W.dLat, 0, (long)N * L * 4));
    CUDA_CHECK(cudaMemset(W.dKr, 0, (long)N * R * 4));
    long H0win = C.head - T;  // Fenster-Start (head zeigt dahinter)
    long Clen = C.head;
    for (long c0 = 0; c0 < Clen; c0 += Cc) {
        int cce = (int)((Clen - c0) < Cc ? (Clen - c0) : Cc);
        // Chunk-Daten wie Forward (Gather + Up)
        for (int b = 0; b < B; ++b)
            cache_gather_kernel<<<((long)B * cce * (L + R) / B + TPB - 1) / TPB,
                                  TPB>>>(
                C.lat + (long)b * C.Cmax * L, C.kr + (long)b * C.Cmax * R,
                W.clat + (long)b * cce * L, W.ckr + (long)b * cce * R, 0,
                (int)c0, cce, C.Cmax, L, R);
        linear_fwd(B * cce, H * dh, L, W.clat, Wuk, W.Kcf);
        linear_fwd(B * cce, H * dh, L, W.clat, Wuv, W.Vcf);
        {
            long n = BH * cce * dh;
            to_bhnt_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.Kcf, W.Kc, B, H,
                                                        cce, dh);
            to_bhnt_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.Vcf, W.Vc, B, H,
                                                        cce, dh);
        }
        {
            long n = BH * cce * R;
            rep_hr_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.ckr, W.krck, B, H,
                                                       cce, R);
        }
        // S recompute (beta=0) + Maske (Helper mit korrekten Strides)
        scores_gemm(W.Kc, Kp.qc, W.S, B, H, T, cce, dh, 0.0f, scale);
        scores_gemm(W.krck, Kp.qr, W.S, B, H, T, cce, R, 1.0f, scale);
        {
            long n = BH * T * cce;
            CUDA_CHECK(cudaMemset(W.P, 0, n * 2));
            causal_mask_kernel<<<(n + TPB - 1) / TPB, TPB>>>(
                W.S, C.base0, (int)c0, C.base0 + H0win, B, H, T, cce);
        }
        // dP, dS (+P-Recompute in bf16), Keep-LSE aus Forward
        dp_gemm(W.dOb, W.Vc, W.dP, B, H, T, cce, dh);
        {
            long rows = BHT;
            long nfull = rows * cce;
            CUDA_CHECK(cudaMemset(W.dS, 0, nfull * 2));
            CUDA_CHECK(cudaMemset(W.P, 0, nfull * 2));
            softmax_bwd_kernel<<<rows, TPB>>>(W.S, W.dP, Kp.LSE, W.dS, W.P,
                                              rows, cce, W.dOf, Kp.Oflat, H, T, dh, scale);
        }
        // dQ akkumulieren (content + rope)
        dq_gemm(W.dS, W.Kc, W.dQc, B, H, T, cce, dh);
        dq_gemm(W.dS, W.krck, W.dQr, B, H, T, cce, R);
        // dKc/dVc pro Chunk (nutzen P aus softmax_bwd)
        dk_gemm(W.dS, Kp.qc, W.dKc, B, H, T, cce, dh);
        dv_gemm(W.P, W.dOb, W.dVc, B, H, T, cce, dh);
// dKc/dVc (B,H,Cc,dh) -> flach bf16 (B*Cc,H*dh) -> dLatc (B*Cc,L).
// Dann Up-Weight-Grade (chunk-akkumuliert) + Rope-Pfad + Scatter.
{
    long nflat = (long)B * cce * H * dh;
    bool first = (c0 == 0);
    bhnt_to_flat_bf16_kernel<<<(nflat + TPB - 1) / TPB, TPB>>>(
        W.dKc, W.dKVb, B, H, cce, dh);
    dlat_gemm(W.dKVb, Wuk, W.dLatc, B * cce, H * dh, L, 0.0f);
    linear_dW(B * cce, H * dh, L, W.dKVb, W.clat, dWuk,
              first ? 0.0f : 1.0f);
    bhnt_to_flat_bf16_kernel<<<(nflat + TPB - 1) / TPB, TPB>>>(
        W.dVc, W.dKVb, B, H, cce, dh);
    dlat_gemm(W.dKVb, Wuv, W.dLatc, B * cce, H * dh, L, 1.0f);
    linear_dW(B * cce, H * dh, L, W.dKVb, W.clat, dWuv,
              first ? 0.0f : 1.0f);
            masked_scatter_kernel<<<((long)B * cce * L + TPB - 1) / TPB, TPB>>>(
                W.dLatc, W.dLat, C.base0, H0win, B, T, cce, L, (int)c0);
}
// Rope-dK pro Kopf -> köpfe-summiert (B*Cc,R) -> zurückrotieren
{
    dk_gemm(W.dS, Kp.qr, W.dKrH, B, H, T, cce, R);
    long n = (long)B * cce * R;
    sum_heads_kernel<<<(n + TPB - 1) / TPB, TPB>>>(
        W.dKrH, W.dKrc, B, H, cce, R);
    rope_bwd_kernel<<<(n + TPB - 1) / TPB, TPB>>>(
        W.dKrc, W.dLatc, C.base0, (int)c0, B, cce, R, theta);
    // ^^^ dLatc als Temporär (bereits gescattert, tot)
    masked_scatter_kernel<<<(n + TPB - 1) / TPB, TPB>>>(
        W.dLatc, W.dKr, C.base0, H0win, B, T, cce, R, (int)c0);
}
    }  // Chunk-Loop
    // ---- Fenster-Grade durch Eingangs-Lineare ----
    {
        long n = (long)N * H * (dh + R);
        dq_join_kernel<<<(n + TPB - 1) / TPB, TPB>>>(
            W.dQc, W.dQr, W.dQflat, B, H, T, dh, R, hpos, theta);
    }
    {
        long n = (long)N * L;
        mla_f2b_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.dLat, W.dLatb, n);
    }
    {
        long n = (long)N * R;
        mla_f2b_kernel<<<(n + TPB - 1) / TPB, TPB>>>(W.dKr, W.dKrb, n);
    }
    // dX = dXq + dXdkv + dXkr (beta-Akku über linear_dX)
    linear_dX(N, H * (dh + R), D, W.dQflat, Wq, dX, 0.0f);
    linear_dX(N, L, D, W.dLatb, Wdkv, dX, 1.0f);
    linear_dX(N, R, D, W.dKrb, Wkr, dX, 1.0f);
    linear_dW(N, H * (dh + R), D, W.dQflat, Xq, dWq);
    linear_dW(N, L, D, W.dLatb, Xq, dWdkev);
    linear_dW(N, R, D, W.dKrb, Xq, dWkr);
}

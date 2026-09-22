#pragma once
// CUDA model, explicit streaming state and shared forward/backward path.
#include "train.cu"
#include <string>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// ---------- Elementar-Kernel ----------
__global__ void add_bf16_kernel(bf16* a, const bf16* b, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = f2bf(bf2f(a[i]) + bf2f(b[i]));
}
__global__ void add_f32_kernel(float* a, const float* b, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
__global__ void zero_f32_kernel(float* a, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0;
}
__global__ void extract_carry_kernel(const float* S, float* carry, int B,
                                     int T, int D) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || d >= D) return;
    carry[(long)b * D + d] = S[((long)b * T + T - 1) * D + d];
}
__global__ void meanvar_kernel(const bf16* X, float* out2, long n) {
    __shared__ float b1[256], b2[256];
    float s = 0, q = 0;
    for (long i = threadIdx.x; i < n; i += blockDim.x) {
        float v = bf2f(X[i]);
        s += v; q += v * v;
    }
    b1[threadIdx.x] = s; b2[threadIdx.x] = q;
    __syncthreads();
    for (int t = 128; t > 0; t >>= 1) {
        if (threadIdx.x < t) {
            b1[threadIdx.x] += b1[threadIdx.x + t];
            b2[threadIdx.x] += b2[threadIdx.x + t];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = b1[0] / n;
        out2[0] = m; out2[1] = b2[0] / n - m * m;
    }
}
__global__ void ema_kernel(float* tgt, const float* src, float tau, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tgt[i] = tau * tgt[i] + (1 - tau) * src[i];
}
__global__ void copy_bf16_kernel(const float* s, bf16* d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = f2bf(s[i]);
}
__global__ void mse_mean_kernel(const float* a, const float* b, float* out,
                                long n) {
    __shared__ float buf[256];
    float s = 0;
    for (long i = threadIdx.x; i < n; i += blockDim.x) {
        float d = a[i] - b[i];
        s += d * d;
    }
    buf[threadIdx.x] = s;
    __syncthreads();
    for (int t = 128; t > 0; t >>= 1) {
        if (threadIdx.x < t) buf[threadIdx.x] += buf[threadIdx.x + t];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = buf[0] / n;
}
__global__ void f32_of_bf16_kernel(const bf16* s, float* d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = bf2f(s[i]);
}

// ---------- Modell ----------
struct MlaLayer {
    int use = 0;
    MlaP p;             // Param-Indizes
    MlaKeep keep;       // Keep pro Fenster
    float *mmean = nullptr, *mrstd = nullptr;  // (N,)
    bf16* Xsnap = nullptr;                     // (N,D) Norm-Input-Snapshot
};
struct Layer {
    size_t decay, gate, gamma, beta, router;
    size_t exp[16];
    bf16* input;
    float *S, *mean, *rstd, *initial;  // S(N,D), stats(N), carry(B,D)
    MoeKeep mc;
};
struct Model {
    ~Model() { for (auto& layer : L) moe_keep_free(layer.mc); }
    Cfg c;
    ParameterStore params;
    DeviceMemory memory;
    size_t emb, tgt, dec, stop;
    std::vector<Layer> L;
    std::vector<MlaLayer> ML;  // gleich groß wie L (use-Flag)
    MlaW MW;                   // shared Workspace
    bf16 *enc, *X, *Sb, *H, *M, *dXs, *dM, *dSnorm, *dH;
    MoeWs moeW;  // shared MoE-Arena (kein malloc pro Fenster)
    bf16 *logits, *dlogits, *stoplog, *dstop, *tgtX;
    float *dXres, *dEnc, *dEncTmp, *probs, *losstmp, *mv, *dDec;
    float *lam, *prod;     // (B,D) boundary adjoint / decay product, layer 0
    // Trace diagnostics: when log_traces is set, backward also accumulates the
    // trace part of the decay (L,D), gate (L,D) and embedding (256,D) gradients.
    bool log_traces = false;
    float* trlog = nullptr;
    int *ids, *nxt, *end;
    long* pos = nullptr;   // (N,) Positionen Device
    std::vector<MemLayer> MEM;  // fact memory (mem=1), same size as L
    MemShared MS;
    // Optional external gradient on the final representation X (N,D), added to
    // the loss gradient in backward_window (stage-2 retrieval loss).
    float* dXext = nullptr;
    MTParams opt;          // multi-tensor view of all parameters (optimizer, zeroing)
    float* moe_stats = nullptr;  // (L, 17) router statistics, read once per window
};

struct StreamState {
    DeviceMemory memory;
    std::vector<float*> carry;
    std::vector<MlaC> cache;
    long position = 0;
    // Hybrid traces (traces=1): ds_(t0-1)/dθ carried across windows.
    std::vector<float*> tdec, tgate;  // per layer (B,D)
    float* temb = nullptr;            // (B,256,D), layer 0 only
};

static void reset_state(StreamState& state, const Cfg& c) {
    for (auto* carry : state.carry)
        CUDA_CHECK(cudaMemset(carry, 0, (long)c.batch * c.dim * sizeof(float)));
    for (auto& cache : state.cache) { cache.head = 0; cache.base0 = 0; }
    long bd = (long)c.batch * c.dim * sizeof(float);
    for (auto* t : state.tdec) CUDA_CHECK(cudaMemset(t, 0, bd));
    for (auto* t : state.tgate) CUDA_CHECK(cudaMemset(t, 0, bd));
    if (state.temb) CUDA_CHECK(cudaMemset(state.temb, 0, 256 * bd));
    state.position = 0;
}

static void build_state(StreamState& state, const Model& m) {
    const Cfg& c = m.c;
    state.carry.resize(c.layers);
    state.cache.resize(c.layers);
    if (c.traces) {
        state.tdec.resize(c.layers); state.tgate.resize(c.layers);
        for (int l = 0; l < c.layers; ++l) {
            state.memory.allocate(state.tdec[l], (long)c.batch * c.dim * 4);
            state.memory.allocate(state.tgate[l], (long)c.batch * c.dim * 4);
        }
        state.memory.allocate(state.temb, 256L * c.batch * c.dim * 4);
    }
    for (int l = 0; l < c.layers; ++l) {
        state.memory.allocate(state.carry[l], (long)c.batch * c.dim * 4);
        if (m.ML[l].use) {
            auto& cache = state.cache[l];
            cache.B = c.batch; cache.Cmax = c.mla_cache;
            cache.L = c.mla_L; cache.R = c.mla_R;
            state.memory.allocate(cache.lat, (long)c.batch * c.mla_cache * c.mla_L * 2);
            state.memory.allocate(cache.kr, (long)c.batch * c.mla_cache * c.mla_R * 2);
            CUDA_CHECK(cudaMemset(cache.lat, 0, (long)c.batch * c.mla_cache * c.mla_L * 2));
            CUDA_CHECK(cudaMemset(cache.kr, 0, (long)c.batch * c.mla_cache * c.mla_R * 2));
        }
    }
    reset_state(state, c);
}

static float lr_at(const Cfg& c, int step) {
    if (step < c.warmup) return c.lr * (step + 1) / (float)c.warmup;
    float p = fminf(1.0f, (step - c.warmup) / (float)c.decaysteps);
    return c.lr * (c.minlr + 0.5f * (1 - c.minlr) *
                   (1 + cosf(3.14159265f * p)));
}

static void build_opt_table(Model& m);

static void build_model(Model& m) {
    Cfg& c = m.c;
    int D = c.dim, L = c.layers, E = c.experts, K = c.topk;
    unsigned seed = c.seed ? c.seed : 1234;
    int B = c.batch, T = c.seqlen;
    int N = B * T;
    long ND = (long)N * D;
    std::vector<float> h;
    auto init_u = [&](size_t p, float a, float b) {
        h.resize(m.params.at(p).n); host_init(h.data(), m.params.at(p).n, a, b, seed);
        CUDA_CHECK(cudaMemcpy(m.params.at(p).master, h.data(), m.params.at(p).n * 4,
                              cudaMemcpyHostToDevice));
        long n = m.params.at(p).n;
        copy_bf16_kernel<<<(n + 255) / 256, 256>>>(m.params.at(p).master, m.params.at(p).work, n);
    };
    m.emb = m.params.add(256L * D);
    init_u(m.emb, -0.05f, 0.05f);
    m.tgt = m.params.add(256L * D);
    CUDA_CHECK(cudaMemcpy(m.params.at(m.tgt).master, m.params.at(m.emb).master, 256L * D * 4,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(m.params.at(m.tgt).work, m.params.at(m.emb).work, 256L * D * 2,
                          cudaMemcpyDeviceToDevice));
    m.dec = m.params.add(256L * D);
    { float a = sqrtf(1.0f / D); init_u(m.dec, -a, a); }
    m.stop = m.params.add(1L * D);
    init_u(m.stop, -0.01f, 0.01f);
    m.L.resize(L);
    for (int l = 0; l < L; ++l) {
        Layer& Ly = m.L[l];
        Ly.decay = m.params.add(D);
        h.resize(D);
        for (int d = 0; d < D; ++d) {
            float half = c.half_min * powf(c.half_max / c.half_min,
                                          d / (float)std::max(1, D - 1));
            float a = expf(-logf(2.0f) / half);
            h[d] = logf(a / (1.0f - a));
        }
        CUDA_CHECK(cudaMemcpy(m.params.at(Ly.decay).master, h.data(), D * 4,
                              cudaMemcpyHostToDevice));
        copy_bf16_kernel<<<(D + 255) / 256, 256>>>(
            m.params.at(Ly.decay).master, m.params.at(Ly.decay).work, D);
        Ly.gate = m.params.add(D);
        init_u(Ly.gate, 0.f, 0.f);
        Ly.gamma = m.params.add(D);
        h.assign(D, 1.0f);
        CUDA_CHECK(cudaMemcpy(m.params.at(Ly.gamma).master, h.data(), D * 4,
                              cudaMemcpyHostToDevice));
        copy_bf16_kernel<<<(D + 255) / 256, 256>>>(
            m.params.at(Ly.gamma).master, m.params.at(Ly.gamma).work, D);
        Ly.beta = m.params.add(D);
        CUDA_CHECK(cudaMemset(m.params.at(Ly.beta).master, 0, D * 4));
        CUDA_CHECK(cudaMemset(m.params.at(Ly.beta).work, 0, D * 2));
        Ly.router = m.params.add((long)E * D);
        h.resize((long)E * D);
        host_normal(h.data(), h.size(), 0.02f, seed);
        CUDA_CHECK(cudaMemcpy(m.params.at(Ly.router).master, h.data(),
                              (long)E * D * 4, cudaMemcpyHostToDevice));
        {
            long n = m.params.at(Ly.router).n;
            copy_bf16_kernel<<<(n + 255) / 256, 256>>>(
                m.params.at(Ly.router).master, m.params.at(Ly.router).work, n);
        }
        float a = sqrtf(1.0f / D);
        for (int e = 0; e < E; ++e) {
            Ly.exp[e] = m.params.add((long)D * D);
            init_u(Ly.exp[e], -a, a);
        }
        m.memory.allocate(Ly.S, ND * 4);
        m.memory.allocate(Ly.mean, N * 4);
        m.memory.allocate(Ly.rstd, N * 4);
        m.memory.allocate(Ly.initial, (long)B * D * 4);
        m.memory.allocate(Ly.input, ND * 2);
        moe_keep_alloc(Ly.mc, N, E, K, D);
    }
    moe_ws_alloc(m.moeW, N, E, K, D);
    if (E > 1) {
        m.memory.allocate(m.moe_stats, (long)L * 17 * 4);
        for (int l = 0; l < L; ++l) m.L[l].mc.stat = m.moe_stats + (long)l * 17;
    }
    m.memory.allocate(m.enc, ND * 2);
    m.memory.allocate(m.X, ND * 2);
    m.memory.allocate(m.Sb, ND * 2);
    m.memory.allocate(m.H, ND * 2);
    m.memory.allocate(m.M, ND * 2);
    m.memory.allocate(m.dXs, ND * 2);
    m.memory.allocate(m.dM, ND * 2);
    m.memory.allocate(m.dSnorm, ND * 2);
    m.memory.allocate(m.dH, ND * 2);
    m.memory.allocate(m.dXres, ND * 4);
    m.memory.allocate(m.dEnc, ND * 4);
    m.memory.allocate(m.dEncTmp, ND * 4);
    m.memory.allocate(m.dDec, (size_t)D * 4);
    m.memory.allocate(m.lam, (long)B * D * 4);
    m.memory.allocate(m.prod, (long)B * D * 4);
    if (c.traces) m.memory.allocate(m.trlog, (2L * L + 256) * D * 4);
    m.memory.allocate(m.ids, N * 4);
    m.memory.allocate(m.nxt, N * 4);
    m.memory.allocate(m.end, N * 4);
    m.memory.allocate(m.logits, (long)N * 256 * 2);
    m.memory.allocate(m.dlogits, (long)N * 256 * 2);
    m.memory.allocate(m.stoplog, (long)N * 2);
    m.memory.allocate(m.dstop, (long)N * 2);
    m.memory.allocate(m.probs, (long)N * 256 * 4);
    m.memory.allocate(m.losstmp, (long)N * 4);
    m.memory.allocate(m.tgtX, ND * 2);
    m.memory.allocate(m.mv, 8);
    m.memory.allocate(m.pos, N * 8);
    // ---- MLA (optional) ----
    m.ML.resize(L);
    if (c.mla) {
        int H = c.mla_heads, dh = c.mla_dh, Lr = c.mla_L, R = c.mla_R;
        int Cc = c.mla_cc;
        float a = sqrtf(1.0f / D);
        for (int l = 0; l < L; ++l) {
            MlaLayer& Ml = m.ML[l];
            Ml.use = (l % c.mla_every == 0) ? 1 : 0;
            if (!Ml.use) continue;
            Ml.p.q = m.params.add((long)D * H * (dh + R)); init_u(Ml.p.q, -a, a);
            Ml.p.dkv = m.params.add((long)D * Lr); init_u(Ml.p.dkv, -a, a);
            Ml.p.kr = m.params.add((long)D * R); init_u(Ml.p.kr, -a, a);
            { float b = sqrtf(1.0f / Lr);
              Ml.p.uk = m.params.add((long)Lr * H * dh); init_u(Ml.p.uk, -b, b);
              Ml.p.uv = m.params.add((long)Lr * H * dh); init_u(Ml.p.uv, -b, b); }
            { float b2 = sqrtf(1.0f / (H * dh));
              Ml.p.o = m.params.add((long)H * dh * D); init_u(Ml.p.o, -b2, b2); }
            Ml.p.gamma = m.params.add(D);
            h.assign(D, 1.0f);
            CUDA_CHECK(cudaMemcpy(m.params.at(Ml.p.gamma).master, h.data(), D * 4,
                                  cudaMemcpyHostToDevice));
            copy_bf16_kernel<<<(D + 255) / 256, 256>>>(
                m.params.at(Ml.p.gamma).master, m.params.at(Ml.p.gamma).work, D);
            Ml.p.beta = m.params.add(D);
            CUDA_CHECK(cudaMemset(m.params.at(Ml.p.beta).master, 0, D * 4));
            CUDA_CHECK(cudaMemset(m.params.at(Ml.p.beta).work, 0, D * 2));
            m.memory.allocate(Ml.keep.qc, (long)B * H * T * dh * 2);
            m.memory.allocate(Ml.keep.qr, (long)B * H * T * R * 2);
            m.memory.allocate(Ml.keep.Oflat, (long)N * H * dh * 2);
            m.memory.allocate(Ml.keep.Xq, ND * 2);
            m.memory.allocate(Ml.keep.LSE, (long)B * H * T * 4);
            m.memory.allocate(Ml.mmean, N * 4);
            m.memory.allocate(Ml.mrstd, N * 4);
            m.memory.allocate(Ml.Xsnap, ND * 2);
        }
        // Workspace einmal max (Cc aus Config)
        MlaW& W = m.MW;
        long BH = (long)B * H;
        m.memory.allocate(W.qf, (long)N * H * (dh + R) * 2);
        m.memory.allocate(W.qcnh, (long)N * H * dh * 2);
        m.memory.allocate(W.qrnh, (long)N * H * R * 2);
        m.memory.allocate(W.qrb, (long)N * H * R * 2);
        m.memory.allocate(W.latw, (long)N * Lr * 2);
        m.memory.allocate(W.krw, (long)N * R * 2);
        m.memory.allocate(W.krb, (long)N * R * 2);
        m.memory.allocate(W.qcb, BH * T * dh * 2);
        m.memory.allocate(W.qrb2, BH * T * R * 2);
        m.memory.allocate(W.Kc, BH * Cc * dh * 2);
        m.memory.allocate(W.Vc, BH * Cc * dh * 2);
        m.memory.allocate(W.Kcf, (long)B * Cc * H * dh * 2);
        m.memory.allocate(W.Vcf, (long)B * Cc * H * dh * 2);
        m.memory.allocate(W.krck, BH * Cc * R * 2);
        m.memory.allocate(W.clat, (long)B * Cc * Lr * 2);
        m.memory.allocate(W.ckr, (long)B * Cc * R * 2);
        m.memory.allocate(W.S, BH * T * Cc * 4);
        m.memory.allocate(W.P, BH * T * Cc * 2);
        m.memory.allocate(W.dS, BH * T * Cc * 2);
        m.memory.allocate(W.dP, BH * T * Cc * 4);
        m.memory.allocate(W.O, BH * T * dh * 4);
        m.memory.allocate(W.m, BH * T * 4);
        m.memory.allocate(W.l, BH * T * 4);
        m.memory.allocate(W.al, BH * T * 4);
        m.memory.allocate(W.dKc, BH * Cc * dh * 4);
        m.memory.allocate(W.dVc, BH * Cc * dh * 4);
        m.memory.allocate(W.dKrH, BH * Cc * R * 4);
        m.memory.allocate(W.dQc, BH * T * dh * 4);
        m.memory.allocate(W.dQr, BH * T * R * 4);
        m.memory.allocate(W.dOf, BH * T * dh * 4);
        m.memory.allocate(W.dOb, BH * T * dh * 2);
        m.memory.allocate(W.dLatc, (long)B * Cc * std::max(Lr, R) * 4);
        m.memory.allocate(W.dKrc, (long)B * Cc * R * 4);
        m.memory.allocate(W.dLat, (long)N * Lr * 4);
        m.memory.allocate(W.dKr, (long)N * R * 4);
        m.memory.allocate(W.dLatb, (long)N * Lr * 2);
        m.memory.allocate(W.dKrb, (long)N * R * 2);
        m.memory.allocate(W.dOflat, (long)N * H * dh * 2);
        m.memory.allocate(W.dQflat, (long)N * H * (dh + R) * 2);
        m.memory.allocate(W.dKVb, (long)B * Cc * H * dh * 2);
    }
    // ---- Fact memory (optional). Added last so all other parameters keep
    // their initialization; Wo = 0 makes the untrained memory an exact no-op.
    m.MEM.resize(L);
    if (c.mem) {
        int M = c.mem_len, HD = c.mem_heads * c.mem_dh;
        MemShared& S = m.MS;
        S.eprev = m.params.add(256L * D); init_u(S.eprev, -0.05f, 0.05f);
        S.pos = m.params.add((long)M * D); init_u(S.pos, -0.05f, 0.05f);
        S.decay = m.params.add(D);  // encoder half-lives 1..64 bytes, gate starts at 0
        h.resize(D);
        for (int d = 0; d < D; ++d) {
            float half = powf(64.f, d / (float)std::max(1, D - 1));
            float a = expf(-logf(2.0f) / half);
            h[d] = logf(a / (1.0f - a));
        }
        CUDA_CHECK(cudaMemcpy(m.params.at(S.decay).master, h.data(), D * 4, cudaMemcpyHostToDevice));
        copy_bf16_kernel<<<(D + 255) / 256, 256>>>(m.params.at(S.decay).master, m.params.at(S.decay).work, D);
        S.gate = m.params.add(D); init_u(S.gate, 0.f, 0.f);
        float a = sqrtf(1.0f / D);
        for (int l = 0; l < L; ++l) {
            MemLayer& Me = m.MEM[l];
            Me.use = (l + 1) % c.mem_every == 0;
            if (!Me.use) continue;
            Me.gamma = m.params.add(D);
            h.assign(D, 1.0f);
            CUDA_CHECK(cudaMemcpy(m.params.at(Me.gamma).master, h.data(), D * 4, cudaMemcpyHostToDevice));
            copy_bf16_kernel<<<(D + 255) / 256, 256>>>(m.params.at(Me.gamma).master, m.params.at(Me.gamma).work, D);
            Me.beta = m.params.add(D);
            Me.wq = m.params.add((long)HD * D); init_u(Me.wq, -a, a);
            Me.wk = m.params.add((long)HD * D); init_u(Me.wk, -a, a);
            Me.wv = m.params.add((long)HD * D); init_u(Me.wv, -a, a);
            Me.wo = m.params.add((long)D * HD);
            for (size_t p : {Me.beta, Me.wo}) {
                CUDA_CHECK(cudaMemset(m.params.at(p).master, 0, m.params.at(p).n * 4));
                CUDA_CHECK(cudaMemset(m.params.at(p).work, 0, m.params.at(p).n * 2));
            }
            m.memory.allocate(Me.Xsnap, ND * 2); m.memory.allocate(Me.Hn, ND * 2);
            m.memory.allocate(Me.Q, (long)N * HD * 2); m.memory.allocate(Me.O, (long)N * HD * 2);
            m.memory.allocate(Me.K, (long)B * M * HD * 2); m.memory.allocate(Me.V, (long)B * M * HD * 2);
            m.memory.allocate(Me.mean, N * 4); m.memory.allocate(Me.rstd, N * 4);
            m.memory.allocate(Me.P, (long)B * c.mem_heads * T * M * 4);
        }
        m.memory.allocate(S.ids, (long)B * M * 4);
        CUDA_CHECK(cudaMemset(S.ids, 0xff, (long)B * M * 4));  // -1: no memory
        m.memory.allocate(S.enc, (long)B * M * D * 2);
        m.memory.allocate(S.enc0, (long)B * M * D * 2);
        m.memory.allocate(S.state, (long)B * M * D * 4);
        m.memory.allocate(S.dEnc0, (long)B * M * D * 4);
        m.memory.allocate(S.dO, (long)N * HD * 2); m.memory.allocate(S.dQ, (long)N * HD * 2);
        m.memory.allocate(S.dK, (long)B * M * HD * 2); m.memory.allocate(S.dV, (long)B * M * HD * 2);
        m.memory.allocate(S.dHn, ND * 2); m.memory.allocate(S.dXln, ND * 2);
        m.memory.allocate(S.dEncB, (long)B * M * D * 2);
        m.memory.allocate(S.dS, (long)B * c.mem_heads * T * M * 4);
        m.memory.allocate(S.dEnc, (long)B * M * D * 4);
        if (c.mem_rdim > 0) {  // retrieval heads, created after all other parameters
            float r = sqrtf(1.0f / D);
            S.rq = m.params.add((long)c.mem_rdim * D); init_u(S.rq, -r, r);
            S.rk = m.params.add((long)c.mem_rdim * D); init_u(S.rk, -r, r);
        }
    }
    build_opt_table(m);  // every parameter exists now
}

// Chunk table over all parameters for the multi-tensor kernels. Built once,
// after every parameter exists. Norm: all but the EMA target; AdamW: all but
// the EMA target and an unused stop head.
static void build_opt_table(Model& m) {
    size_t P = m.params.values.size();
    std::vector<float*> master(P), mm(P), vv(P), grad(P);
    std::vector<bf16*> work(P);
    std::vector<unsigned char> flags(P);
    std::vector<MTChunk> chunks;
    for (size_t i = 0; i < P; ++i) {
        auto& p = m.params.at(i);
        master[i] = p.master; mm[i] = p.m; vv[i] = p.v; grad[i] = p.grad; work[i] = p.work;
        bool frozen = i == m.tgt, unused_stop = i == m.stop && m.c.stop == 0;
        flags[i] = (frozen ? 0 : 1) | (frozen || unused_stop ? 0 : 2);
        for (long s = 0; s < p.n; s += MT_CHUNK)
            chunks.push_back({(int)i, (int)std::min<long>(MT_CHUNK, p.n - s), s});
    }
    auto up = [&](auto*& dst, const auto& src) {
        m.memory.allocate(dst, src.size() * sizeof(src[0]));
        CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(src[0]), cudaMemcpyHostToDevice));
    };
    up(m.opt.master, master); up(m.opt.m, mm); up(m.opt.v, vv); up(m.opt.grad, grad);
    up(m.opt.work, work); up(m.opt.flags, flags); up(m.opt.chunks, chunks);
    m.opt.nchunks = (int)chunks.size();
    m.memory.allocate(m.opt.sumsq, sizeof(double));
}

static float host_sum(float* d, int n) {
    std::vector<float> h;
    h.resize(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, (size_t)n * 4, cudaMemcpyDeviceToHost));
    double s = 0;
    for (int i = 0; i < n; ++i) s += h[i];
    return (float)s;
}

// Per-Experten bf16-Zeigerliste (Host) für moe_forward/backward.
static void exp_ptrs(Model& m, int l, bf16** out) {
    for (int e = 0; e < m.c.experts; ++e) out[e] = m.params.at(m.L[l].exp[e]).work;
}
static void exp_gptrs(Model& m, int l, float** out) {
    for (int e = 0; e < m.c.experts; ++e) out[e] = m.params.at(m.L[l].exp[e]).grad;
}

// ---------- Fenster-Step (Forward + Backward + Adam) ----------
static void forward_window(Model& m, StreamState& state, float& tot, float& ce_out) {
    Cfg& c = m.c;
    int B = c.batch, T = c.seqlen, D = c.dim, E = c.experts, K = c.topk;
    int N = B * T;
    long ND = (long)N * D;
    const int TPB = 256;
    long blocksL = (ND + TPB - 1) / TPB;
    int blocks = (N + TPB - 1) / TPB;

    // ---- Forward ----
    emb_forward(m.params.at(m.emb).work, m.ids, m.enc, N, D);
    CUDA_CHECK(cudaMemcpy(m.X, m.enc, ND * 2, cudaMemcpyDeviceToDevice));
    if (c.mem)
        mem_encode(m.params.at(m.emb).work, m.params.at(m.MS.eprev).work, m.params.at(m.MS.pos).work,
                   m.params.at(m.MS.decay).master, m.params.at(m.MS.gate).master, m.MS, B, c.mem_len, D);
    // Positionen (Lockstep über Streams)
    if (c.mla) {
        std::vector<long> hp;
        hp.resize(N);
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t) hp[b * T + t] = state.position + t;
        CUDA_CHECK(cudaMemcpy(m.pos, hp.data(), (size_t)N * 8,
                              cudaMemcpyHostToDevice));
    }
    float aux_acc = 0, z_acc = 0;
    bf16* Wx[16];
    for (size_t l = 0; l < m.L.size(); ++l) {
        Layer& Ly = m.L[l];
        CUDA_CHECK(cudaMemcpy(Ly.input, m.X, ND * 2, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(Ly.initial, state.carry[l], (long)B * D * 4,
                              cudaMemcpyDeviceToDevice));
        CellOpt opt; opt.ids = m.ids; opt.docsep = c.docsep;
        state_forward(Ly.input, Ly.S, m.params.at(Ly.decay).master, Ly.initial,
                      B, T, D, c.gated ? m.params.at(Ly.gate).master : nullptr, opt);
        copy_bf16_kernel<<<blocksL, TPB>>>(Ly.S, m.Sb, ND);
        layernorm_fwd(m.Sb, m.params.at(Ly.gamma).master, m.params.at(Ly.beta).master, m.H,
                      Ly.mean, Ly.rstd, N, D);
        exp_ptrs(m, (int)l, Wx);
        aux_acc += moe_forward(m.H, m.params.at(Ly.router).work, Wx, m.X,
                               Ly.mc, m.moeW, N, E, K, D, 1.0f);
        // Residual-Add steckt in combine (beta=1), kein separater Pass.
        // MLA-Block (optional): norm auf X-Stream + Attention + residual
        if (c.mla && m.ML[l].use) {
            MlaLayer& Ml = m.ML[l];
            CUDA_CHECK(cudaMemcpy(Ml.Xsnap, m.X, ND * 2,
                                  cudaMemcpyDeviceToDevice));
            layernorm_fwd(m.X, m.params.at(Ml.p.gamma).master, m.params.at(Ml.p.beta).master,
                          m.H, Ml.mmean, Ml.mrstd, N, D);
            float sc = 1.0f / sqrtf((float)c.mla_dh);
            mla_forward(m.H, N, m.params.at(Ml.p.q).work, m.params.at(Ml.p.dkv).work,
                        m.params.at(Ml.p.kr).work, m.params.at(Ml.p.uk).work, m.params.at(Ml.p.uv).work,
                        m.params.at(Ml.p.o).work, state.cache[l], Ml.keep, m.MW, m.M,
                        m.pos, B, T, c.mla_heads, c.mla_dh, c.mla_L,
                        c.mla_R, c.mla_cc, c.mla_cache, c.mla_theta, sc, D);
            add_bf16_kernel<<<blocksL, TPB>>>(m.X, m.M, ND);
        }
        if (c.mem && m.MEM[l].use) {
            MemLayer& Me = m.MEM[l];
            mem_forward(m.X, Me, m.MS, m.params.at(Me.gamma).master, m.params.at(Me.beta).master,
                        m.params.at(Me.wq).work, m.params.at(Me.wk).work, m.params.at(Me.wv).work,
                        m.params.at(Me.wo).work, m.M, B, T, c.mem_len, c.mem_heads, c.mem_dh, D);
        }
        // carry-out
        dim3 gc(B, (D + TPB - 1) / TPB);
        extract_carry_kernel<<<gc, TPB>>>(Ly.S, state.carry[l], B, T, D);
    }
    state.position += T;
    if (E > 1) {  // one host read for all layers' router statistics
        std::vector<float> hs((size_t)m.L.size() * 17);
        CUDA_CHECK(cudaMemcpy(hs.data(), m.moe_stats, hs.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t l = 0; l < m.L.size(); ++l) {
            aux_acc += moe_aux_from(m.L[l].mc, &hs[l * 17]);
            z_acc += m.L[l].mc.zloss;
        }
    }
    // Decode
    linear_fwd(N, 256, D, m.X, m.params.at(m.dec).work, m.logits);
    // CE (one host synchronization per window)
    ce_fwd_kernel<<<blocks, TPB>>>(m.logits, m.nxt, m.probs, m.losstmp, N);
    float ce_sum = host_sum(m.losstmp, N);
    ce_out = ce_sum / N;
    // Optional terms are computed only when enabled (each costs a host sync).
    float stop_mean = 0, var_loss = 0, mse = 0;
    if (c.stop > 0) {
        linear_fwd(N, 1, D, m.X, m.params.at(m.stop).work, m.stoplog);
        stop_fwd_kernel<<<blocks, TPB>>>(m.stoplog, m.end, m.losstmp, c.stopposw, N);
        stop_mean = host_sum(m.losstmp, N) / N;
    }
    if (c.latent > 0) {  // MSE against the EMA target embedding of the next byte
        emb_forward(m.params.at(m.tgt).work, m.nxt, m.tgtX, N, D);
        f32_of_bf16_kernel<<<blocksL, TPB>>>(m.X, m.dXres, ND);  // temporaries
        f32_of_bf16_kernel<<<blocksL, TPB>>>(m.tgtX, m.dEnc, ND);
        mse_mean_kernel<<<1, 256>>>(m.dXres, m.dEnc, m.mv, ND);
        CUDA_CHECK(cudaMemcpy(&mse, m.mv, 4, cudaMemcpyDeviceToHost));
    }
    if (c.var > 0) {
        float hmv[2];
        meanvar_kernel<<<1, 256>>>(m.X, m.mv, ND);
        CUDA_CHECK(cudaMemcpy(hmv, m.mv, 8, cudaMemcpyDeviceToHost));
        var_loss = fmaxf(0.0f, 1.0f - sqrtf(hmv[1] + 1e-4f));
    }
    tot = c.var * var_loss + c.latent * mse + c.ce * ce_out +
          c.stop * stop_mean + (c.aux * aux_acc + c.zloss * z_acc) / m.L.size();

    CUDA_CHECK(cudaGetLastError());
}

// Forward must immediately precede backward on this model's workspace.
// With traces=1, backward also advances the stream's traces (training only).
static void backward_window(Model& m, StreamState& state) {
    const Cfg& c = m.c;
    int B = c.batch, T = c.seqlen, D = c.dim, N = B * T;
    long ND = (long)N * D;
    const int TPB = 256;
    long blocksL = (ND + TPB - 1) / TPB;
    int blocks = (N + TPB - 1) / TPB;
    bf16* Wx[16];
    mt_zero_kernel<<<m.opt.nchunks, 256>>>(m.opt);  // all gradients, one launch
    bool logging = m.log_traces && m.trlog;
    if (logging) CUDA_CHECK(cudaMemset(m.trlog, 0, (2L * c.layers + 256) * D * 4));
    // Reconstruct loss auxiliaries from the shared forward activations.
    float hmv[2] = {0.f, 1.f};
    if (c.var > 0) {
        meanvar_kernel<<<1, 256>>>(m.X, m.mv, ND);
        CUDA_CHECK(cudaMemcpy(hmv, m.mv, 8, cudaMemcpyDeviceToHost));
    }
    if (c.latent > 0) f32_of_bf16_kernel<<<blocksL, TPB>>>(m.tgtX, m.dEnc, ND);
    // ---- Backward ----
    CUDA_CHECK(cudaMemset(m.dXres, 0, ND * 4));
    ce_bwd_kernel<<<blocks, TPB>>>(m.probs, m.nxt, m.dlogits, c.ce, N);
    linear_dW(N, 256, D, m.dlogits, m.X, m.params.at(m.dec).grad);
    linear_dX(N, 256, D, m.dlogits, m.params.at(m.dec).work, m.dXs);
    cast_add_kernel<<<blocksL, TPB>>>(m.dXs, m.dXres, ND);  // bf16->fp32
    stop_bwd_kernel<<<blocks, TPB>>>(m.stoplog, m.end, m.dstop, c.stopposw,
                                     c.stop, N);
    linear_dW(N, 1, D, m.dstop, m.X, m.params.at(m.stop).grad);
    linear_dX(N, 1, D, m.dstop, m.params.at(m.stop).work, m.dXs);
    cast_add_kernel<<<blocksL, TPB>>>(m.dXs, m.dXres, ND);
    // Latent-Anteil (X-f32 nach dEncTmp; tgt-f32 steht noch in dEnc)
    f32_of_bf16_kernel<<<blocksL, TPB>>>(m.X, m.dEncTmp, ND);
    latent_bwd_kernel<<<blocksL, TPB>>>(m.dEncTmp, m.dEnc, m.dXres, c.latent,
                                        ND);
    // Var-Anteil (nur aktiv wenn Hinge greift)
    var_bwd_kernel<<<blocksL, TPB>>>(
        m.dEncTmp, m.dXres, hmv[0],
        (hmv[1] + 1e-4f >= 1.0f)
            ? 0.0f
            : -0.5f / sqrtf(hmv[1] + 1e-4f) * c.var,
        ND);
    if (m.dXext) add_f32_kernel<<<blocksL, TPB>>>(m.dXres, m.dXext, ND);
    if (c.stop > 0) {
        stop_bwd_kernel<<<blocks, TPB>>>(m.stoplog, m.end, m.dstop, c.stopposw, c.stop, N);
        linear_dW(N, 1, D, m.dstop, m.X, m.params.at(m.stop).grad);
        linear_dX(N, 1, D, m.dstop, m.params.at(m.stop).work, m.dXs);
        cast_add_kernel<<<blocksL, TPB>>>(m.dXs, m.dXres, ND);
    }
    if (c.latent > 0 || c.var > 0) f32_of_bf16_kernel<<<blocksL, TPB>>>(m.X, m.dEncTmp, ND);
    // Latent part (X in dEncTmp, target in dEnc)
    if (c.latent > 0) latent_bwd_kernel<<<blocksL, TPB>>>(m.dEncTmp, m.dEnc, m.dXres, c.latent, ND);
    // Variance hinge (active only below unit variance)
    if (c.var > 0)
        var_bwd_kernel<<<blocksL, TPB>>>(m.dEncTmp, m.dXres, hmv[0],
            (hmv[1] + 1e-4f >= 1.0f) ? 0.0f : -0.5f / sqrtf(hmv[1] + 1e-4f) * c.var, ND);
    // Residual-Stream zurück nach bf16; dEnc neu für State-Pfad
    copy_bf16_kernel<<<blocksL, TPB>>>(m.dXres, m.dXs, ND);
    CUDA_CHECK(cudaMemset(m.dEnc, 0, ND * 4));
    float* dWxp[16];
    if (c.mem) CUDA_CHECK(cudaMemset(m.MS.dEnc, 0, (long)B * c.mem_len * D * 4));
    for (int l = (int)m.L.size() - 1; l >= 0; --l) {
        Layer& Ly = m.L[l];
        // Memory was last in the forward, so it comes first here.
        if (c.mem && m.MEM[l].use) {
            MemLayer& Me = m.MEM[l];
            auto G = [&](size_t p) { return m.params.at(p).grad; };
            mem_backward(m.dXs, Me, m.MS, m.params.at(Me.gamma).master, m.params.at(Me.wq).work,
                         m.params.at(Me.wk).work, m.params.at(Me.wv).work, m.params.at(Me.wo).work,
                         G(Me.gamma), G(Me.beta), G(Me.wq), G(Me.wk), G(Me.wv), G(Me.wo),
                         B, T, c.mem_len, c.mem_heads, c.mem_dh, D);
        }
        // MLA zuerst (war zuletzt im Forward)
        if (c.mla && m.ML[l].use) {
            MlaLayer& Ml = m.ML[l];
            CUDA_CHECK(cudaMemcpy(m.dM, m.dXs, ND * 2,
                                  cudaMemcpyDeviceToDevice));
            float sc = 1.0f / sqrtf((float)c.mla_dh);
            mla_backward(m.dM, N, m.params.at(Ml.p.q).work, m.params.at(Ml.p.dkv).work,
                         m.params.at(Ml.p.kr).work, m.params.at(Ml.p.uk).work, m.params.at(Ml.p.uv).work,
                         m.params.at(Ml.p.o).work, m.params.at(Ml.p.q).grad, m.params.at(Ml.p.dkv).grad,
                         m.params.at(Ml.p.kr).grad, m.params.at(Ml.p.uk).grad, m.params.at(Ml.p.uv).grad,
                         m.params.at(Ml.p.o).grad, state.cache[l], Ml.keep, m.MW, m.dH,
                         Ml.keep.Xq, m.pos, B, T, c.mla_heads, c.mla_dh,
                         c.mla_L, c.mla_R, c.mla_cc, c.mla_theta, sc, D);
            layernorm_bwd(Ml.Xsnap, m.dH, m.params.at(Ml.p.gamma).master, Ml.mmean,
                          Ml.mrstd, m.dSnorm, m.params.at(Ml.p.gamma).grad,
                          m.params.at(Ml.p.beta).grad, N, D);
            add_bf16_kernel<<<blocksL, TPB>>>(m.dXs, m.dSnorm, ND);
        }
        // dM = dXs-Kopie; Residual-Passthrough bleibt in dXs stehen
        CUDA_CHECK(cudaMemcpy(m.dM, m.dXs, ND * 2, cudaMemcpyDeviceToDevice));
        exp_ptrs(m, l, Wx);
        exp_gptrs(m, l, dWxp);
        moe_backward(m.dM, Wx, m.params.at(Ly.router).work, m.params.at(Ly.router).grad, dWxp,
                     m.dH, Ly.mc, m.moeW, c.aux / c.layers, c.zloss / c.layers);
        copy_bf16_kernel<<<blocksL, TPB>>>(Ly.S, m.Sb, ND);
        layernorm_bwd(m.Sb, m.dH, m.params.at(Ly.gamma).master, Ly.mean, Ly.rstd,
                      m.dSnorm, m.params.at(Ly.gamma).grad, m.params.at(Ly.beta).grad, N, D);
        const float* gate = c.gated ? m.params.at(Ly.gate).master : nullptr;
        CellOpt opt; opt.ids = m.ids; opt.docsep = c.docsep; opt.gamma = c.trace_decay;
        if (c.traces) {
            opt.trDec = state.tdec[l]; opt.trGate = state.tgate[l];
            if (l == 0) { opt.lam = m.lam; opt.prod = m.prod; }
            if (logging) {
                opt.logDec = m.trlog + (long)l * D; opt.logGate = m.trlog + (long)(c.layers + l) * D;
                opt.logEmb = m.trlog + 2L * c.layers * D;
            }
        }
        cell_backward(m.dSnorm, Ly.S, m.params.at(Ly.decay).master, m.dEncTmp,
                      m.params.at(Ly.decay).grad, B, T, D, Ly.input, Ly.initial,
                      gate, m.params.at(Ly.gate).grad, opt);
        if (c.traces && l == 0)
            emb_trace(Ly.input, Ly.S, Ly.initial, m.params.at(Ly.decay).master, gate,
                      state.temb, m.params.at(m.emb).grad, B, T, D, opt);
        // Chain through the previous layer, including residual passthrough.
        cast_add_kernel<<<blocksL, TPB>>>(m.dXs, m.dEncTmp, ND);
        copy_bf16_kernel<<<blocksL, TPB>>>(m.dEncTmp, m.dXs, ND);
    }
    if (c.mem)
        mem_encode_backward(m.MS, m.params.at(m.MS.decay).master, m.params.at(m.MS.gate).master,
                            m.params.at(m.MS.decay).grad, m.params.at(m.MS.gate).grad,
                            m.params.at(m.emb).grad, m.params.at(m.MS.eprev).grad,
                            m.params.at(m.MS.pos).grad, B, c.mem_len, D);
    // Residual-Start + Embed-Grade
    cast_add_kernel<<<blocksL, TPB>>>(m.dXs, m.dEnc, ND);
    emb_backward(m.dEnc, m.ids, m.params.at(m.emb).grad, N, D);
}


static void release_window(Model& m) {
    (void)m;  // Arenas persistieren bewusst (kein malloc/free pro Fenster)
}

static void optimizer_step(Model& m, int step) {
    const Cfg& c = m.c;
    const float b1 = .9f, b2 = .999f;
    CUDA_CHECK(cudaMemsetAsync(m.opt.sumsq, 0, sizeof(double)));
    mt_sumsq_kernel<<<m.opt.nchunks, 256>>>(m.opt);
    mt_adam_kernel<<<m.opt.nchunks, 256>>>(m.opt, c.gradclip, lr_at(c, step), b1, b2, 1e-8f, .01f,
                                           1.0f - powf(b1, step + 1), 1.0f - powf(b2, step + 1));
    auto& tgt = m.params.at(m.tgt);
    ema_kernel<<<(tgt.n + 255) / 256, 256>>>(tgt.master, m.params.at(m.emb).master, c.ematau, tgt.n);
    copy_bf16_kernel<<<(tgt.n + 255) / 256, 256>>>(tgt.master, tgt.work, tgt.n);
    double sumsq;  // the only host synchronization of the step
    CUDA_CHECK(cudaMemcpy(&sumsq, m.opt.sumsq, sizeof(double), cudaMemcpyDeviceToHost));
    if (!std::isfinite(sumsq)) throw std::runtime_error("non-finite gradient; update refused");
}

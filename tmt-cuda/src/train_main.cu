#pragma once
// S2-Harness Teil 2: Elementar-Kernel, Build, Fenster-Step, Main, ckpt.
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
struct Layer {
    size_t decay, gamma, beta, router;
    size_t exp[16];
    float *S, *mean, *rstd, *carry;  // S(N,D), stats(N), carry(B,D)
    MoeCache mc;
};
struct Model {
    Cfg c;
    size_t emb, tgt, dec, stop;
    std::vector<Layer> L;
    bf16 *enc, *X, *Sb, *H, *M, *dXs, *dM, *dSnorm, *dH;
    bf16 *logits, *dlogits, *stoplog, *dstop, *tgtX;
    float *dXres, *dEnc, *dEncTmp, *probs, *losstmp, *mv, *dDec;
    int *ids, *nxt, *end;
};

static float lr_at(const Cfg& c, int step) {
    if (step < c.warmup) return c.lr * (step + 1) / (float)c.warmup;
    float p = fminf(1.0f, (step - c.warmup) / (float)c.decaysteps);
    return c.lr * (c.minlr + 0.5f * (1 - c.minlr) *
                   (1 + cosf(3.14159265f * p)));
}

static void build_model(Model& m) {
    Cfg& c = m.c;
    int D = c.dim, L = c.layers, E = c.experts;
    unsigned seed = c.seed ? c.seed : 1234;
    int N = c.batch * c.seqlen;
    long ND = (long)N * D;
    std::vector<float> h;
    auto init_u = [&](size_t p, float a, float b) {
        h.resize(P(p).n); host_init(h.data(), P(p).n, a, b, seed);
        CUDA_CHECK(cudaMemcpy(P(p).master, h.data(), P(p).n * 4,
                              cudaMemcpyHostToDevice));
        long n = P(p).n;
        copy_bf16_kernel<<<(n + 255) / 256, 256>>>(P(p).master, P(p).work, n);
    };
    m.emb = new_par(256L * D);
    init_u(m.emb, -0.05f, 0.05f);
    m.tgt = new_par(256L * D);
    CUDA_CHECK(cudaMemcpy(P(m.tgt).master, P(m.emb).master, 256L * D * 4,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(P(m.tgt).work, P(m.emb).work, 256L * D * 2,
                          cudaMemcpyDeviceToDevice));
    m.dec = new_par(256L * D);
    { float a = sqrtf(1.0f / D); init_u(m.dec, -a, a); }
    m.stop = new_par(1L * D);
    init_u(m.stop, -0.01f, 0.01f);
    m.L.resize(L);
    for (int l = 0; l < L; ++l) {
        Layer& Ly = m.L[l];
        Ly.decay = new_par(D);
        h.assign(D, 2.0f);
        CUDA_CHECK(cudaMemcpy(P(Ly.decay).master, h.data(), D * 4,
                              cudaMemcpyHostToDevice));
        copy_bf16_kernel<<<(D + 255) / 256, 256>>>(
            P(Ly.decay).master, P(Ly.decay).work, D);
        Ly.gamma = new_par(D);
        h.assign(D, 1.0f);
        CUDA_CHECK(cudaMemcpy(P(Ly.gamma).master, h.data(), D * 4,
                              cudaMemcpyHostToDevice));
        copy_bf16_kernel<<<(D + 255) / 256, 256>>>(
            P(Ly.gamma).master, P(Ly.gamma).work, D);
        Ly.beta = new_par(D);
        CUDA_CHECK(cudaMemset(P(Ly.beta).master, 0, D * 4));
        CUDA_CHECK(cudaMemset(P(Ly.beta).work, 0, D * 2));
        Ly.router = new_par((long)E * D);
        h.resize((long)E * D);
        host_normal(h.data(), h.size(), 0.02f, seed);
        CUDA_CHECK(cudaMemcpy(P(Ly.router).master, h.data(),
                              (long)E * D * 4, cudaMemcpyHostToDevice));
        {
            long n = P(Ly.router).n;
            copy_bf16_kernel<<<(n + 255) / 256, 256>>>(
                P(Ly.router).master, P(Ly.router).work, n);
        }
        float a = sqrtf(1.0f / D);
        for (int e = 0; e < E; ++e) {
            Ly.exp[e] = new_par((long)D * D);
            init_u(Ly.exp[e], -a, a);
        }
        CUDA_CHECK(cudaMalloc(&Ly.S, ND * 4));
        CUDA_CHECK(cudaMalloc(&Ly.mean, N * 4));
        CUDA_CHECK(cudaMalloc(&Ly.rstd, N * 4));
        CUDA_CHECK(cudaMalloc(&Ly.carry, (long)c.batch * D * 4));
        CUDA_CHECK(cudaMemset(Ly.carry, 0, (long)c.batch * D * 4));
    }
    CUDA_CHECK(cudaMalloc(&m.enc, ND * 2));
    CUDA_CHECK(cudaMalloc(&m.X, ND * 2));
    CUDA_CHECK(cudaMalloc(&m.Sb, ND * 2));
    CUDA_CHECK(cudaMalloc(&m.H, ND * 2));
    CUDA_CHECK(cudaMalloc(&m.M, ND * 2));
    CUDA_CHECK(cudaMalloc(&m.dXs, ND * 2));
    CUDA_CHECK(cudaMalloc(&m.dM, ND * 2));
    CUDA_CHECK(cudaMalloc(&m.dSnorm, ND * 2));
    CUDA_CHECK(cudaMalloc(&m.dH, ND * 2));
    CUDA_CHECK(cudaMalloc(&m.dXres, ND * 4));
    CUDA_CHECK(cudaMalloc(&m.dEnc, ND * 4));
    CUDA_CHECK(cudaMalloc(&m.dEncTmp, ND * 4));
    CUDA_CHECK(cudaMalloc(&m.dDec, (size_t)D * 4));
    CUDA_CHECK(cudaMalloc(&m.ids, N * 4));
    CUDA_CHECK(cudaMalloc(&m.nxt, N * 4));
    CUDA_CHECK(cudaMalloc(&m.end, N * 4));
    CUDA_CHECK(cudaMalloc(&m.logits, (long)N * 256 * 2));
    CUDA_CHECK(cudaMalloc(&m.dlogits, (long)N * 256 * 2));
    CUDA_CHECK(cudaMalloc(&m.stoplog, (long)N * 2));
    CUDA_CHECK(cudaMalloc(&m.dstop, (long)N * 2));
    CUDA_CHECK(cudaMalloc(&m.probs, (long)N * 256 * 4));
    CUDA_CHECK(cudaMalloc(&m.losstmp, (long)N * 4));
    CUDA_CHECK(cudaMalloc(&m.tgtX, ND * 2));
    CUDA_CHECK(cudaMalloc(&m.mv, 8));
}

static float host_sum(float* d, int n) {
    static std::vector<float> h;
    h.resize(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, (size_t)n * 4, cudaMemcpyDeviceToHost));
    double s = 0;
    for (int i = 0; i < n; ++i) s += h[i];
    return (float)s;
}

// Per-Experten bf16-Zeigerliste (Host) für moe_forward/backward.
static void exp_ptrs(Model& m, int l, bf16** out) {
    for (int e = 0; e < m.c.experts; ++e) out[e] = P(m.L[l].exp[e]).work;
}
static void exp_gptrs(Model& m, int l, float** out) {
    for (int e = 0; e < m.c.experts; ++e) out[e] = P(m.L[l].exp[e]).grad;
}

// ---------- Fenster-Step (Forward + Backward + Adam) ----------
static void window_step(Model& m, int step, float& tot, float& ce_out) {
    Cfg& c = m.c;
    int B = c.batch, T = c.seqlen, D = c.dim, E = c.experts, K = c.topk;
    int N = B * T;
    long ND = (long)N * D;
    const int TPB = 256;
    long blocksL = (ND + TPB - 1) / TPB;
    int blocks = (N + TPB - 1) / TPB;
    for (auto& p : PARS) CUDA_CHECK(cudaMemset(p.grad, 0, p.n * 4));

    // ---- Forward ----
    emb_forward(P(m.emb).work, m.ids, m.enc, N, D);
    CUDA_CHECK(cudaMemcpy(m.X, m.enc, ND * 2, cudaMemcpyDeviceToDevice));
    float aux_acc = 0;
    bf16* Wx[16];
    for (size_t l = 0; l < m.L.size(); ++l) {
        Layer& Ly = m.L[l];
        state_forward(m.enc, Ly.S, P(Ly.decay).master, Ly.carry, B, T, D);
        copy_bf16_kernel<<<blocksL, TPB>>>(Ly.S, m.Sb, ND);
        layernorm_fwd(m.Sb, P(Ly.gamma).master, P(Ly.beta).master, m.H,
                      Ly.mean, Ly.rstd, N, D);
        exp_ptrs(m, (int)l, Wx);
        aux_acc += moe_forward(m.H, P(Ly.router).work, Wx, m.M,
                               Ly.mc, N, E, K, D);
        add_bf16_kernel<<<blocksL, TPB>>>(m.X, m.M, ND);
        // carry-out
        dim3 gc(B, (D + TPB - 1) / TPB);
        extract_carry_kernel<<<gc, TPB>>>(Ly.S, Ly.carry, B, T, D);
    }
    // Decode
    linear_fwd(N, 256, D, m.X, P(m.dec).work, m.logits);
    linear_fwd(N, 1, D, m.X, P(m.stop).work, m.stoplog);
    // CE
    ce_fwd_kernel<<<blocks, TPB>>>(m.logits, m.nxt, m.probs, m.losstmp, N);
    float ce_sum = host_sum(m.losstmp, N);
    ce_out = ce_sum / N;
    // Stop
    stop_fwd_kernel<<<blocks, TPB>>>(m.stoplog, m.end, m.losstmp, c.stopposw,
                                     N);
    float stop_mean = host_sum(m.losstmp, N) / N;
    // Latent-Target
    emb_forward(P(m.tgt).work, m.nxt, m.tgtX, N, D);
    // Var-Hinge
    meanvar_kernel<<<1, 256>>>(m.X, m.mv, ND);
    float hmv[2];
    CUDA_CHECK(cudaMemcpy(hmv, m.mv, 8, cudaMemcpyDeviceToHost));
    float var_loss = fmaxf(0.0f, 1.0f - sqrtf(hmv[1] + 1e-4f));
    // Latent-MSE-Wert
    f32_of_bf16_kernel<<<blocksL, TPB>>>(m.X, m.dXres, ND);  // reuse als tmp
    // (mse via Kernel unten nach tgtX-f32? tgtX bf16 -> cast nötig: reuse dEnc)
    f32_of_bf16_kernel<<<blocksL, TPB>>>(m.tgtX, m.dEnc, ND);
    mse_mean_kernel<<<1, 256>>>(m.dXres, m.dEnc, m.mv, ND);
    float hmse[1];
    CUDA_CHECK(cudaMemcpy(hmse, m.mv, 4, cudaMemcpyDeviceToHost));
    tot = c.var * var_loss + c.latent * hmse[0] + c.ce * ce_out +
          c.stop * stop_mean + c.aux * aux_acc / m.L.size();

    // ---- Backward ----
    CUDA_CHECK(cudaMemset(m.dXres, 0, ND * 4));
    ce_bwd_kernel<<<blocks, TPB>>>(m.probs, m.nxt, m.dlogits, c.ce, N);
    linear_dW(N, 256, D, m.dlogits, m.X, P(m.dec).grad);
    linear_dX(N, 256, D, m.dlogits, P(m.dec).work, m.dXs);
    cast_add_kernel<<<blocksL, TPB>>>(m.dXs, m.dXres, ND);  // bf16->fp32
    stop_bwd_kernel<<<blocks, TPB>>>(m.stoplog, m.end, m.dstop, c.stopposw,
                                     c.stop, N);
    linear_dW(N, 1, D, m.dstop, m.X, P(m.stop).grad);
    linear_dX(N, 1, D, m.dstop, P(m.stop).work, m.dXs);
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
    // Residual-Stream zurück nach bf16; dEnc neu für State-Pfad
    copy_bf16_kernel<<<blocksL, TPB>>>(m.dXres, m.dXs, ND);
    CUDA_CHECK(cudaMemset(m.dEnc, 0, ND * 4));
    float* dWxp[16];
    for (int l = (int)m.L.size() - 1; l >= 0; --l) {
        Layer& Ly = m.L[l];
        // dM = dXs-Kopie; Residual-Passthrough bleibt in dXs stehen
        CUDA_CHECK(cudaMemcpy(m.dM, m.dXs, ND * 2, cudaMemcpyDeviceToDevice));
        exp_ptrs(m, l, Wx);
        exp_gptrs(m, l, dWxp);
        moe_backward(m.dM, Wx, P(Ly.router).work, P(Ly.router).grad, dWxp,
                     m.dH, Ly.mc);
        copy_bf16_kernel<<<blocksL, TPB>>>(Ly.S, m.Sb, ND);
        layernorm_bwd(m.Sb, m.dH, P(Ly.gamma).master, Ly.mean, Ly.rstd,
                      m.dSnorm, P(Ly.gamma).grad, P(Ly.beta).grad, N, D);
        cell_backward(m.dSnorm, Ly.S, P(Ly.decay).master, m.dEncTmp,
                      P(Ly.decay).grad, B, T, D);
        add_f32_kernel<<<blocksL, TPB>>>(m.dEnc, m.dEncTmp, ND);
    }
    // Residual-Start + Embed-Grade
    cast_add_kernel<<<blocksL, TPB>>>(m.dXs, m.dEnc, ND);
    emb_backward(m.dEnc, m.ids, P(m.emb).grad, N, D);
}

// ---------- Checkpoints (roh binär: masters + m + v + step) ----------
static void save_ckpt(Model& m, const char* path, int step) {
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f) { std::printf("save open FAIL\n"); exit(1); }
    Cfg& c = m.c;
    int magic = 0x544D5432;
    fwrite(&magic, 4, 1, f);
    fwrite(&c.dim, 4, 1, f); fwrite(&c.layers, 4, 1, f);
    fwrite(&c.experts, 4, 1, f); fwrite(&c.topk, 4, 1, f);
    fwrite(&step, 4, 1, f);
    int npar = (int)PARS.size();
    fwrite(&npar, 4, 1, f);
    std::vector<float> h;
    for (auto& p : PARS) {
        h.resize(p.n);
        CUDA_CHECK(cudaMemcpy(h.data(), p.master, p.n * 4,
                              cudaMemcpyDeviceToHost));
        fwrite(h.data(), 4, p.n, f);
        CUDA_CHECK(cudaMemcpy(h.data(), p.m, p.n * 4,
                              cudaMemcpyDeviceToHost));
        fwrite(h.data(), 4, p.n, f);
        CUDA_CHECK(cudaMemcpy(h.data(), p.v, p.n * 4,
                              cudaMemcpyDeviceToHost));
        fwrite(h.data(), 4, p.n, f);
    }
    fclose(f);
    rename(tmp, path);
}

static int load_ckpt(Model& m, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int magic, dim, layers, experts, topk, step, npar;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x544D5432) {
        fclose(f); return 0;
    }
    fread(&dim, 4, 1, f); fread(&layers, 4, 1, f);
    fread(&experts, 4, 1, f); fread(&topk, 4, 1, f);
    fread(&step, 4, 1, f); fread(&npar, 4, 1, f);
    if (dim != m.c.dim || layers != m.c.layers || experts != m.c.experts ||
        topk != m.c.topk || npar != (int)PARS.size()) {
        std::printf("ckpt passt nicht (dims)\n"); fclose(f); return 0;
    }
    std::vector<float> h;
    for (auto& p : PARS) {
        h.resize(p.n);
        if (fread(h.data(), 4, p.n, f) != (size_t)p.n) {
            fclose(f); return 0;
        }
        CUDA_CHECK(cudaMemcpy(p.master, h.data(), p.n * 4,
                              cudaMemcpyHostToDevice));
        long n = p.n;
        copy_bf16_kernel<<<(n + 255) / 256, 256>>>(p.master, p.work, n);
        fread(h.data(), 4, p.n, f);
        CUDA_CHECK(cudaMemcpy(p.m, h.data(), p.n * 4,
                              cudaMemcpyHostToDevice));
        fread(h.data(), 4, p.n, f);
        CUDA_CHECK(cudaMemcpy(p.v, h.data(), p.n * 4,
                              cudaMemcpyHostToDevice));
    }
    fclose(f);
    // Target-Encoder aus Embed neu ableiten? Nein: tgt ist eigener Param,
    // steht im ckpt. Fertig.
    return step;
}

// ---------- Main ----------
int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: train data ckpt [key=val...]\n");
        return 1;
    }
    const char* data_path = argv[1];
    const char* ckpt_path = argv[2];
    Model m;
    for (int i = 3; i < argc; ++i) {
        const char* kv = argv[i];
        const char* eq = strchr(kv, '=');
        if (!eq) { std::printf("arg braucht key=val: %s\n", kv); return 1; }
        std::string k(kv, eq);
        set_cfg(m.c, k.c_str(), eq + 1);
    }
    Cfg& c = m.c;
    int B = c.batch, T = c.seqlen, D = c.dim, N = B * T;
    std::printf("tmt-train S2: dim=%d L=%d E=%d k=%d B=%d T=%d\n", D, c.layers,
                c.experts, c.topk, B, T);
    build_model(m);
    // Params zählen
    {
        long tot = 0;
        for (auto& p : PARS) tot += p.n;
        std::printf("params(fp32-master, inkl tgt/Adam getrennt): %.1fM\n",
                    tot / 1e6);
    }
    int step0 = load_ckpt(m, ckpt_path);
    std::printf("start step=%d\n", step0);
    // Daten einlesen (RAM reicht: 110 GB; enwik8 100 MB)
    int fd = open(data_path, O_RDONLY);
    if (fd < 0) { std::printf("data open FAIL\n"); return 1; }
    struct stat st;
    fstat(fd, &st);
    size_t fsz = st.st_size;
    if (fsz > (size_t)8 << 30) {
        std::printf("datei >8GB: streaming fehlt noch (S2b)\n");
        return 1;
    }
    unsigned char* raw = (unsigned char*)malloc(fsz);
    size_t got = 0;
    while (got < fsz) {
        ssize_t r = read(fd, raw + got, fsz - got);
        if (r <= 0) { std::printf("data read FAIL\n"); return 1; }
        got += r;
    }
    close(fd);
    std::printf("data: %.1f MB\n", fsz / 1e6);
    long per = (fsz - 1) / B;
    std::vector<int> hids(N), hnxt(N), hend(N);
    int step = step0, carried = 0, logged = 0;
    long pos = 0;
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    auto now_s = [&]() {
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        return (t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) / 1e9;
    };
    while (c.steps == 0 || step < step0 + c.steps) {
        if (pos + T + 1 >= per) {
            pos = 0;  // neue Epoche, carries frisch
            for (auto& Ly : m.L)
                CUDA_CHECK(cudaMemset(Ly.carry, 0, (long)B * D * 4));
            carried = 0;
        }
        if (carried >= c.maxcarry) {
            for (auto& Ly : m.L)
                CUDA_CHECK(cudaMemset(Ly.carry, 0, (long)B * D * 4));
            carried = 0;
        }
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t) {
                int cc = raw[b * per + pos + t];
                int nn = raw[b * per + pos + t + 1];
                hids[b * T + t] = cc;
                hnxt[b * T + t] = nn;
                hend[b * T + t] = (nn == 10) ? 1 : 0;
            }
        CUDA_CHECK(cudaMemcpy(m.ids, hids.data(), (size_t)N * 4,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m.nxt, hnxt.data(), (size_t)N * 4,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m.end, hend.data(), (size_t)N * 4,
                              cudaMemcpyHostToDevice));
        float tot, ce;
        window_step(m, step, tot, ce);
        // Clip + Adam + EMA
        if (c.gradclip > 0) {
            double acc = 0;
            for (auto& p : PARS) {
                float n = 0;
                cublasSnrm2(cublas_handle(), (int)p.n, p.grad, 1, &n);
                acc += (double)n * n;
            }
            float nrm = sqrtf((float)acc);
            if (nrm > c.gradclip) {
                float sc = c.gradclip / nrm;
                for (auto& p : PARS)
                    cublasSscal(cublas_handle(), (int)p.n, &sc, p.grad, 1);
            }
        }
        float lr = lr_at(c, step);
        for (auto& p : PARS)
            adam_step_one(p.master, p.m, p.v, p.work, p.grad, p.n, lr,
                          0.9f, 0.999f, 1e-8f, 0.01f, step + 1);
        {
            long n = 256L * D;
            ema_kernel<<<(n + 255) / 256, 256>>>(
                P(m.tgt).master, P(m.emb).master, c.ematau, n);
            copy_bf16_kernel<<<(n + 255) / 256, 256>>>(
                P(m.tgt).master, P(m.tgt).work, n);
        }
        // MoeCache pro Layer freigeben (Backward verbraucht)
        for (auto& Ly : m.L) moe_cache_free(Ly.mc);
        ++step;
        ++logged;
        pos += T;
        carried += T;
        if (logged % 20 == 0) {
            double dt = now_s();
            std::printf(
                "\n[step %d] loss %.4f CE %.4f BPC %.4f (%dx%d tok, %.0f tok/s, lr %.2e)",
                step, tot, ce, ce / 0.6931f, B, T,
                logged * (double)(B * T) / (dt > 0 ? dt : 1), lr);
            fflush(stdout);
        }
        if (step % c.saveevery == 0) {
            save_ckpt(m, ckpt_path, step);
            std::printf("\n[saved %d]", step);
            fflush(stdout);
        }
    }
    save_ckpt(m, ckpt_path, step);
    std::printf("\nfertig step=%d\n", step);
    return 0;
}

// C API over the CUDA model for the Zig host programs (zig/tmt.zig).
// Opaque handles; C++ exceptions never cross the boundary: functions return
// 0 on success or -1 with the message available from tmt_last_error().
#include "generate.h"
#include "capi.h"
#include <memory>
#include <csignal>

static thread_local std::string last_error;

template<class F> static int guard(F f) {
    try { f(); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return -1; }
    catch (...) { last_error = "unknown error"; return -1; }
}

struct tmt_gen {
    std::unique_ptr<Generator> g;
    std::string config;
};

extern "C" {

const char* tmt_last_error(void) { return last_error.c_str(); }

tmt_gen* tmt_gen_open(const char* checkpoint) {
    tmt_gen* h = nullptr;
    if (guard([&] {
            auto p = std::make_unique<tmt_gen>();
            p->g = std::make_unique<Generator>(checkpoint, 1u);
            p->config = config_text(p->g->file_cfg);
            h = p.release();
        })) return nullptr;
    return h;
}

void tmt_gen_close(tmt_gen* h) { delete h; }

const char* tmt_gen_config(const tmt_gen* h) { return h->config.c_str(); }

int tmt_gen_feed(tmt_gen* h, int byte, float* logits, float* stop_prob) {
    return guard([&] {
        h->g->feed(byte);
        std::memcpy(logits, h->g->logits, 256 * sizeof(float));
        if (stop_prob) *stop_prob = h->g->stop_prob;
    });
}

int tmt_gen_reset(tmt_gen* h) { return guard([&] { h->g->reset(); }); }

long tmt_gen_fed(const tmt_gen* h) { return h->g->fed; }

// ---- training ----
struct tmt_cfg { Cfg c; std::string text; };
struct tmt_model {
    Model m;
    StreamState state;
    std::vector<float> host;  // staging for diagnostics
    std::vector<float*> acc;  // gradient accumulator (kgtrain), lazily allocated
    float* dXext = nullptr;
};

tmt_cfg* tmt_cfg_new(void) { return new tmt_cfg{}; }
tmt_cfg* tmt_cfg_from_checkpoint(const char* path) {
    tmt_cfg* h = nullptr;
    if (guard([&] { h = new tmt_cfg{checkpoint_config(path), {}}; })) return nullptr;
    return h;
}
tmt_cfg* tmt_cfg_copy(const tmt_cfg* c) { return new tmt_cfg{c->c, {}}; }
void tmt_cfg_free(tmt_cfg* c) { delete c; }
int tmt_cfg_set(tmt_cfg* c, const char* key, const char* value) { return guard([&] { set_cfg(c->c, key, value); }); }
int tmt_cfg_validate(const tmt_cfg* c) { return guard([&] { validate_cfg(c->c); }); }
const char* tmt_cfg_text(tmt_cfg* c) { c->text = config_text(c->c); return c->text.c_str(); }

tmt_model* tmt_model_new(const tmt_cfg* c) {
    tmt_model* h = nullptr;
    if (guard([&] {
            auto p = std::make_unique<tmt_model>();
            p->m.c = c->c;
            build_model(p->m);
            build_state(p->state, p->m);
            h = p.release();
        })) return nullptr;
    return h;
}
void tmt_model_free(tmt_model* m) { delete m; }

static Progress to_progress(const tmt_progress* p) {
    Progress q; q.step = p->step; q.cursor = p->cursor; q.epoch = p->epoch;
    q.carried = p->carried; q.data_size = p->data_size; q.data_hash = p->data_hash;
    return q;
}
int tmt_model_load(tmt_model* m, const char* path, tmt_progress* p) {
    return guard([&] {
        Progress q; load_checkpoint(path, m->m, m->state, q);
        *p = {q.step, q.cursor, q.epoch, q.carried, q.data_size, q.data_hash};
    });
}
int tmt_model_load_weights(tmt_model* m, const char* path) {
    return guard([&] { load_weights_only(path, m->m, checkpoint_config(path)); });
}
int tmt_model_save(tmt_model* m, const char* path, const tmt_progress* p) {
    return guard([&] { Progress q = to_progress(p); save_checkpoint(path, m->m, m->state, q); });
}
int tmt_model_reset_state(tmt_model* m) { return guard([&] { reset_state(m->state, m->m.c); }); }

int tmt_model_forward(tmt_model* m, const int* ids, const int* targets, const int* ends, float* loss, float* ce) {
    return guard([&] {
        long N = (long)m->m.c.batch * m->m.c.seqlen;
        CUDA_CHECK(cudaMemcpy(m->m.ids, ids, N * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m->m.nxt, targets, N * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m->m.end, ends, N * 4, cudaMemcpyHostToDevice));
        forward_window(m->m, m->state, *loss, *ce);
    });
}
int tmt_model_losses(tmt_model* m, float* out) {
    return guard([&] {
        int N = m->m.c.batch * m->m.c.seqlen;
        ce_fwd_kernel<<<(N + 255) / 256, 256>>>(m->m.logits, m->m.nxt, m->m.probs, m->m.losstmp, N);
        CUDA_CHECK(cudaMemcpy(out, m->m.losstmp, (size_t)N * 4, cudaMemcpyDeviceToHost));
    });
}
int tmt_model_backward(tmt_model* m, int log_traces) {
    return guard([&] {
        m->m.log_traces = log_traces != 0;
        backward_window(m->m, m->state);
        release_window(m->m);
    });
}
int tmt_model_optimizer_step(tmt_model* m, int step) { return guard([&] { optimizer_step(m->m, step); }); }
int tmt_model_expert_counts(const tmt_model* m, int64_t* out) {
    return guard([&] {
        for (int l = 0; l < m->m.c.layers; ++l)
            for (int e = 0; e < m->m.c.experts; ++e) out[l * m->m.c.experts + e] = m->m.L[l].mc.hcnt[e];
    });
}
static std::vector<float> host_copy(const float* p, long n) {
    std::vector<float> v(n); CUDA_CHECK(cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost)); return v;
}
int tmt_model_trace_stats(tmt_model* h, double* out) {
    return guard([&] {
        Model& m = h->m; const Cfg& c = m.c; int L = c.layers, D = c.dim;
        if (!m.trlog) throw std::runtime_error("trace statistics need traces=1");
        auto log = host_copy(m.trlog, (2L * L + 256) * D);
        auto sums = [&](std::vector<std::pair<size_t, long>> groups, double* r) {
            double pp = 0, ww = 0, pw = 0;
            for (auto [param, logoff] : groups) {
                auto g = host_copy(m.params.at(param).grad, m.params.at(param).n);
                for (size_t i = 0; i < g.size(); ++i) {
                    double p = log[logoff + i], w = g[i] - p;
                    pp += p * p; ww += w * w; pw += p * w;
                }
            }
            r[0] = pp; r[1] = ww; r[2] = pw;
        };
        std::vector<std::pair<size_t, long>> dec, gate;
        for (int l = 0; l < L; ++l) { dec.push_back({m.L[l].decay, (long)l * D}); gate.push_back({m.L[l].gate, (long)(L + l) * D}); }
        sums(dec, out); sums(gate, out + 3); sums({{m.emb, 2L * L * D}}, out + 6);
    });
}
int tmt_model_state_buckets(tmt_model* h, double* sum, int64_t* count) {
    return guard([&] {
        Model& m = h->m; const Cfg& c = m.c; int D = c.dim, B = c.batch;
        const double edges[] = {16, 128, 1024, 8192, 1e300};
        for (int k = 0; k < 5; ++k) { sum[k] = 0; count[k] = 0; }
        for (int l = 0; l < c.layers; ++l) {
            auto decay = host_copy(m.params.at(m.L[l].decay).master, D);
            auto carry = host_copy(h->state.carry[l], (long)B * D);
            for (int d = 0; d < D; ++d) {
                double a = 1 / (1 + exp(-(double)decay[d]));
                double half = a >= 1 ? 1e300 : log(0.5) / log(a);
                int k = 0; while (half >= edges[k]) ++k;
                for (int b = 0; b < B; ++b) sum[k] += fabs(carry[(long)b * D + d]);
                count[k] += B;
            }
        }
    });
}
int tmt_model_param_count(const tmt_model* m) { return (int)m->m.params.values.size(); }
long tmt_model_param_size(const tmt_model* m, int j) { return m->m.params.values.at(j).n; }
const char* tmt_model_param_group(const tmt_model* h, int j) {
    const Model& m = h->m; size_t p = (size_t)j;
    if (p == m.emb) return "embedding";
    if (p == m.dec) return "decoder";
    for (int l = 0; l < m.c.layers; ++l) {
        const Layer& L = m.L[l];
        if (p == L.decay) return "decay";
        if (p == L.gate && m.c.gated) return "gate";
        if (p == L.gamma || p == L.beta) return "norm";
        if (p == L.router && m.c.experts > 1) return "router";
        for (int e = 0; e < m.c.experts; ++e) if (p == L.exp[e]) return "experts";
        if (m.ML[l].use) {
            const MlaP& q = m.ML[l].p;
            for (size_t id : {q.q, q.dkv, q.kr, q.uk, q.uv, q.o, q.gamma, q.beta}) if (p == id) return "mla";
        }
    }
    return "";
}
int tmt_model_param_grad(tmt_model* m, int j, float* out) {
    return guard([&] {
        const Par& p = m->m.params.values.at(j);
        CUDA_CHECK(cudaMemcpy(out, p.grad, p.n * 4, cudaMemcpyDeviceToHost));
    });
}
int tmt_model_param_master(const tmt_model* h, int j, float* out) {
    return guard([&] {
        const Par& p = h->m.params.values.at(j);
        CUDA_CHECK(cudaMemcpy(out, p.master, p.n * 4, cudaMemcpyDeviceToHost));
    });
}
int tmt_model_param_set_grad(tmt_model* h, int j, const float* in) {
    return guard([&] {
        const Par& p = h->m.params.values.at(j);
        CUDA_CHECK(cudaMemcpy(p.grad, in, p.n * 4, cudaMemcpyHostToDevice));
    });
}
static void ensure_acc(tmt_model* h) {
    if (!h->acc.empty()) return;
    h->acc.resize(h->m.params.values.size());
    for (size_t j = 0; j < h->acc.size(); ++j) h->m.memory.allocate(h->acc[j], h->m.params.at(j).n * 4);
}
int tmt_model_acc_zero(tmt_model* h) {
    return guard([&] {
        ensure_acc(h);
        for (size_t j = 0; j < h->acc.size(); ++j)
            CUDA_CHECK(cudaMemset(h->acc[j], 0, h->m.params.at(j).n * 4));
    });
}
int tmt_model_acc_add(tmt_model* h, int j) {
    return guard([&] {
        ensure_acc(h);
        size_t first = j < 0 ? 0 : (size_t)j, last = j < 0 ? h->acc.size() : (size_t)j + 1;
        for (size_t i = first; i < last; ++i) {
            long n = h->m.params.at(i).n;
            add_f32_kernel<<<(n + 255) / 256, 256>>>(h->acc[i], h->m.params.at(i).grad, n);
        }
    });
}
int tmt_model_acc_store(tmt_model* h) {
    return guard([&] {
        ensure_acc(h);
        for (size_t j = 0; j < h->acc.size(); ++j)
            CUDA_CHECK(cudaMemcpy(h->m.params.at(j).grad, h->acc[j], h->m.params.at(j).n * 4, cudaMemcpyDeviceToDevice));
    });
}
int tmt_model_retrieval_param(const tmt_model* h, int which) {
    if (!h->m.c.mem || h->m.c.mem_rdim <= 0) return -1;
    return (int)(which == 0 ? h->m.MS.rq : h->m.MS.rk);
}
int tmt_model_forward_mem(tmt_model* h, const int* ids, const int* targets, const int* mem,
                          float* loss, float* ce) {
    return guard([&] {
        Model& m = h->m;
        long N = (long)m.c.batch * m.c.seqlen;
        std::vector<int> end(N, 0);
        CUDA_CHECK(cudaMemcpy(m.ids, ids, N * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m.nxt, targets, N * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m.end, end.data(), N * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m.MS.ids, mem, (long)m.c.batch * m.c.mem_len * 4, cudaMemcpyHostToDevice));
        reset_state(h->state, m.c);
        forward_window(m, h->state, *loss, *ce);
    });
}
int tmt_model_backward_ext(tmt_model* h, const float* dX) {
    return guard([&] {
        Model& m = h->m;
        long N = (long)m.c.batch * m.c.seqlen * m.c.dim;
        if (dX) {
            if (!h->dXext) m.memory.allocate(h->dXext, N * 4);
            CUDA_CHECK(cudaMemcpy(h->dXext, dX, N * 4, cudaMemcpyHostToDevice));
            m.dXext = h->dXext;
        }
        backward_window(m, h->state);
        m.dXext = nullptr;
        release_window(m);
    });
}
int tmt_model_read_rows(tmt_model* h, const int* last, float* out) {
    return guard([&] {
        Model& m = h->m; int T = m.c.seqlen, D = m.c.dim;
        std::vector<bf16> row(D);
        for (int b = 0; b < m.c.batch; ++b) {
            if (last[b] < 0) continue;
            CUDA_CHECK(cudaMemcpy(row.data(), m.X + ((long)b * T + last[b]) * D, D * 2, cudaMemcpyDeviceToHost));
            for (int d = 0; d < D; ++d) out[(long)b * D + d] = bf2f(row[d]);
        }
    });
}
int tmt_model_logits(tmt_model* h, float* out) {
    return guard([&] {
        Model& m = h->m; long N = (long)m.c.batch * m.c.seqlen;
        std::vector<bf16> v((size_t)N * 256);
        CUDA_CHECK(cudaMemcpy(v.data(), m.logits, v.size() * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < v.size(); ++i) out[i] = bf2f(v[i]);
    });
}
int tmt_dev_alloc(void** p, unsigned long n) { return guard([&] { CUDA_CHECK(cudaMalloc(p, n)); }); }
int tmt_dev_free(void* p) { return guard([&] { CUDA_CHECK(cudaFree(p)); }); }
int tmt_dev_upload(void* dst, const void* src, unsigned long n) {
    return guard([&] { CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice)); });
}
int tmt_dev_download(void* dst, const void* src, unsigned long n) {
    return guard([&] { CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost)); });
}
int tmt_ref_emb_forward(const void* W, const int* ids, void* out, int N, int D) {
    return guard([&] { emb_forward((const bf16*)W, ids, (bf16*)out, N, D); CUDA_CHECK(cudaDeviceSynchronize()); });
}
int tmt_ref_emb_backward(const float* dOut, const int* ids, float* dW, int N, int D) {
    return guard([&] { emb_backward(dOut, ids, dW, N, D); CUDA_CHECK(cudaDeviceSynchronize()); });
}
int tmt_ref_add_f32(float* acc, const float* x, long n) {
    return guard([&] {
        add_f32_kernel<<<(n + 255) / 256, 256>>>(acc, x, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    });
}
int tmt_ref_copy_bf16(const float* src, void* dst, long n) {
    return guard([&] {
        copy_bf16_kernel<<<(n + 255) / 256, 256>>>(src, (bf16*)dst, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    });
}
int tmt_synchronize(void) { return guard([&] { CUDA_CHECK(cudaDeviceSynchronize()); }); }

static volatile sig_atomic_t stop_flag = 0;
static void on_stop(int) { stop_flag = 1; }
void tmt_install_stop_handler(void) { std::signal(SIGINT, on_stop); std::signal(SIGTERM, on_stop); }
int tmt_stop_requested(void) { return stop_flag; }

}  // extern "C"

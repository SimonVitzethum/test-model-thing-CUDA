// gradcheck: DATA CHECKPOINT [len=4096] [window=128] [seqs=4] [offset=0] [key=value ...]
// Compares windowed gradients against the exact BPTT gradient of whole
// sequences, using trained weights. For each parameter type it reports the
// cosine similarity of truncated BPTT (traces=0) and the hybrid (traces=1)
// to full BPTT over `len` bytes. Nothing is trained or written.
#include "checkpoint.h"
#include <fstream>

struct Group { const char* name; std::vector<size_t> params; };

static std::vector<Group> groups(const Model& m) {
    std::vector<Group> g{{"decay", {}}, {"gate", {}}, {"embedding", {m.emb}}, {"norm", {}},
                         {"router", {}}, {"experts", {}}, {"decoder", {m.dec}}, {"mla", {}}};
    for (int l = 0; l < m.c.layers; ++l) {
        const Layer& Ly = m.L[l];
        g[0].params.push_back(Ly.decay);
        if (m.c.gated) g[1].params.push_back(Ly.gate);
        g[3].params.push_back(Ly.gamma); g[3].params.push_back(Ly.beta);
        if (m.c.experts > 1) g[4].params.push_back(Ly.router);
        for (int e = 0; e < m.c.experts; ++e) g[5].params.push_back(Ly.exp[e]);
        if (m.ML[l].use) {
            const MlaP& p = m.ML[l].p;
            for (size_t id : {p.q, p.dkv, p.kr, p.uk, p.uv, p.o, p.gamma, p.beta}) g[7].params.push_back(id);
        }
    }
    return g;
}

// Accumulated gradients of one configuration over all windows of the sequences.
static std::vector<std::vector<float>> run(const Cfg& cfg, const Cfg& file_cfg, const std::string& path,
                                           const std::vector<unsigned char>& data,
                                           const std::vector<size_t>& starts, int len, double& ce_mean) {
    Model m; m.c = cfg; build_model(m);
    load_weights_only(path, m, file_cfg);
    StreamState state; build_state(state, m);
    int B = cfg.batch, T = cfg.seqlen, N = B * T;
    std::vector<float*> acc(m.params.values.size());
    for (size_t j = 0; j < acc.size(); ++j) {
        m.memory.allocate(acc[j], m.params.at(j).n * 4);
        CUDA_CHECK(cudaMemset(acc[j], 0, m.params.at(j).n * 4));
    }
    std::vector<int> ids(N), nxt(N), end(N);
    double ce_sum = 0;
    for (int w = 0; w < len / T; ++w) {
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t) {
                size_t o = starts[b] + (size_t)w * T + t;
                ids[b * T + t] = data[o]; nxt[b * T + t] = data[o + 1]; end[b * T + t] = data[o + 1] == 10;
            }
        CUDA_CHECK(cudaMemcpy(m.ids, ids.data(), N * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m.nxt, nxt.data(), N * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m.end, end.data(), N * 4, cudaMemcpyHostToDevice));
        float loss, ce; forward_window(m, state, loss, ce);
        backward_window(m, state);
        ce_sum += ce;
        for (size_t j = 0; j < acc.size(); ++j) {
            long n = m.params.at(j).n;
            add_f32_kernel<<<(n + 255) / 256, 256>>>(acc[j], m.params.at(j).grad, n);
        }
        release_window(m);
    }
    ce_mean = ce_sum / (len / T);
    std::vector<std::vector<float>> out;
    for (size_t j = 0; j < acc.size(); ++j) {
        out.emplace_back(m.params.at(j).n);
        CUDA_CHECK(cudaMemcpy(out.back().data(), acc[j], m.params.at(j).n * 4, cudaMemcpyDeviceToHost));
    }
    return out;
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::printf("usage: gradcheck DATA CHECKPOINT [len=4096] [window=128] [seqs=4] [offset=0] [key=value ...]\n");
            return 1;
        }
        std::string path = argv[2];
        Cfg file_cfg = checkpoint_config(path), cfg = file_cfg;
        int len = 4096, window = 128, seqs = 4;
        size_t offset = 0;
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i]; size_t eq = a.find('=');
            if (eq == std::string::npos) throw std::runtime_error("expected key=value");
            std::string k = a.substr(0, eq), v = a.substr(eq + 1);
            if (k == "len") len = std::stoi(v);
            else if (k == "window") window = std::stoi(v);
            else if (k == "seqs") seqs = std::stoi(v);
            else if (k == "offset") offset = std::stoul(v);
            else set_cfg(cfg, k, v);
        }
        if (len <= 0 || window <= 0 || seqs <= 0 || len % window)
            throw std::runtime_error("require len, window, seqs > 0 and len divisible by window");
        std::ifstream in(argv[1], std::ios::binary);
        std::vector<unsigned char> data((std::istreambuf_iterator<char>(in)), {});
        if (data.size() < offset + (size_t)seqs * (len + 1)) throw std::runtime_error("dataset too small");
        size_t stride = (data.size() - offset - len - 1) / seqs;
        std::vector<size_t> starts;
        for (int s = 0; s < seqs; ++s) starts.push_back(offset + s * stride);

        cfg.batch = seqs;  // with mla=1, len must not exceed mla_cache
        Cfg full = cfg, tbptt = cfg, hybrid = cfg;
        full.seqlen = len; full.traces = 0;
        tbptt.seqlen = window; tbptt.traces = 0;
        hybrid.seqlen = window; hybrid.traces = 1;
        for (Cfg* c : {&full, &tbptt, &hybrid}) validate_cfg(*c);
        double ce_full, ce_t, ce_h;
        auto gf = run(full, file_cfg, path, data, starts, len, ce_full);
        auto gt = run(tbptt, file_cfg, path, data, starts, len, ce_t);
        auto gh = run(hybrid, file_cfg, path, data, starts, len, ce_h);

        Model shape; shape.c = full; build_model(shape);
        std::printf("gradcheck: len=%d window=%d seqs=%d trace_decay=%g docsep=%d  CE full=%.4f windowed=%.4f\n",
                    len, window, seqs, cfg.trace_decay, cfg.docsep, ce_full, ce_t);
        std::printf("%-10s %12s %12s %14s\n", "params", "cos TBPTT", "cos hybrid", "|hybrid|/|full|");
        // Windowed sums of per-window means equal (len/window) x the full-window mean.
        double scale = (double)len / window;
        for (auto& g : groups(shape)) {
            if (g.params.empty()) continue;
            double ff = 0, tt = 0, hh = 0, ft = 0, fh = 0;
            for (size_t j : g.params)
                for (size_t i = 0; i < gf[j].size(); ++i) {
                    double f = gf[j][i] * scale, t = gt[j][i], h = gh[j][i];
                    ff += f * f; tt += t * t; hh += h * h; ft += f * t; fh += f * h;
                }
            if (ff == 0) continue;
            std::printf("%-10s %12.4f %12.4f %14.4f\n", g.name, ft / std::max(sqrt(ff * tt), 1e-30),
                        fh / std::max(sqrt(ff * hh), 1e-30), sqrt(hh / ff));
        }
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}

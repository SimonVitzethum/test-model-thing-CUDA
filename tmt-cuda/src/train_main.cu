// CUDA training/evaluation CLI. Both modes use the same model forward path.
#include "checkpoint.h"
#include <chrono>
#include <csignal>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

struct Dataset {
    int fd = -1;
    size_t size = 0;
    unsigned char* bytes = nullptr;
    uint64_t hash = HASH_INIT;
    explicit Dataset(const char* path) {
        fd = open(path, O_RDONLY);
        struct stat st{};
        if (fd < 0 || fstat(fd, &st) || st.st_size < 2) {
            if (fd >= 0) close(fd);
            throw std::runtime_error("dataset missing or shorter than two bytes");
        }
        size = st.st_size;
        void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) { close(fd); throw std::runtime_error("dataset mmap failed"); }
        bytes = static_cast<unsigned char*>(mapped);
        hash = hash_bytes(hash, bytes, size);
    }
    ~Dataset() { if (bytes) munmap(bytes, size); if (fd >= 0) close(fd); }
};
static volatile sig_atomic_t interrupted = 0;
static void stop_requested(int) { interrupted = 1; }
static int integer_option(const std::string& value) {
    int n; std::istringstream in(value);
    if (!(in >> n) || !(in >> std::ws).eof() || n < 0) throw std::runtime_error("invalid nonnegative integer");
    return n;
}
int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::printf("usage: train DATA CHECKPOINT [mode=train|eval] [steps=N] [saveevery=N] [key=value ...]\n");
            return 1;
        }
        std::string path = argv[2], mode = "train";
        struct stat checkpoint_stat{};
        bool exists = stat(path.c_str(), &checkpoint_stat) == 0;
        if (!exists && errno != ENOENT) throw std::runtime_error("cannot stat checkpoint");
        Cfg cfg = exists ? checkpoint_config(path) : Cfg{};
        const std::string saved_config = config_text(cfg);
        int steps = 0, saveevery = 500;
        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i]; size_t eq = arg.find('=');
            if (eq == std::string::npos) throw std::runtime_error("expected key=value");
            auto key = arg.substr(0, eq), value = arg.substr(eq + 1);
            if (key == "mode") mode = value;
            else if (key == "steps") steps = integer_option(value);
            else if (key == "saveevery") saveevery = integer_option(value);
            else set_cfg(cfg, key, value);
        }
        if (mode != "train" && mode != "eval") throw std::runtime_error("mode must be train or eval");
        bool evaluation = mode == "eval";
        if (evaluation && !exists) throw std::runtime_error("evaluation requires an existing checkpoint");
        validate_cfg(cfg);
        if (exists && config_text(cfg) != saved_config)
            throw std::runtime_error("configuration differs from checkpoint; choose a new checkpoint path");
        Dataset data(argv[1]);
        if (data.size < (size_t)cfg.batch * (evaluation ? 2 : cfg.seqlen + 1))
            throw std::runtime_error("dataset too small for batch/seqlen");
        Model m; m.c = cfg; build_model(m);
        StreamState state; build_state(state, m);
        Progress progress;
        if (exists) load_checkpoint(path, m, state, progress);
        if (!evaluation && exists && (progress.data_size != data.size || progress.data_hash != data.hash))
            throw std::runtime_error("resume dataset differs from checkpoint");
        if (evaluation) { reset_state(state, cfg); progress = Progress{}; }
        progress.data_size = data.size; progress.data_hash = data.hash;
        int B = cfg.batch, T = cfg.seqlen, N = B * T;
        size_t per = data.size / B;
        if (!evaluation && (progress.cursor % T || progress.cursor > ((per - 1) / T) * T))
            throw std::runtime_error("invalid checkpoint dataset cursor");
        std::vector<int> ids(N), targets(N), ends(N), valid(N);
        std::vector<float> losses(N);
        uint64_t begin_step = progress.step, measured = 0;
        double ce_sum = 0;
        // Router-Balance-Akku (gegen MoE-Collapse)
        std::vector<std::vector<long>> use_acc(cfg.layers,
                                               std::vector<long>(cfg.experts, 0));
        long use_win = 0;
        auto begin = std::chrono::steady_clock::now();
        std::signal(SIGINT, stop_requested); std::signal(SIGTERM, stop_requested);
        std::printf("CUDA hierarchical byte model: D=%d L=%d E=%d k=%d B=%d T=%d mode=%s step=%llu\n",
                    cfg.dim, cfg.layers, cfg.experts, cfg.topk, B, T, mode.c_str(),
                    (unsigned long long)progress.step);
        while (!interrupted && (!steps || progress.step - begin_step < (uint64_t)steps)) {
            if (progress.step >= INT_MAX - 1) throw std::runtime_error("optimizer step limit reached");
            if (!evaluation && progress.cursor + T >= per) {
                progress.cursor = 0; ++progress.epoch;
                progress.carried = 0; reset_state(state, cfg);
            }
            if (evaluation && progress.cursor >= (data.size + B - 1) / B - 1) break;
            if (cfg.maxcarry && progress.carried >= (uint64_t)cfg.maxcarry) {
                progress.carried = 0; reset_state(state, cfg);
            }
            uint64_t count = 0;
            for (int b = 0; b < B; ++b) {
                size_t start = evaluation ? data.size * b / B : per * b;
                size_t end = evaluation ? data.size * (b + 1) / B : start + per;
                for (int t = 0; t < T; ++t) {
                    int i = b * T + t;
                    size_t offset = start + progress.cursor + t;
                    valid[i] = offset + 1 < end;
                    ids[i] = valid[i] ? data.bytes[offset] : 0;
                    targets[i] = valid[i] ? data.bytes[offset + 1] : 0;
                    ends[i] = targets[i] == 10;
                    count += valid[i];
                }
            }
            CUDA_CHECK(cudaMemcpy(m.ids, ids.data(), N * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(m.nxt, targets.data(), N * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(m.end, ends.data(), N * 4, cudaMemcpyHostToDevice));
            float loss, ce; forward_window(m, state, loss, ce);
            if (!evaluation && cfg.experts > 1) {
                for (int l = 0; l < cfg.layers; ++l)
                    for (int e = 0; e < cfg.experts; ++e)
                        use_acc[l][e] += m.L[l].mc.hcnt[e];
                ++use_win;
            }
            if (evaluation) {
                // Score only real next-byte pairs in the final padded window.
                ce_fwd_kernel<<<(N + 255) / 256, 256>>>(m.logits, m.nxt, m.probs, m.losstmp, N);
                CUDA_CHECK(cudaMemcpy(losses.data(), m.losstmp, N * 4, cudaMemcpyDeviceToHost));
                for (int i = 0; i < N; ++i) if (valid[i]) ce_sum += losses[i];
            } else {
                if (!std::isfinite(loss)) throw std::runtime_error("non-finite loss; update refused");
                backward_window(m, state); optimizer_step(m, progress.step);
                ce_sum += ce * count;
            }
            release_window(m);
            if (!std::isfinite(ce_sum)) throw std::runtime_error("non-finite evaluation score");
            measured += count; ++progress.step; progress.cursor += T; progress.carried += T;
            if (!evaluation && saveevery && progress.step % saveevery == 0)
                save_checkpoint(path, m, state, progress);
            if (progress.step % 20 == 0)
                std::printf("step=%llu loss=%.6f ce=%.6f bpb=%.6f\n", (unsigned long long)progress.step, loss, ce, ce / log(2.));
            if (!evaluation && cfg.experts > 1 && use_win > 0 && progress.step % 100 == 0) {
                long tot = 0;
                for (auto& row : use_acc)
                    for (long v : row) tot += v;
                double mn = 1, mx = 0;
                long dead = 0;
                for (auto& row : use_acc)
                    for (long v : row) {
                        double s = tot ? (double)v / tot * cfg.layers * cfg.experts : 0;
                        mn = std::min(mn, s); mx = std::max(mx, s);
                        if (s < 0.01) ++dead;
                    }
                std::printf("router: min=%.3f max=%.3f dead=%ld/%d (share 1.0=uniform)\n",
                            mn, mx, dead, cfg.layers * cfg.experts);
                for (auto& row : use_acc) std::fill(row.begin(), row.end(), 0);
                use_win = 0;
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        if (!evaluation) save_checkpoint(path, m, state, progress);
        if (!measured) throw std::runtime_error("no byte pairs evaluated");
        // Stable machine-readable record for experiment runners.
        std::printf("{\"mode\":\"%s\",\"steps\":%llu,\"bytes\":%llu,\"ce\":%.9g,\"bpb\":%.9g,\"seconds\":%.6f,\"bytes_per_second\":%.3f}\n",
                    mode.c_str(), (unsigned long long)(progress.step - begin_step),
                    (unsigned long long)measured, ce_sum / measured, ce_sum / measured / log(2.),
                    seconds, measured / std::max(seconds, 1e-9));
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}

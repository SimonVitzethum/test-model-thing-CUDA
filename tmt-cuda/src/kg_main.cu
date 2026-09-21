// kgtrain: train and evaluate the fact memory on knowledge-graph QA data.
//   kgtrain train DATA.tsv CKPT [steps=N] [saveevery=N] [key=value ...]
//   kgtrain eval  DATA.tsv CKPT [memory=on|off|shuffled] [show=N]
// DATA.tsv rows: question <tab> answer <tab> memory [<tab> subject], as written by
// tools/wikidata_kg.py. Each example is one window "question answer\n"; the loss
// covers only the answer bytes and the final newline. The memory holds the
// subject's facts. Evaluation reports exact match (every answer byte is the
// argmax given the correct prefix) and the answer CE per byte.
#include "checkpoint.h"
#include <csignal>
#include <fstream>
#include <random>

struct Example { std::string question, answer, memory; };

static std::vector<Example> load_examples(const std::string& path, uint64_t& size, uint64_t& hash) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::string text((std::istreambuf_iterator<char>(in)), {});
    size = text.size(); hash = hash_bytes(HASH_INIT, text.data(), text.size());
    std::vector<Example> out;
    std::istringstream lines(text); std::string line;
    while (std::getline(lines, line)) {
        std::vector<std::string> f; size_t start = 0, tab;
        while ((tab = line.find('\t', start)) != std::string::npos) { f.push_back(line.substr(start, tab - start)); start = tab + 1; }
        f.push_back(line.substr(start));
        if (f.size() >= 3 && !f[0].empty() && !f[1].empty()) out.push_back({f[0], f[1], f[2]});
    }
    if (out.empty()) throw std::runtime_error("no examples in " + path);
    return out;
}

// Fills window b of the batch. Returns the number of scored positions, or -1
// if the example does not fit into seqlen.
static int fill_window(const Example& e, const std::string& memory, int b, const Cfg& c,
                       std::vector<int>& ids, std::vector<int>& nxt, std::vector<int>& mem) {
    std::string text = e.question + " " + e.answer + "\n";
    int T = c.seqlen, M = c.mem_len, start = (int)e.question.size() + 1;
    if ((int)text.size() > T) return -1;
    int scored = 0;
    for (int t = 0; t < T; ++t) {
        int i = b * T + t;
        ids[i] = t < (int)text.size() ? (unsigned char)text[t] : ' ';
        bool target = t + 1 < (int)text.size() && t + 1 >= start;
        nxt[i] = target ? (unsigned char)text[t + 1] : -1;
        scored += target;
    }
    for (int j = 0; j < M; ++j) mem[b * M + j] = j < (int)memory.size() ? (unsigned char)memory[j] : -1;
    return scored;
}

static void upload_batch(Model& m, const std::vector<int>& ids, const std::vector<int>& nxt,
                         const std::vector<int>& mem) {
    std::vector<int> end(ids.size(), 0);
    CUDA_CHECK(cudaMemcpy(m.ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.nxt, nxt.data(), nxt.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.end, end.data(), end.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.MS.ids, mem.data(), mem.size() * 4, cudaMemcpyHostToDevice));
}

static volatile sig_atomic_t interrupted = 0;
static void stop_requested(int) { interrupted = 1; }

int main(int argc, char** argv) {
    try {
        if (argc < 4 || (std::string(argv[1]) != "train" && std::string(argv[1]) != "eval")) {
            std::printf("usage: kgtrain train DATA.tsv CKPT [steps=N] [saveevery=N] [key=value ...]\n"
                        "       kgtrain eval  DATA.tsv CKPT [memory=on|off|shuffled] [show=N]\n");
            return 1;
        }
        bool training = std::string(argv[1]) == "train";
        std::string path = argv[3];
        uint64_t data_size, data_hash;
        auto examples = load_examples(argv[2], data_size, data_hash);
        bool exists = access(path.c_str(), F_OK) == 0;
        if (!training && !exists) throw std::runtime_error("evaluation requires an existing checkpoint");
        Cfg cfg = exists ? checkpoint_config(path) : Cfg{};
        if (!exists) cfg.mem = 1;
        const std::string saved = config_text(cfg);
        int steps = 0, saveevery = 500, show = 5;
        std::string memory_mode = "on";
        for (int i = 4; i < argc; ++i) {
            std::string a = argv[i]; size_t eq = a.find('=');
            if (eq == std::string::npos) throw std::runtime_error("expected key=value");
            std::string k = a.substr(0, eq), v = a.substr(eq + 1);
            if (k == "steps") steps = std::stoi(v);
            else if (k == "saveevery") saveevery = std::stoi(v);
            else if (k == "memory") memory_mode = v;
            else if (k == "show") show = std::stoi(v);
            else set_cfg(cfg, k, v);
        }
        if (memory_mode != "on" && memory_mode != "off" && memory_mode != "shuffled")
            throw std::runtime_error("memory must be on, off or shuffled");
        validate_cfg(cfg);
        if (!cfg.mem) throw std::runtime_error("kgtrain requires mem=1");
        if (exists && config_text(cfg) != saved)
            throw std::runtime_error("configuration differs from checkpoint; choose a new checkpoint path");
        Model m; m.c = cfg; build_model(m);
        StreamState state; build_state(state, m);
        Progress progress;
        if (exists) load_checkpoint(path, m, state, progress);
        if (training && exists && (progress.data_size != data_size || progress.data_hash != data_hash))
            throw std::runtime_error("resume dataset differs from checkpoint");
        progress.data_size = data_size; progress.data_hash = data_hash; progress.cursor = 0;
        int B = cfg.batch, T = cfg.seqlen, N = B * T, M = cfg.mem_len;
        std::vector<int> ids(N), nxt(N), mem((size_t)B * M);
        std::vector<float> losses(N);
        std::signal(SIGINT, stop_requested); std::signal(SIGTERM, stop_requested);
        auto answer_ce = [&](double& sum, long& count) {
            ce_fwd_kernel<<<(N + 255) / 256, 256>>>(m.logits, m.nxt, m.probs, m.losstmp, N);
            CUDA_CHECK(cudaMemcpy(losses.data(), m.losstmp, N * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < N; ++i) if (nxt[i] >= 0) { sum += losses[i]; ++count; }
        };
        if (training) {
            std::printf("kgtrain: %zu examples, D=%d L=%d mem_len=%d heads=%d every=%d B=%d T=%d step=%llu\n",
                        examples.size(), cfg.dim, cfg.layers, M, cfg.mem_heads, cfg.mem_every, B, T,
                        (unsigned long long)progress.step);
            uint64_t begin = progress.step; double window_ce = 0; long window_count = 0;
            while (!interrupted && (!steps || progress.step - begin < (uint64_t)steps)) {
                std::mt19937 rng((unsigned)(cfg.seed * 2654435761u + progress.step));  // resumable
                std::uniform_int_distribution<size_t> pick(0, examples.size() - 1);
                for (int b = 0; b < B; ++b) {
                    int tries = 0;
                    while (true) {
                        const Example& e = examples[pick(rng)];
                        if (fill_window(e, e.memory, b, cfg, ids, nxt, mem) >= 0) break;
                        if (++tries > 1000) throw std::runtime_error("examples do not fit into seqlen");
                    }
                }
                upload_batch(m, ids, nxt, mem);
                reset_state(state, cfg);
                float loss, ce; forward_window(m, state, loss, ce);
                answer_ce(window_ce, window_count);
                if (!std::isfinite(loss)) throw std::runtime_error("non-finite loss; update refused");
                backward_window(m, state); optimizer_step(m, progress.step); release_window(m);
                ++progress.step;
                if (progress.step % 20 == 0) {
                    std::printf("step=%llu answer_ce=%.4f\n", (unsigned long long)progress.step,
                                window_ce / std::max(window_count, 1L));
                    window_ce = 0; window_count = 0;
                }
                if (saveevery && progress.step % saveevery == 0) { reset_state(state, cfg); save_checkpoint(path, m, state, progress); }
            }
            reset_state(state, cfg);
            save_checkpoint(path, m, state, progress);
            std::printf("{\"mode\":\"train\",\"steps\":%llu}\n", (unsigned long long)(progress.step - begin));
            return 0;
        }
        // ---- evaluation ----
        size_t n = examples.size();
        auto memory_for = [&](size_t i) -> std::string {
            if (memory_mode == "off") return "";
            if (memory_mode == "on") return examples[i].memory;
            for (size_t k = 1; k < n; ++k) {  // another subject's facts, with a different answer
                const Example& o = examples[(i + n / 2 + k) % n];
                if (o.answer != examples[i].answer && o.memory != examples[i].memory) return o.memory;
            }
            return "";
        };
        long exact = 0, evaluated = 0, skipped = 0, count = 0; double ce_sum = 0;
        std::vector<bf16> logits((size_t)N * 256);
        for (size_t first = 0; first < n; first += B) {
            std::vector<size_t> slot(B, SIZE_MAX);
            for (int b = 0; b < B; ++b) {
                size_t i = first + b;
                bool ok = i < n && fill_window(examples[i], memory_for(i), b, cfg, ids, nxt, mem) >= 0;
                if (i < n && !ok) ++skipped;
                if (ok) slot[b] = i;
                else {
                    for (int t = 0; t < T; ++t) { ids[b * T + t] = ' '; nxt[b * T + t] = -1; }
                    for (int j = 0; j < M; ++j) mem[b * M + j] = -1;
                }
            }
            upload_batch(m, ids, nxt, mem);
            reset_state(state, cfg);
            float loss, ce; forward_window(m, state, loss, ce);
            answer_ce(ce_sum, count);
            CUDA_CHECK(cudaMemcpy(logits.data(), m.logits, logits.size() * 2, cudaMemcpyDeviceToHost));
            for (int b = 0; b < B; ++b) {
                if (slot[b] == SIZE_MAX) continue;
                bool all = true; std::string predicted;
                for (int t = 0; t < T; ++t) {
                    int i = b * T + t;
                    if (nxt[i] < 0) continue;
                    int best = 0;
                    for (int k = 1; k < 256; ++k)
                        if (bf2f(logits[(size_t)i * 256 + k]) > bf2f(logits[(size_t)i * 256 + best])) best = k;
                    all &= best == nxt[i];
                    if (best != '\n') predicted += (char)best;
                }
                exact += all; ++evaluated;
                if (evaluated <= show)
                    std::printf("Q: %s | expected: %s | predicted: %s\n", examples[slot[b]].question.c_str(),
                                examples[slot[b]].answer.c_str(), predicted.c_str());
            }
            release_window(m);
        }
        if (!evaluated) throw std::runtime_error("no example fits into seqlen");
        std::printf("{\"mode\":\"eval\",\"memory\":\"%s\",\"examples\":%ld,\"skipped\":%ld,\"exact\":%.4f,\"answer_ce\":%.4f}\n",
                    memory_mode.c_str(), evaluated, skipped, (double)exact / evaluated, ce_sum / std::max(count, 1L));
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}

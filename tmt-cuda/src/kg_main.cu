// kgtrain: train and evaluate the fact memory on knowledge-graph QA data.
//   kgtrain train DATA.tsv CKPT [steps=N] [saveevery=N] [nodes=NODES.tsv] [key=value ...]
//   kgtrain eval  DATA.tsv CKPT [memory=on|off|shuffled|retrieved] [nodes=NODES.tsv] [show=N]
//   kgtrain ask   CKPT nodes=NODES.tsv ["question"] [top=5] [maxlen=80] [temp=0]
//     (stage-2 checkpoint; without a question it reads questions from stdin)
// DATA.tsv rows: question <tab> answer <tab> memory <tab> subject, as written by
// kgprep qa. Each example is one window "question answer\n"; the loss covers
// only the answer bytes and the final newline. Evaluation reports exact match
// (every answer byte is the argmax given the correct prefix) and answer CE.
//
// Stage 1 (memory=on): the data loader puts the subject's facts into the memory.
// Stage 2 (mem_rdim > 0 and nodes=...): the model retrieves the subject itself.
// Its state at the last question byte gives a query, its state at the last byte
// of each node label gives a key (src/retrieval.h); the best-scoring node's
// facts are loaded. Training adds an InfoNCE loss over the batch's subjects and
// `negbatches` batches of random nodes; the answer pass keeps the true facts.
#include "checkpoint.h"
#include "retrieval.h"
#include <csignal>
#include <iostream>
#include <fstream>
#include <random>
#include <unordered_map>

struct Example { std::string question, answer, memory, subject; };
struct Node { std::string qid, label, memory; };

static std::vector<std::vector<std::string>> read_tsv(const std::string& path, uint64_t* size, uint64_t* hash) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::string text((std::istreambuf_iterator<char>(in)), {});
    if (size) *size = text.size();
    if (hash) *hash = hash_bytes(HASH_INIT, text.data(), text.size());
    std::vector<std::vector<std::string>> rows;
    std::istringstream lines(text); std::string line;
    while (std::getline(lines, line)) {
        std::vector<std::string> f; size_t start = 0, tab;
        while ((tab = line.find('\t', start)) != std::string::npos) { f.push_back(line.substr(start, tab - start)); start = tab + 1; }
        f.push_back(line.substr(start));
        rows.push_back(std::move(f));
    }
    return rows;
}

static std::vector<Example> load_examples(const std::string& path, uint64_t& size, uint64_t& hash) {
    std::vector<Example> out;
    for (auto& f : read_tsv(path, &size, &hash))
        if (f.size() >= 3 && !f[0].empty() && !f[1].empty()) out.push_back({f[0], f[1], f[2], f.size() > 3 ? f[3] : ""});
    if (out.empty()) throw std::runtime_error("no examples in " + path);
    return out;
}

static std::vector<Node> load_nodes(const std::string& path) {
    std::vector<Node> out;
    for (auto& f : read_tsv(path, nullptr, nullptr))
        if (f.size() == 3 && !f[1].empty()) out.push_back({f[0], f[1], f[2]});
    if (out.empty()) throw std::runtime_error("no nodes in " + path);
    return out;
}

// Answer window b: "question answer\n", loss on the answer. Returns the number
// of scored positions, or -1 if the example does not fit into seqlen.
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

// Reading window b (question or node label), no loss and no memory. Returns
// the position of the last byte (where the state is read), or -1 if too long.
static int fill_reading(const std::string& text, int b, const Cfg& c, std::vector<int>& ids,
                        std::vector<int>& nxt, std::vector<int>& mem) {
    int T = c.seqlen, M = c.mem_len;
    if (text.empty() || (int)text.size() > T) return -1;
    for (int t = 0; t < T; ++t) { ids[b * T + t] = t < (int)text.size() ? (unsigned char)text[t] : ' '; nxt[b * T + t] = -1; }
    for (int j = 0; j < M; ++j) mem[b * M + j] = -1;
    return (int)text.size() - 1;
}

static void upload_batch(Model& m, const std::vector<int>& ids, const std::vector<int>& nxt,
                         const std::vector<int>& mem) {
    std::vector<int> end(ids.size(), 0);
    CUDA_CHECK(cudaMemcpy(m.ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.nxt, nxt.data(), nxt.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.end, end.data(), end.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.MS.ids, mem.data(), mem.size() * 4, cudaMemcpyHostToDevice));
}

// Final representation rows (b, last[b]) of the current forward, as float.
static std::vector<float> read_rows(Model& m, const std::vector<int>& last) {
    int T = m.c.seqlen, D = m.c.dim;
    std::vector<float> out(last.size() * D);
    std::vector<bf16> row(D);
    for (size_t b = 0; b < last.size(); ++b) {
        if (last[b] < 0) continue;
        CUDA_CHECK(cudaMemcpy(row.data(), m.X + ((long)b * T + last[b]) * D, D * 2, cudaMemcpyDeviceToHost));
        for (int d = 0; d < D; ++d) out[b * D + d] = bf2f(row[d]);
    }
    return out;
}

static std::vector<float> host_master(Model& m, size_t p) {
    std::vector<float> v(m.params.at(p).n);
    CUDA_CHECK(cudaMemcpy(v.data(), m.params.at(p).master, v.size() * 4, cudaMemcpyDeviceToHost));
    return v;
}

// Interactive stage-2 demo: retrieve the subject from the model's own state,
// show the best nodes, load the winner's facts and generate the answer byte by
// byte (autoregressive, not teacher-forced).
static int run_ask(int argc, char** argv) {
    std::string path = argv[2], nodes_path, question;
    int top = 5, maxlen = 80;
    float temp = 0.f;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i]; size_t eq = a.find('=');
        if (eq == std::string::npos) { question = a; continue; }
        std::string k = a.substr(0, eq), v = a.substr(eq + 1);
        if (k == "nodes") nodes_path = v;
        else if (k == "top") top = std::stoi(v);
        else if (k == "maxlen") maxlen = std::stoi(v);
        else if (k == "temp") temp = std::stof(v);
        else throw std::runtime_error("unknown option: " + k);
    }
    Cfg cfg = checkpoint_config(path);
    if (!cfg.mem || cfg.mem_rdim <= 0 || nodes_path.empty())
        throw std::runtime_error("ask needs a stage-2 checkpoint (mem_rdim > 0) and nodes=NODES.tsv");
    auto nodes = load_nodes(nodes_path);
    Model m; m.c = cfg; build_model(m);
    StreamState state; build_state(state, m);
    load_weights_only(path, m, cfg);
    int B = cfg.batch, T = cfg.seqlen, M = cfg.mem_len, D = cfg.dim, R = cfg.mem_rdim, N = B * T;
    std::vector<int> ids(N), nxt(N), mem((size_t)B * M);
    auto read_batch = [&](const std::vector<std::string>& texts, std::vector<int>& last) {
        last.assign(B, -1);
        for (int b = 0; b < B; ++b) {
            if (b < (int)texts.size()) last[b] = fill_reading(texts[b], b, cfg, ids, nxt, mem);
            if (last[b] < 0) fill_reading(" ", b, cfg, ids, nxt, mem);
        }
        upload_batch(m, ids, nxt, mem);
        reset_state(state, cfg);
        float loss, ce; forward_window(m, state, loss, ce);
        return read_rows(m, last);
    };
    std::fprintf(stderr, "indexing %zu nodes...\n", nodes.size());
    auto Wk = host_master(m, m.MS.rk), Wq = host_master(m, m.MS.rq);
    std::vector<float> keys(nodes.size() * R, 0.f);
    for (size_t first = 0; first < nodes.size(); first += B) {
        std::vector<std::string> batch;
        for (size_t i = first; i < std::min(nodes.size(), first + B); ++i) batch.push_back(nodes[i].label);
        std::vector<int> last;
        auto rows = read_batch(batch, last);
        for (size_t b = 0; b < batch.size(); ++b)
            if (last[b] >= 0) { auto u = head_unit(Wk, &rows[b * D], R, D); std::copy(u.begin(), u.end(), keys.begin() + (first + b) * R); }
    }
    std::mt19937 rng(1);
    auto answer = [&](const std::string& q) {
        std::vector<int> last;
        auto rows = read_batch({q}, last);
        if (last[0] < 0) { std::printf("(question longer than seqlen=%d)\n", T); return; }
        auto qv = head_unit(Wq, &rows[0], R, D);
        std::vector<std::pair<float, int>> scored(nodes.size());
        for (size_t k = 0; k < nodes.size(); ++k) {
            float s = 0;
            for (int r = 0; r < R; ++r) s += qv[r] * keys[k * R + r];
            scored[k] = {s, (int)k};
        }
        int shown = std::min<int>(top, (int)scored.size());
        std::partial_sort(scored.begin(), scored.begin() + shown, scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
        std::printf("retrieved:");
        for (int i = 0; i < shown; ++i) std::printf(" %s (%.3f)%s", nodes[scored[i].second].label.c_str(), scored[i].first, i + 1 < shown ? "," : "\n");
        const Node& best = nodes[scored[0].second];
        std::printf("memory:    %s\n", best.memory.c_str());
        // Generate "question answer\n": one window forward per byte, memory = best node's facts.
        std::string text = q + " ", out;
        std::vector<bf16> row(256);
        for (int step = 0; step < maxlen && (int)text.size() < T; ++step) {
            for (int t = 0; t < T; ++t) { ids[t] = t < (int)text.size() ? (unsigned char)text[t] : ' '; nxt[t] = -1; }
            for (int j = 0; j < M; ++j) mem[j] = j < (int)best.memory.size() ? (unsigned char)best.memory[j] : -1;
            for (int b = 1; b < B; ++b) { fill_reading(" ", b, cfg, ids, nxt, mem); }
            upload_batch(m, ids, nxt, mem);
            reset_state(state, cfg);
            float loss, ce; forward_window(m, state, loss, ce);
            CUDA_CHECK(cudaMemcpy(row.data(), m.logits + (size_t)(text.size() - 1) * 256, 256 * 2, cudaMemcpyDeviceToHost));
            int pick = 0;
            if (temp <= 0) { for (int k = 1; k < 256; ++k) if (bf2f(row[k]) > bf2f(row[pick])) pick = k; }
            else {
                double p[256], sum = 0; float mx = -1e30f;
                for (int k = 0; k < 256; ++k) mx = std::max(mx, bf2f(row[k]));
                for (int k = 0; k < 256; ++k) sum += p[k] = exp((bf2f(row[k]) - mx) / temp);
                double r = std::uniform_real_distribution<double>(0, 1)(rng) * sum;
                for (pick = 0; pick < 255 && (r -= p[pick]) > 0; ++pick) {}
            }
            if (pick == '\n') break;
            text += (char)pick; out += (char)pick;
        }
        std::printf("answer:    %s\n", out.c_str());
        std::fflush(stdout);
    };
    if (!question.empty()) { answer(question); return 0; }
    std::string line;
    bool tty = isatty(0);
    while (true) {
        if (tty) { std::printf("question> "); std::fflush(stdout); }
        if (!std::getline(std::cin, line) || line == "/quit") break;
        if (!line.empty()) answer(line);
    }
    return 0;
}

static volatile sig_atomic_t interrupted = 0;
static void stop_requested(int) { interrupted = 1; }

int main(int argc, char** argv) {
    try {
        if (argc >= 3 && std::string(argv[1]) == "ask") return run_ask(argc, argv);
        if (argc < 4 || (std::string(argv[1]) != "train" && std::string(argv[1]) != "eval")) {
            std::printf("usage: kgtrain train DATA.tsv CKPT [steps=N] [saveevery=N] [nodes=NODES.tsv] [key=value ...]\n"
                        "       kgtrain eval  DATA.tsv CKPT [memory=on|off|shuffled|retrieved] [nodes=NODES.tsv] [show=N]\n"
                        "       kgtrain ask   CKPT nodes=NODES.tsv [\"question\"] [top=5] [maxlen=80] [temp=0]\n");
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
        int steps = 0, saveevery = 500, show = 5, negbatches = 1;
        float tau = 0.05f, rweight = 1.f;
        std::string memory_mode = "on", nodes_path;
        for (int i = 4; i < argc; ++i) {
            std::string a = argv[i]; size_t eq = a.find('=');
            if (eq == std::string::npos) throw std::runtime_error("expected key=value");
            std::string k = a.substr(0, eq), v = a.substr(eq + 1);
            if (k == "steps") steps = std::stoi(v);
            else if (k == "saveevery") saveevery = std::stoi(v);
            else if (k == "memory") memory_mode = v;
            else if (k == "show") show = std::stoi(v);
            else if (k == "nodes") nodes_path = v;
            else if (k == "negbatches") negbatches = std::stoi(v);
            else if (k == "tau") tau = std::stof(v);
            else if (k == "rweight") rweight = std::stof(v);
            else set_cfg(cfg, k, v);
        }
        if (memory_mode != "on" && memory_mode != "off" && memory_mode != "shuffled" && memory_mode != "retrieved")
            throw std::runtime_error("memory must be on, off, shuffled or retrieved");
        validate_cfg(cfg);
        if (!cfg.mem) throw std::runtime_error("kgtrain requires mem=1");
        if (exists && config_text(cfg) != saved)
            throw std::runtime_error("configuration differs from checkpoint; choose a new checkpoint path");
        bool retrieval = cfg.mem_rdim > 0 && !nodes_path.empty();
        if (memory_mode == "retrieved" && !retrieval)
            throw std::runtime_error("memory=retrieved requires a checkpoint with mem_rdim > 0 and nodes=...");
        if (negbatches < 0 || tau <= 0) throw std::runtime_error("invalid retrieval options");
        std::vector<Node> nodes;
        std::unordered_map<std::string, int> node_of;
        if (retrieval) {
            nodes = load_nodes(nodes_path);
            for (size_t i = 0; i < nodes.size(); ++i) node_of[nodes[i].qid] = (int)i;
        }
        Model m; m.c = cfg; build_model(m);
        StreamState state; build_state(state, m);
        Progress progress;
        if (exists) load_checkpoint(path, m, state, progress);
        if (training && exists && (progress.data_size != data_size || progress.data_hash != data_hash))
            throw std::runtime_error("resume dataset differs from checkpoint");
        progress.data_size = data_size; progress.data_hash = data_hash; progress.cursor = 0;
        int B = cfg.batch, T = cfg.seqlen, N = B * T, M = cfg.mem_len, D = cfg.dim, R = cfg.mem_rdim;
        std::vector<int> ids(N), nxt(N), mem((size_t)B * M);
        std::vector<float> losses(N);
        std::signal(SIGINT, stop_requested); std::signal(SIGTERM, stop_requested);
        auto answer_ce = [&](double& sum, long& count) {
            ce_fwd_kernel<<<(N + 255) / 256, 256>>>(m.logits, m.nxt, m.probs, m.losstmp, N);
            CUDA_CHECK(cudaMemcpy(losses.data(), m.losstmp, N * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < N; ++i) if (nxt[i] >= 0) { sum += losses[i]; ++count; }
        };
        // Forward one reading batch (questions or labels) and return the rows at the last bytes.
        auto read_batch = [&](const std::vector<std::string>& texts, std::vector<int>& last) {
            last.assign(B, -1);
            for (int b = 0; b < B; ++b) {
                if (b < (int)texts.size()) last[b] = fill_reading(texts[b], b, cfg, ids, nxt, mem);
                if (last[b] < 0) fill_reading(" ", b, cfg, ids, nxt, mem);
            }
            upload_batch(m, ids, nxt, mem);
            reset_state(state, cfg);
            float loss, ce; forward_window(m, state, loss, ce);
            return read_rows(m, last);
        };
        if (training) {
            std::printf("kgtrain: %zu examples, D=%d L=%d mem_len=%d heads=%d every=%d B=%d T=%d step=%llu%s\n",
                        examples.size(), cfg.dim, cfg.layers, M, cfg.mem_heads, cfg.mem_every, B, T,
                        (unsigned long long)progress.step, retrieval ? " retrieval" : "");
            // Gradient accumulator over the passes of one step (backward_window overwrites grads).
            std::vector<float*> acc(m.params.values.size());
            for (size_t j = 0; j < acc.size(); ++j) m.memory.allocate(acc[j], m.params.at(j).n * 4);
            float* dXext; m.memory.allocate(dXext, (long)N * D * 4);
            auto zero_acc = [&] { for (size_t j = 0; j < acc.size(); ++j) CUDA_CHECK(cudaMemset(acc[j], 0, m.params.at(j).n * 4)); };
            auto add_grads = [&] {
                for (size_t j = 0; j < acc.size(); ++j) {
                    long n = m.params.at(j).n;
                    add_f32_kernel<<<(n + 255) / 256, 256>>>(acc[j], m.params.at(j).grad, n);
                }
            };
            // Backward of a reading batch with the retrieval gradient on the read rows.
            auto backward_rows = [&](const std::vector<std::string>& texts, const std::vector<float>& drows) {
                std::vector<int> last;
                read_batch(texts, last);
                std::vector<float> g((size_t)N * D, 0.f);
                for (int b = 0; b < B; ++b)
                    if (last[b] >= 0)
                        for (int d = 0; d < D; ++d) g[((size_t)b * T + last[b]) * D + d] = rweight * drows[(size_t)b * D + d];
                CUDA_CHECK(cudaMemcpy(dXext, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
                m.dXext = dXext; backward_window(m, state); m.dXext = nullptr;
                add_grads(); release_window(m);
            };
            uint64_t begin = progress.step;
            double window_ce = 0, window_rloss = 0, window_racc = 0; long window_count = 0, rsteps = 0;
            while (!interrupted && (!steps || progress.step - begin < (uint64_t)steps)) {
                std::mt19937 rng((unsigned)(cfg.seed * 2654435761u + progress.step));  // resumable
                std::uniform_int_distribution<size_t> pick(0, examples.size() - 1);
                std::vector<size_t> chosen;
                std::vector<std::string> seen;
                for (int b = 0; b < B; ++b) {
                    for (int tries = 0;; ++tries) {
                        if (tries > 10000) throw std::runtime_error("examples do not fit into seqlen");
                        size_t i = pick(rng);
                        const Example& e = examples[i];
                        if (retrieval && (!node_of.count(e.subject) ||
                            fill_reading(e.question, b, cfg, ids, nxt, mem) < 0 ||
                            fill_reading(nodes[node_of[e.subject]].label, b, cfg, ids, nxt, mem) < 0)) continue;
                        if (fill_window(e, e.memory, b, cfg, ids, nxt, mem) < 0) continue;
                        chosen.push_back(i); seen.push_back(e.subject); break;
                    }
                }
                zero_acc();
                // Answer pass with the true facts (stage 1).
                for (int b = 0; b < B; ++b) fill_window(examples[chosen[b]], examples[chosen[b]].memory, b, cfg, ids, nxt, mem);
                upload_batch(m, ids, nxt, mem);
                reset_state(state, cfg);
                float loss, ce; forward_window(m, state, loss, ce);
                answer_ce(window_ce, window_count);
                if (!std::isfinite(loss)) throw std::runtime_error("non-finite loss; update refused");
                backward_window(m, state); add_grads(); release_window(m);
                if (retrieval) {
                    // Candidates: each distinct subject once (duplicates in the batch share
                    // their positive), then random other nodes as negatives.
                    std::vector<std::string> questions, subjects;
                    std::vector<std::vector<std::string>> labels(1 + negbatches);
                    std::vector<int> pos(B);
                    for (int b = 0; b < B; ++b) {
                        const Example& e = examples[chosen[b]];
                        questions.push_back(e.question);
                        auto it = std::find(subjects.begin(), subjects.end(), e.subject);
                        pos[b] = (int)(it - subjects.begin());
                        if (it == subjects.end()) { subjects.push_back(e.subject); labels[0].push_back(nodes[node_of[e.subject]].label); }
                    }
                    std::uniform_int_distribution<size_t> any(0, nodes.size() - 1);
                    for (int j = 0; j <= negbatches; ++j)
                        for (int tries = 0; (int)labels[j].size() < B; ++tries) {
                            const Node& n = nodes[any(rng)];
                            bool fresh = std::find(seen.begin(), seen.end(), n.qid) == seen.end();
                            if ((fresh || tries > 1000) && (int)n.label.size() <= T) labels[j].push_back(n.label);
                        }
                    std::vector<int> last;
                    std::vector<float> xq = read_batch(questions, last), y;
                    for (auto& batch : labels) { auto rows = read_batch(batch, last); y.insert(y.end(), rows.begin(), rows.end()); }
                    RetrievalBatch rb; rb.B = B; rb.C = B * (1 + negbatches); rb.D = D; rb.R = R; rb.tau = tau;
                    std::vector<float> dx, dy; std::vector<double> dWq, dWk;
                    retrieval_loss(rb, xq, y, pos, host_master(m, m.MS.rq), host_master(m, m.MS.rk), dx, dy, dWq, dWk);
                    window_rloss += rb.loss; window_racc += rb.accuracy; ++rsteps;
                    for (size_t j = 0; j < labels.size(); ++j)
                        backward_rows(labels[j], std::vector<float>(dy.begin() + j * B * D, dy.begin() + (j + 1) * B * D));
                    backward_rows(questions, dx);
                    for (auto [p, g] : {std::pair<size_t, std::vector<double>*>{m.MS.rq, &dWq}, {m.MS.rk, &dWk}}) {
                        std::vector<float> h(g->size());
                        for (size_t i = 0; i < h.size(); ++i) h[i] = rweight * (float)(*g)[i];
                        CUDA_CHECK(cudaMemcpy(m.params.at(p).grad, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
                        add_f32_kernel<<<(h.size() + 255) / 256, 256>>>(acc[p], m.params.at(p).grad, (long)h.size());
                    }
                }
                for (size_t j = 0; j < acc.size(); ++j)
                    CUDA_CHECK(cudaMemcpy(m.params.at(j).grad, acc[j], m.params.at(j).n * 4, cudaMemcpyDeviceToDevice));
                optimizer_step(m, progress.step);
                ++progress.step;
                if (progress.step % 20 == 0) {
                    std::printf("step=%llu answer_ce=%.4f", (unsigned long long)progress.step, window_ce / std::max(window_count, 1L));
                    if (rsteps) std::printf(" retrieval_loss=%.4f batch_acc=%.3f", window_rloss / rsteps, window_racc / rsteps);
                    std::printf("\n"); std::fflush(stdout);
                    window_ce = window_rloss = window_racc = 0; window_count = rsteps = 0;
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
        std::vector<float> keys;  // (nodes, R) unit keys of the whole index
        if (memory_mode == "retrieved") {
            auto Wk = host_master(m, m.MS.rk);
            keys.assign(nodes.size() * R, 0.f);
            for (size_t first = 0; first < nodes.size(); first += B) {
                std::vector<std::string> batch;
                for (size_t i = first; i < std::min(nodes.size(), first + B); ++i) batch.push_back(nodes[i].label);
                std::vector<int> last;
                auto rows = read_batch(batch, last);
                for (size_t b = 0; b < batch.size(); ++b) {
                    if (last[b] < 0) continue;  // label longer than seqlen: never retrieved
                    auto u = head_unit(Wk, &rows[b * D], R, D);
                    std::copy(u.begin(), u.end(), keys.begin() + (first + b) * R);
                }
            }
        }
        auto Wq = memory_mode == "retrieved" ? host_master(m, m.MS.rq) : std::vector<float>();
        long exact = 0, evaluated = 0, skipped = 0, count = 0, top1 = 0, top1_label = 0, top5 = 0;
        double ce_sum = 0;
        std::vector<bf16> logits((size_t)N * 256);
        for (size_t first = 0; first < n; first += B) {
            std::vector<std::string> memory(B);
            std::vector<std::string> retrieved_label(B);
            if (memory_mode == "retrieved") {
                std::vector<std::string> questions;
                for (int b = 0; b < B && first + b < n; ++b) questions.push_back(examples[first + b].question);
                std::vector<int> last;
                auto rows = read_batch(questions, last);
                for (size_t b = 0; b < questions.size(); ++b) {
                    if (last[b] < 0) continue;
                    auto q = head_unit(Wq, &rows[b * D], R, D);
                    std::vector<std::pair<float, int>> best;  // top 5 by cosine
                    for (size_t k = 0; k < nodes.size(); ++k) {
                        float s = 0;
                        for (int r = 0; r < R; ++r) s += q[r] * keys[k * R + r];
                        if (best.size() < 5 || s > best.back().first) {
                            best.insert(std::upper_bound(best.begin(), best.end(), std::make_pair(s, (int)k),
                                        [](auto& a, auto& c) { return a.first > c.first; }), {s, (int)k});
                            if (best.size() > 5) best.pop_back();
                        }
                    }
                    const Example& e = examples[first + b];
                    const Node& got = nodes[best[0].second];
                    memory[b] = got.memory; retrieved_label[b] = got.label;
                    top1 += got.qid == e.subject;
                    auto it = node_of.find(e.subject);
                    top1_label += it != node_of.end() && got.label == nodes[it->second].label;
                    for (auto& [s, k] : best) top5 += nodes[k].qid == e.subject;
                }
            } else {
                for (int b = 0; b < B && first + b < n; ++b) {
                    size_t i = first + b;
                    if (memory_mode == "on") memory[b] = examples[i].memory;
                    else if (memory_mode == "shuffled")
                        for (size_t k = 1; k < n; ++k) {  // another subject's facts, with a different answer
                            const Example& o = examples[(i + n / 2 + k) % n];
                            if (o.answer != examples[i].answer && o.memory != examples[i].memory) { memory[b] = o.memory; break; }
                        }
                }
            }
            std::vector<size_t> slot(B, SIZE_MAX);
            for (int b = 0; b < B; ++b) {
                size_t i = first + b;
                bool ok = i < n && fill_window(examples[i], memory[b], b, cfg, ids, nxt, mem) >= 0;
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
                if (evaluated <= show) {
                    std::printf("Q: %s | expected: %s | predicted: %s", examples[slot[b]].question.c_str(),
                                examples[slot[b]].answer.c_str(), predicted.c_str());
                    if (memory_mode == "retrieved") std::printf(" | retrieved: %s", retrieved_label[b].c_str());
                    std::printf("\n");
                }
            }
            release_window(m);
        }
        if (!evaluated) throw std::runtime_error("no example fits into seqlen");
        std::printf("{\"mode\":\"eval\",\"memory\":\"%s\",\"examples\":%ld,\"skipped\":%ld,\"exact\":%.4f,\"answer_ce\":%.4f",
                    memory_mode.c_str(), evaluated, skipped, (double)exact / evaluated, ce_sum / std::max(count, 1L));
        if (memory_mode == "retrieved")
            std::printf(",\"index_nodes\":%zu,\"retrieval_top1\":%.4f,\"retrieval_top1_label\":%.4f,\"retrieval_top5\":%.4f",
                        nodes.size(), (double)top1 / n, (double)top1_label / n, (double)top5 / n);
        std::printf("}\n");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}

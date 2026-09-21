// sample: CHECKPOINT "prompt..." [temp=0.7] [maxlen=256] [stop=0.5] [seed=1]
// Nutzt exakt den Trainings-Forward (forward_window, B=1/T=1) Byte für Byte;
// StreamState (Carry + MLA-Cache) läuft persistent. Nur Gewichte laden.
#include "checkpoint.h"
#include <cmath>
#include <cstring>
#include <random>

// Gewichte-only Loader: volle Integritätsprüfung (verify_checkpoint) plus
// Positionsbeweis (Dateiende exakt erreicht). Optimizer/Momente und
// Stream-State werden übersprungen (fseek), Master direkt per DMA.
static void load_weights_only(const std::string& path, Model& m,
                              const Cfg& file_cfg) {
    verify_checkpoint(path);
    Checkpoint io(path, false);
    Cfg stored;
    checkpoint_header(io, stored);
    require_same_model(stored, m.c);
    Progress pr;
    io.scalar(pr.step); io.scalar(pr.cursor); io.scalar(pr.epoch);
    io.scalar(pr.carried); io.scalar(pr.data_size); io.scalar(pr.data_hash);
    uint64_t count = 0;
    io.scalar(count);
    if (count != m.params.values.size())
        throw std::runtime_error("parameter count mismatch");
    for (auto& p : m.params.values) {
        uint64_t n = 0;
        io.scalar(n);
        if (n != (uint64_t)p.n) throw std::runtime_error("parameter shape mismatch");
        io.device(p.master, n * 4);
        if (fseek(io.file, (long)(n * 4), SEEK_CUR) ||
            fseek(io.file, (long)(n * 4), SEEK_CUR))
            throw std::runtime_error("checkpoint skip failed");
        copy_bf16_kernel<<<(n + 255) / 256, 256>>>(p.master, p.work, n);
    }
    long spos = 0;
    io.scalar(spos);  // state.position (wird frisch aufgebaut, ignoriert)
    (void)spos;
    int fB = file_cfg.batch, fD = file_cfg.dim;
    for (int l = 0; l < file_cfg.layers; ++l) {
        if (fseek(io.file, (long)fB * fD * 4, SEEK_CUR))
            throw std::runtime_error("checkpoint skip failed");
        if (l < m.c.layers && m.ML[l].use) {
            long head = 0, base0 = 0;
            io.scalar(head); io.scalar(base0);
            int Lr = file_cfg.mla_L, R = file_cfg.mla_R, Cm = file_cfg.mla_cache;
            for (int b = 0; b < fB; ++b) {
                if (fseek(io.file, head * Lr * 2, SEEK_CUR) ||
                    fseek(io.file, head * R * 2, SEEK_CUR))
                    throw std::runtime_error("checkpoint skip failed");
            }
            (void)Cm;
        } else if (file_cfg.mla && l % file_cfg.mla_every == 0) {
            long head = 0, base0 = 0;
            io.scalar(head); io.scalar(base0);
            int Lr = file_cfg.mla_L, R = file_cfg.mla_R;
            for (int b = 0; b < fB; ++b) {
                if (fseek(io.file, head * Lr * 2, SEEK_CUR) ||
                    fseek(io.file, head * R * 2, SEEK_CUR))
                    throw std::runtime_error("checkpoint skip failed");
            }
        }
    }
    // Positionsbeweis: exakt 8 Checksummen-Bytes müssen übrig sein.
    long pos = ftell(io.file);
    if (fseek(io.file, 0, SEEK_END)) throw std::runtime_error("seek failed");
    long end = ftell(io.file);
    if (end - pos != 8) throw std::runtime_error("checkpoint layout mismatch");
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::printf("usage: sample CHECKPOINT \"prompt\" [temp=0.7] [maxlen=256] [stop=0.5] [seed=1]\n");
            return 1;
        }
        std::string path = argv[1], prompt = argv[2];
        float temp = 0.7f;
        int maxlen = 256;
        float stop_thr = 0.5f;
        unsigned seed = 1;
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            auto eq = a.find('=');
            if (eq == std::string::npos) throw std::runtime_error("expected key=value");
            std::string k = a.substr(0, eq), v = a.substr(eq + 1);
            if (k == "temp") temp = std::stof(v);
            else if (k == "maxlen") maxlen = std::stoi(v);
            else if (k == "stop") stop_thr = std::stof(v);
            else if (k == "seed") seed = (unsigned)std::stoul(v);
            else throw std::runtime_error("unknown option: " + k);
        }
        Cfg file_cfg = checkpoint_config(path);
        Cfg cfg = file_cfg;
        cfg.batch = 1; cfg.seqlen = 1;  // Single-Step (nur Runtime-Keys)
        validate_cfg(cfg);
        Model m;
        m.c = cfg;
        build_model(m);
        StreamState state;
        build_state(state, m);
        load_weights_only(path, m, file_cfg);
        std::mt19937 rng(seed);
        std::vector<unsigned char> out;
        auto step_byte = [&](int b) -> std::pair<int, float> {
            int ids = b, nxt = 0, end = 0;
            CUDA_CHECK(cudaMemcpy(m.ids, &ids, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(m.nxt, &nxt, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(m.end, &end, 4, cudaMemcpyHostToDevice));
            float tot = 0, ce = 0;
            forward_window(m, state, tot, ce);
            float logits[256], stop[1];
            CUDA_CHECK(cudaMemcpy(logits, m.logits, 256 * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(stop, m.stoplog, 2, cudaMemcpyDeviceToHost));
            // bf16 -> fp32 auf Host
            float lf[256];
            for (int i = 0; i < 256; ++i) {
                uint16_t bits;
                memcpy(&bits, (char*)logits + i * 2, 2);
                uint32_t u = (uint32_t)bits << 16;
                memcpy(&lf[i], &u, 4);
            }
            uint16_t sbits;
            memcpy(&sbits, stop, 2);
            uint32_t su = (uint32_t)sbits << 16;
            float sstop;
            memcpy(&sstop, &su, 4);
            float mx = lf[0];
            for (int i = 1; i < 256; ++i) mx = fmaxf(mx, lf[i]);
            if (temp <= 0) {
                int best = 0;
                for (int i = 1; i < 256; ++i)
                    if (lf[i] > lf[best]) best = i;
                return {best, 1.0f / (1.0f + expf(-sstop))};
            }
            double se = 0;
            for (int i = 0; i < 256; ++i) se += exp((lf[i] - mx) / temp);
            double r = std::uniform_real_distribution<double>(0, 1)(rng) * se, acc = 0;
            int pick = 255;
            for (int i = 0; i < 256; ++i) {
                acc += exp((lf[i] - mx) / temp);
                if (acc >= r) { pick = i; break; }
            }
            return {pick, 1.0f / (1.0f + expf(-sstop))};
        };
        for (size_t i = 0; i < prompt.size(); ++i)
            step_byte((unsigned char)prompt[i]);
        for (int i = 0; i < maxlen; ++i) {
            int prev = out.empty() ? (unsigned char)prompt.back() : out.back();
            auto [b, s] = step_byte(prev);
            if (s > stop_thr) break;
            out.push_back((unsigned char)b);
            fwrite(&out.back(), 1, 1, stdout);
            fflush(stdout);
        }
        std::printf("\n");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}

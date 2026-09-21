// sample: CHECKPOINT "prompt..." [temp=0.7] [maxlen=256] [stop=0.5] [seed=1]
// Nutzt exakt den Trainings-Forward (forward_window, B=1/T=1) Byte für Byte;
// StreamState (Carry + MLA-Cache) läuft persistent. Nur Gewichte laden.
#include "checkpoint.h"
#include <cmath>
#include <cstring>
#include <random>

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
        // Stop-Head ohne Training (stop=0) liefert Zufallslogits -> ignorieren.
        const bool use_stop = file_cfg.stop > 0;
        // Jedes Byte genau einmal einspeisen: die Vorhersage nach dem letzten
        // Prompt-Byte liefert direkt das erste Ausgabe-Byte.
        std::pair<int, float> next{0, 0.f};
        for (size_t i = 0; i < prompt.size(); ++i)
            next = step_byte((unsigned char)prompt[i]);
        for (int i = 0; i < maxlen; ++i) {
            auto [b, s] = next;
            if (use_stop && s > stop_thr) break;
            out.push_back((unsigned char)b);
            fwrite(&out.back(), 1, 1, stdout);
            fflush(stdout);
            if (i + 1 < maxlen) next = step_byte(b);
        }
        std::printf("\n");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}

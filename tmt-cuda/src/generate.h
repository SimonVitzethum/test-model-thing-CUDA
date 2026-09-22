#pragma once
// Byte-by-byte generation with a persistent recurrent state (shared by sample
// and chat). Uses exactly the training forward (forward_window with B=1, T=1);
// the StreamState (carry + MLA cache) lives as long as the generator.
#include "checkpoint.h"
#include <cmath>
#include <random>

struct Generator {
    Cfg file_cfg;
    Model m;
    StreamState state;
    std::mt19937 rng;
    float logits[256] = {};
    float stop_prob = 0;   // sigmoid of the stop head for the last fed byte
    bool ready = false;    // logits hold the prediction after the last fed byte
    long fed = 0;          // bytes fed since the last reset

    Generator(const std::string& path, unsigned seed) : file_cfg(checkpoint_config(path)), rng(seed) {
        Cfg cfg = file_cfg;
        cfg.batch = 1; cfg.seqlen = 1;  // single step (runtime keys only)
        validate_cfg(cfg);
        m.c = cfg;
        build_model(m);
        build_state(state, m);
        load_weights_only(path, m, file_cfg);
    }
    // An untrained stop head (stop=0) gives random logits and is ignored.
    bool stop_trained() const { return file_cfg.stop > 0; }

    void feed(int byte) {
        int ids = byte, nxt = -1, end = 0;
        CUDA_CHECK(cudaMemcpy(m.ids, &ids, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m.nxt, &nxt, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m.end, &end, 4, cudaMemcpyHostToDevice));
        float tot = 0, ce = 0;
        forward_window(m, state, tot, ce);
        bf16 raw[256], stop;
        CUDA_CHECK(cudaMemcpy(raw, m.logits, 256 * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&stop, m.stoplog, 2, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 256; ++i) logits[i] = bf2f(raw[i]);
        stop_prob = 1.f / (1.f + expf(-bf2f(stop)));
        ready = true; ++fed;
    }
    void feed(const std::string& text) { for (unsigned char c : text) feed((int)c); }

    // Next byte from the prediction after the last fed byte (argmax if temp <= 0).
    int sample(float temp) {
        if (!ready) throw std::runtime_error("nothing fed yet");
        if (temp <= 0) return (int)(std::max_element(logits, logits + 256) - logits);
        float mx = *std::max_element(logits, logits + 256);
        double p[256], sum = 0;
        for (int i = 0; i < 256; ++i) sum += p[i] = exp((logits[i] - mx) / temp);
        double r = std::uniform_real_distribution<double>(0, 1)(rng) * sum;
        for (int i = 0; i < 256; ++i) if ((r -= p[i]) <= 0) return i;
        return 255;
    }

    void reset() { reset_state(state, m.c); ready = false; fed = 0; }
};

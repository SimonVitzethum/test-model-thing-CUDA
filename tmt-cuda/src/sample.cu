// sample: CHECKPOINT "prompt..." [temp=0.7] [maxlen=256] [stop=0.5] [seed=1]
// Feeds the prompt byte by byte, then continues it (src/generate.h).
#include "generate.h"

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::printf("usage: sample CHECKPOINT \"prompt\" [temp=0.7] [maxlen=256] [stop=0.5] [seed=1]\n");
            return 1;
        }
        std::string path = argv[1], prompt = argv[2];
        float temp = 0.7f, stop_thr = 0.5f;
        int maxlen = 256;
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
        if (prompt.empty()) throw std::runtime_error("empty prompt");
        Generator g(path, seed);
        // Every byte is fed exactly once: the prediction after the last prompt
        // byte gives the first output byte.
        g.feed(prompt);
        for (int i = 0; i < maxlen; ++i) {
            if (g.stop_trained() && g.stop_prob > stop_thr) break;
            int b = g.sample(temp);
            std::fputc(b, stdout);
            std::fflush(stdout);
            if (i + 1 < maxlen) g.feed(b);
        }
        std::printf("\n");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}

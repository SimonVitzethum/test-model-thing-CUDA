// chat: interactive CLI with a persistent recurrent state.
//   chat CHECKPOINT [temp=0.7] [maxlen=512] [seed=1] [mode=auto|dialog|raw]
//
// The state is never truncated: everything said so far stays in the recurrent
// memory until /reset (no context window).
//
// dialog  (checkpoints trained with dialog=1, see tools/dialogprep.cpp):
//         each input is fed as 0x02 text 0x04 0x03 and the model answers until it
//         ends its turn with 0x04 (or maxlen). A conversation starts with 0x1E.
// raw     (plain text checkpoints): the input is fed as text and the model
//         continues it until a newline (or maxlen).
// Commands: /reset  /temp X  /maxlen N  /mode dialog|raw  /help  /quit
#include "generate.h"
#include <iostream>
#include <unistd.h>

constexpr int CONV = 0x1E, USER = 0x02, ASSISTANT = 0x03, END = 0x04;

static bool is_marker(int b) { return b == CONV || b == USER || b == ASSISTANT || b == END; }

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::printf("usage: chat CHECKPOINT [temp=0.7] [maxlen=512] [seed=1] [mode=auto|dialog|raw]\n");
            return 1;
        }
        float temp = 0.7f;
        int maxlen = 512;
        unsigned seed = 1;
        std::string mode = "auto";
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i]; size_t eq = a.find('=');
            if (eq == std::string::npos) throw std::runtime_error("expected key=value");
            std::string k = a.substr(0, eq), v = a.substr(eq + 1);
            if (k == "temp") temp = std::stof(v);
            else if (k == "maxlen") maxlen = std::stoi(v);
            else if (k == "seed") seed = (unsigned)std::stoul(v);
            else if (k == "mode") mode = v;
            else throw std::runtime_error("unknown option: " + k);
        }
        Generator g(argv[1], seed);
        if (mode == "auto") mode = g.file_cfg.dialog ? "dialog" : "raw";
        if (mode != "dialog" && mode != "raw") throw std::runtime_error("mode must be auto, dialog or raw");
        bool tty = isatty(0);
        std::fprintf(stderr, "tmt chat (%s mode, temp %.2f). /help for commands.\n", mode.c_str(), temp);
        auto begin = [&] { g.reset(); if (mode == "dialog") g.feed(CONV); };
        begin();
        std::string line;
        while (true) {
            if (tty) { std::printf("you> "); std::fflush(stdout); }
            if (!std::getline(std::cin, line)) break;
            if (!line.empty() && line[0] == '/') {
                std::string cmd = line.substr(0, line.find(' ')), arg = line.find(' ') == std::string::npos ? "" : line.substr(line.find(' ') + 1);
                if (cmd == "/quit" || cmd == "/exit") break;
                else if (cmd == "/reset") { begin(); std::printf("(state reset)\n"); }
                else if (cmd == "/temp" && !arg.empty()) { temp = std::stof(arg); std::printf("(temp %.2f)\n", temp); }
                else if (cmd == "/maxlen" && !arg.empty()) { maxlen = std::stoi(arg); std::printf("(maxlen %d)\n", maxlen); }
                else if (cmd == "/mode" && (arg == "dialog" || arg == "raw")) { mode = arg; begin(); std::printf("(%s mode, state reset)\n", mode.c_str()); }
                else std::printf("commands: /reset /temp X /maxlen N /mode dialog|raw /quit  (%ld bytes in state)\n", g.fed);
                continue;
            }
            if (line.empty()) continue;
            if (mode == "dialog") { g.feed(USER); g.feed(line); g.feed(END); g.feed(ASSISTANT); }
            else g.feed(line);
            std::printf("tmt> ");
            if (mode == "raw") std::printf("%s", line.c_str());
            bool ended = false;
            for (int i = 0; i < maxlen; ++i) {
                int b = g.sample(temp);
                if (mode == "dialog" && is_marker(b)) { g.feed(END); ended = true; break; }  // model ended its turn
                if (mode == "raw" && b == '\n') { g.feed(b); ended = true; break; }
                std::fputc(b, stdout); std::fflush(stdout);
                g.feed(b);
            }
            if (!ended) g.feed(mode == "dialog" ? END : '\n');  // close the turn at maxlen
            std::printf("\n");
        }
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}

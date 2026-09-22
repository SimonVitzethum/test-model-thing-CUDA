// dialogprep: build byte-level dialog training data for `train dialog=1`.
//
//   zcat oasst_ready.trees.jsonl.gz | dialogprep oasst OUT_PREFIX [lang=en] [test_frac=0.05]
//   dialogprep tsv PAIRS.tsv OUT_PREFIX [test_frac=0.05]     (user <tab> assistant per line)
//
// Output: OUT_PREFIX_train.bin and OUT_PREFIX_test.bin, a plain byte stream:
//   0x1E                      start of a conversation (use docsep=30 to reset state)
//   0x02 text 0x04            user turn
//   0x03 text 0x04            assistant turn
// With dialog=1 the loss covers only assistant text and its closing 0x04, so the
// model learns to answer and to end its turn. Control bytes inside messages are
// replaced by spaces (newlines and tabs are kept).
//
// oasst: OpenAssistant oasst1 message trees (Apache-2.0). Every root-to-leaf path
// whose messages are all in `lang` becomes one conversation. The split is by
// tree, so test conversations never share a prompt with training.
#include "json_scan.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

constexpr char CONV = 0x1E, USER = 0x02, ASSISTANT = 0x03, END = 0x04;

static std::string sanitize(const std::string& s) {
    std::string out;
    for (unsigned char c : s) out += (c < 0x20 && c != '\n' && c != '\t') ? ' ' : (char)c;
    size_t a = out.find_first_not_of(" \n\t"), b = out.find_last_not_of(" \n\t");
    return a == std::string::npos ? "" : out.substr(a, b - a + 1);
}
static uint64_t fnv(std::string_view s) {
    uint64_t h = 14695981039346656037ull;
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
    return h;
}
static bool in_test(std::string_view key, double frac) {
    return (double)(fnv(key) >> 11) / (double)(1ull << 53) < frac;
}

struct Writer {
    std::ofstream out[2];
    size_t conversations[2] = {0, 0}, bytes[2] = {0, 0};
    explicit Writer(const std::string& prefix)
        : out{std::ofstream(prefix + "_train.bin", std::ios::binary), std::ofstream(prefix + "_test.bin", std::ios::binary)} {
        if (!out[0] || !out[1]) throw std::runtime_error("cannot write " + prefix + "_*.bin");
    }
    void conversation(int split, const std::vector<std::pair<bool, std::string>>& turns) {
        std::string s(1, CONV);
        for (auto& [assistant, text] : turns) { s += assistant ? ASSISTANT : USER; s += text; s += END; }
        out[split] << s;
        ++conversations[split]; bytes[split] += s.size();
    }
    void report(const std::string& prefix) {
        const char* names[2] = {"train", "test"};
        for (int k = 0; k < 2; ++k)
            std::fprintf(stderr, "wrote %s_%s.bin: %zu conversations, %zu bytes\n", prefix.c_str(), names[k],
                         conversations[k], bytes[k]);
    }
};

// Depth-first over an oasst message tree; every leaf closes one conversation.
static void walk(const Json& j, size_t msg, const std::string& lang, std::vector<std::pair<bool, std::string>>& path,
                 Writer& w, int split) {
    size_t l = j.find(msg, "lang"), role = j.find(msg, "role"), text = j.find(msg, "text");
    if (l == std::string::npos || role == std::string::npos || text == std::string::npos) return;
    if (j.raw_string(l) != lang) return;
    std::string t = sanitize(unescape(j.raw_string(text), true));
    if (t.empty()) return;
    bool assistant = j.raw_string(role) == "assistant";
    if (path.empty() && assistant) return;  // conversations start with the user
    path.emplace_back(assistant, t);
    bool leaf = true;
    j.each(j.find(msg, "replies"), [&](size_t reply) { leaf = false; walk(j, reply, lang, path, w, split); });
    if (leaf && assistant) w.conversation(split, path);  // end on an assistant turn
    path.pop_back();
}

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> opt;
        std::vector<std::string> pos;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i]; size_t eq = a.find('=');
            if (eq == std::string::npos) pos.push_back(a); else opt[a.substr(0, eq)] = a.substr(eq + 1);
        }
        auto get = [&](const char* k, const char* d) { return opt.count(k) ? opt[k] : std::string(d); };
        double frac = std::stod(get("test_frac", "0.05"));
        if (pos.size() == 2 && pos[0] == "oasst") {
            Writer w(pos[1]);
            std::string line, lang = get("lang", "en");
            long trees = 0, errors = 0;
            while (std::getline(std::cin, line)) {
                if (line.empty()) continue;
                try {
                    Json j{line};
                    size_t id = j.find(0, "message_tree_id"), prompt = j.find(0, "prompt");
                    if (id == std::string::npos || prompt == std::string::npos) continue;
                    std::vector<std::pair<bool, std::string>> path;
                    walk(j, prompt, lang, path, w, in_test(j.raw_string(id), frac));
                    ++trees;
                } catch (const std::exception&) { ++errors; }
            }
            std::fprintf(stderr, "%ld trees, %ld parse errors\n", trees, errors);
            w.report(pos[1]);
            return 0;
        }
        if (pos.size() == 3 && pos[0] == "tsv") {
            std::ifstream in(pos[1]);
            if (!in) throw std::runtime_error("cannot read " + pos[1]);
            Writer w(pos[2]);
            std::string line;
            while (std::getline(in, line)) {
                size_t tab = line.find('\t');
                if (tab == std::string::npos) continue;
                std::string u = sanitize(line.substr(0, tab)), a = sanitize(line.substr(tab + 1));
                if (!u.empty() && !a.empty()) w.conversation(in_test(line, frac), {{false, u}, {true, a}});
            }
            w.report(pos[2]);
            return 0;
        }
        std::fprintf(stderr, "usage: dialogprep oasst OUT_PREFIX [lang=en] [test_frac=0.05]   < trees.jsonl\n"
                             "       dialogprep tsv PAIRS.tsv OUT_PREFIX [test_frac=0.05]\n");
        return 1;
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}

// kgprep: knowledge-graph preparation for kgtrain (C++17, no dependencies).
//
//   kgprep dump OUT.tsv [threads=N] [limit=N]      < Wikidata JSON dump on stdin
//   kgprep qa GRAPH.tsv OUT_PREFIX [mem_len=256] [test_frac=0.1] [seed=1]
//
// dump: one streaming pass over a Wikidata JSON dump (one entity per line, as in
//   latest-all.json.gz; decompress with e.g. `pigz -dc` or `lbzip2 -dc`). Keeps the
//   item-valued facts (PROPS) of entities with an English Wikipedia article and the
//   English labels of all items, then writes nodes and facts whose endpoints are
//   labeled. Memory: roughly 12 bytes plus the label per labeled item (a few GB for
//   the full dump).
// qa: turns the graph into question/answer/memory rows, split by SUBJECT so test
//   questions are about entities never seen in training.
//
// Graph TSV:  N <tab> QID <tab> label   |   F <tab> QID <tab> PID <tab> QID
// QA TSV:     question <tab> answer <tab> memory <tab> subject QID
// Keep PROPS in sync with tools/wikidata_kg.py (API fetch).
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

struct Prop { const char* pid; const char* name; const char* question; };
static const Prop PROPS[] = {
    {"P36", "capital", "What is the capital of %s?"},
    {"P30", "continent", "On which continent is %s?"},
    {"P38", "currency", "What is the currency of %s?"},
    {"P37", "official language", "What is the official language of %s?"},
    {"P17", "country", "In which country is %s?"},
    {"P19", "place of birth", "Where was %s born?"},
    {"P20", "place of death", "Where did %s die?"},
    {"P27", "citizenship", "What is the citizenship of %s?"},
    {"P106", "occupation", "What was the occupation of %s?"},
    {"P50", "author", "Who wrote %s?"},
    {"P136", "genre", "What is the genre of %s?"},
    {"P495", "country of origin", "Where does %s come from?"},
    {"P186", "made from", "What is %s made of?"},
    {"P57", "director", "Who directed %s?"},
    {"P1412", "language spoken", "Which language did %s speak?"},
};
constexpr int NPROPS = sizeof(PROPS) / sizeof(PROPS[0]);
constexpr size_t MAX_LABEL = 60;

static int prop_index(std::string_view pid) {
    for (int i = 0; i < NPROPS; ++i) if (pid == PROPS[i].pid) return i;
    return -1;
}

// ------------------------------------------------------------ JSON scanner
// Navigates one JSON line without building a DOM: positions are offsets into
// the line; skip() jumps over a complete value.
struct Json {
    std::string_view s;
    size_t ws(size_t p) const { while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\r' || s[p] == '\n')) ++p; return p; }
    size_t skip_string(size_t p) const {  // p at opening quote, returns after closing quote
        for (++p; p < s.size(); ++p) {
            if (s[p] == '\\') ++p;
            else if (s[p] == '"') return p + 1;
        }
        throw std::runtime_error("unterminated string");
    }
    size_t skip(size_t p) const {
        p = ws(p);
        if (p >= s.size()) throw std::runtime_error("unexpected end");
        char c = s[p];
        if (c == '"') return skip_string(p);
        if (c == '{' || c == '[') {
            int depth = 0;
            for (; p < s.size(); ++p) {
                if (s[p] == '"') { p = skip_string(p) - 1; continue; }
                if (s[p] == '{' || s[p] == '[') ++depth;
                else if (s[p] == '}' || s[p] == ']') { if (--depth == 0) return p + 1; }
            }
            throw std::runtime_error("unterminated container");
        }
        while (p < s.size() && s[p] != ',' && s[p] != '}' && s[p] != ']') ++p;  // number/literal
        return p;
    }
    std::string_view raw_string(size_t p) const {  // content between quotes (escapes kept)
        size_t e = skip_string(p);
        return s.substr(p + 1, e - p - 2);
    }
    // Value position of `key` in the object starting at p, or npos.
    size_t find(size_t p, std::string_view key) const {
        if (p == std::string::npos) return p;
        p = ws(p);
        if (p >= s.size() || s[p] != '{') return std::string::npos;
        p = ws(p + 1);
        while (p < s.size() && s[p] != '}') {
            std::string_view k = raw_string(p);
            p = ws(skip_string(p));
            if (s[p] != ':') throw std::runtime_error("expected ':'");
            size_t v = ws(p + 1);
            if (k == key) return v;
            p = ws(skip(v));
            if (s[p] == ',') p = ws(p + 1);
        }
        return std::string::npos;
    }
    template<class F> void each(size_t p, F f) const {  // array elements or object values
        if (p == std::string::npos) return;
        p = ws(p);
        char open = s[p];
        if (open != '[' && open != '{') return;
        p = ws(p + 1);
        char close = open == '[' ? ']' : '}';
        while (p < s.size() && s[p] != close) {
            if (open == '{') { p = ws(skip_string(p)); p = ws(p + 1); }
            f(p);
            p = ws(skip(p));
            if (s[p] == ',') p = ws(p + 1);
        }
    }
};

static void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) { out += (char)(0xC0 | cp >> 6); out += (char)(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { out += (char)(0xE0 | cp >> 12); out += (char)(0x80 | (cp >> 6 & 63)); out += (char)(0x80 | (cp & 63)); }
    else { out += (char)(0xF0 | cp >> 18); out += (char)(0x80 | (cp >> 12 & 63)); out += (char)(0x80 | (cp >> 6 & 63)); out += (char)(0x80 | (cp & 63)); }
}
static std::string unescape(std::string_view r) {
    std::string out;
    for (size_t i = 0; i < r.size(); ++i) {
        if (r[i] != '\\') { out += r[i]; continue; }
        char c = r[++i];
        switch (c) {
            case 'n': case 't': case 'r': case 'b': case 'f': out += ' '; break;
            case 'u': {
                uint32_t cp = std::stoul(std::string(r.substr(i + 1, 4)), nullptr, 16); i += 4;
                if (cp >= 0xD800 && cp < 0xDC00 && i + 6 < r.size() && r[i + 1] == '\\' && r[i + 2] == 'u') {
                    uint32_t lo = std::stoul(std::string(r.substr(i + 3, 4)), nullptr, 16);
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); i += 6;
                }
                append_utf8(out, cp); break;
            }
            default: out += c;
        }
    }
    return out;
}
// Collapse whitespace; reject empty, too long or tab/newline-containing labels.
static bool clean_label(std::string& label) {
    std::string out; bool space = false;
    for (char c : label) {
        bool w = c == ' ' || c == '\t' || c == '\n' || c == '\r';
        if (w) { space = !out.empty(); continue; }
        if (space) out += ' ';
        space = false; out += c;
    }
    label = out;
    return !label.empty() && label.size() <= MAX_LABEL;
}
static uint32_t qnum(std::string_view id) {
    if (id.size() < 2 || id[0] != 'Q') return 0;
    uint64_t v = 0;
    for (char c : id.substr(1)) { if (c < '0' || c > '9') return 0; v = v * 10 + (c - '0'); }
    return v <= UINT32_MAX ? (uint32_t)v : 0;
}

struct Fact { uint32_t s; uint16_t p; uint32_t o; };
struct Batch { std::vector<std::pair<uint32_t, std::string>> labels; std::vector<Fact> facts; };

static void parse_entity(std::string_view line, Batch& out) {
    Json j{line};
    size_t type = j.find(0, "type"), id = j.find(0, "id");
    if (type == std::string::npos || id == std::string::npos || j.raw_string(type) != "item") return;
    uint32_t q = qnum(j.raw_string(id));
    if (!q) return;
    size_t en = j.find(j.find(j.find(0, "labels"), "en"), "value");
    if (en == std::string::npos) return;
    std::string label = unescape(j.raw_string(en));
    if (!clean_label(label)) return;
    out.labels.emplace_back(q, label);
    if (j.find(j.find(0, "sitelinks"), "enwiki") == std::string::npos) return;
    size_t claims = j.find(0, "claims");
    for (int pi = 0; pi < NPROPS; ++pi) {
        std::vector<std::pair<bool, uint32_t>> values;  // (preferred, object)
        j.each(j.find(claims, PROPS[pi].pid), [&](size_t st) {
            size_t rank = j.find(st, "rank");
            std::string_view r = rank == std::string::npos ? "normal" : j.raw_string(rank);
            if (r == "deprecated") return;
            size_t snak = j.find(st, "mainsnak"), kind = j.find(snak, "snaktype");
            if (kind == std::string::npos || j.raw_string(kind) != "value") return;
            size_t v = j.find(j.find(j.find(snak, "datavalue"), "value"), "id");
            if (v == std::string::npos) return;
            if (uint32_t o = qnum(j.raw_string(v))) values.emplace_back(r == "preferred", o);
        });
        bool preferred = std::any_of(values.begin(), values.end(), [](auto& v) { return v.first; });
        for (auto& [pref, o] : values)
            if (!preferred || pref) out.facts.push_back({q, (uint16_t)pi, o});
    }
}

static int cmd_dump(const std::string& out_path, int threads, long limit) {
    std::vector<std::string> lines;
    std::vector<Batch> results;
    std::mutex mu; std::condition_variable cv;
    std::vector<std::vector<std::string>> queue; bool done = false;
    std::vector<std::pair<uint32_t, std::string>> labels;
    std::vector<Fact> facts;
    std::atomic<long> errors{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&] {
            while (true) {
                std::vector<std::string> work;
                {
                    std::unique_lock<std::mutex> lock(mu);
                    cv.wait(lock, [&] { return !queue.empty() || done; });
                    if (queue.empty()) return;
                    work = std::move(queue.back()); queue.pop_back();
                }
                cv.notify_all();
                Batch b;
                for (auto& l : work) {
                    try { parse_entity(l, b); } catch (const std::exception&) { ++errors; }
                }
                std::lock_guard<std::mutex> lock(mu);
                labels.insert(labels.end(), std::make_move_iterator(b.labels.begin()), std::make_move_iterator(b.labels.end()));
                facts.insert(facts.end(), b.facts.begin(), b.facts.end());
            }
        });
    std::string line; long n = 0;
    std::vector<std::string> batch;
    std::ios::sync_with_stdio(false);
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == ',' || line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.size() < 2 || line[0] != '{') continue;
        batch.push_back(std::move(line));
        if (batch.size() == 2048) {
            std::unique_lock<std::mutex> lock(mu);
            cv.wait(lock, [&] { return queue.size() < (size_t)threads * 4; });
            queue.push_back(std::move(batch)); batch.clear();
            cv.notify_all();
        }
        if (++n % 1000000 == 0) std::fprintf(stderr, "%ld entities read\n", n);
        if (limit && n >= limit) break;
    }
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!batch.empty()) queue.push_back(std::move(batch));
        done = true;
    }
    cv.notify_all();
    for (auto& t : pool) t.join();
    std::sort(labels.begin(), labels.end(), [](auto& a, auto& b) { return a.first < b.first; });
    auto label_of = [&](uint32_t q) -> const std::string* {
        auto it = std::lower_bound(labels.begin(), labels.end(), q, [](auto& a, uint32_t v) { return a.first < v; });
        return it != labels.end() && it->first == q ? &it->second : nullptr;
    };
    std::sort(facts.begin(), facts.end(), [](const Fact& a, const Fact& b) {
        return std::tie(a.s, a.p, a.o) < std::tie(b.s, b.p, b.o);
    });
    facts.erase(std::unique(facts.begin(), facts.end(), [](const Fact& a, const Fact& b) {
        return a.s == b.s && a.p == b.p && a.o == b.o; }), facts.end());
    std::vector<Fact> kept; std::set<uint32_t> used;
    for (auto& f : facts)
        if (f.s != f.o && label_of(f.s) && label_of(f.o)) { kept.push_back(f); used.insert(f.s); used.insert(f.o); }
    std::ofstream out(out_path);
    for (uint32_t q : used) out << "N\tQ" << q << '\t' << *label_of(q) << '\n';
    for (auto& f : kept) out << "F\tQ" << f.s << '\t' << PROPS[f.p].pid << "\tQ" << f.o << '\n';
    std::fprintf(stderr, "wrote %s: %zu nodes, %zu facts (%ld entities, %ld parse errors)\n",
                 out_path.c_str(), used.size(), kept.size(), n, errors.load());
    return 0;
}

// ------------------------------------------------------------------ QA
static uint64_t fnv(std::string_view s) {
    uint64_t h = 14695981039346656037ull;
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
    return h;
}

static int cmd_qa(const std::string& graph, const std::string& prefix, size_t mem_len, double test_frac, unsigned seed) {
    std::ifstream in(graph);
    if (!in) throw std::runtime_error("cannot read " + graph);
    std::unordered_map<std::string, std::string> labels;
    std::vector<std::string> order;  // subjects in file order
    std::unordered_map<std::string, std::vector<std::pair<int, std::string>>> by_subject;
    std::string line;
    while (std::getline(in, line)) {
        std::vector<std::string> f; std::stringstream ss(line); std::string part;
        while (std::getline(ss, part, '\t')) f.push_back(part);
        if (f.size() == 3 && f[0] == "N") labels[f[1]] = f[2];
        else if (f.size() == 4 && f[0] == "F" && f[1] != f[3]) {
            int p = prop_index(f[2]);
            if (p < 0) continue;
            auto& v = by_subject[f[1]];
            if (v.empty()) order.push_back(f[1]);
            v.emplace_back(p, f[3]);
        }
    }
    std::mt19937 rng(seed);
    std::vector<std::vector<std::string>> rows[2];  // train, test
    std::set<std::string> subjects[2];
    std::unordered_map<std::string, std::string> memories;  // subject -> rendered memory
    for (auto& s : order) {
        auto& sf = by_subject[s];
        if (!labels.count(s)) continue;
        std::vector<std::string> items;
        for (auto& [p, o] : sf) if (labels.count(o)) items.push_back(std::string(PROPS[p].name) + ": " + labels[o]);
        std::shuffle(items.begin(), items.end(), rng);
        std::string memory = labels[s] + ": ";
        for (auto& item : items) {  // whole facts only, within the memory budget
            if (memory.size() + item.size() + 2 > mem_len) break;
            memory += item + "; ";
        }
        memories[s] = memory;
        std::map<int, std::set<std::string>> values;
        for (auto& [p, o] : sf) if (labels.count(o)) values[p].insert(o);
        int split = (double)(fnv(s) >> 11) / (double)(1ull << 53) < test_frac;
        for (auto& [p, objs] : values) {
            if (objs.size() != 1) continue;  // multi-valued: no single correct answer
            const std::string& answer = labels[*objs.begin()];
            if (memory.find(std::string(PROPS[p].name) + ": " + answer + ";") == std::string::npos) continue;
            char question[512];
            std::snprintf(question, sizeof question, PROPS[p].question, labels[s].c_str());
            rows[split].push_back({question, answer, memory, s});
            subjects[split].insert(s);
        }
    }
    const char* names[2] = {"train", "test"};
    for (int k = 0; k < 2; ++k) {
        std::shuffle(rows[k].begin(), rows[k].end(), rng);
        std::string path = prefix + "_" + names[k] + ".tsv";
        std::ofstream out(path);
        for (auto& r : rows[k]) out << r[0] << '\t' << r[1] << '\t' << r[2] << '\t' << r[3] << '\n';
        std::fprintf(stderr, "wrote %s: %zu examples, %zu subjects\n", path.c_str(), rows[k].size(), subjects[k].size());
    }
    // Retrieval index for stage 2: every labeled node with the memory it would
    // load (subjects: the same text as in the QA rows; other nodes: label only).
    std::vector<std::string> ids;
    for (auto& [q, l] : labels) ids.push_back(q);
    std::sort(ids.begin(), ids.end(), [](const std::string& a, const std::string& b) {
        return a.size() != b.size() ? a.size() < b.size() : a < b; });
    std::ofstream nodes(prefix + "_nodes.tsv");
    for (auto& q : ids) {
        auto it = memories.find(q);
        nodes << q << '\t' << labels[q] << '\t' << (it != memories.end() ? it->second : labels[q] + ": ") << '\n';
    }
    std::fprintf(stderr, "wrote %s_nodes.tsv: %zu nodes\n", prefix.c_str(), ids.size());
    return 0;
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
        if (pos.size() == 2 && pos[0] == "dump") {
            int threads = std::stoi(get("threads", "0"));
            if (threads <= 0) threads = std::max(1u, std::thread::hardware_concurrency());
            return cmd_dump(pos[1], threads, std::stol(get("limit", "0")));
        }
        if (pos.size() == 3 && pos[0] == "qa")
            return cmd_qa(pos[1], pos[2], std::stoul(get("mem_len", "256")), std::stod(get("test_frac", "0.1")),
                          (unsigned)std::stoul(get("seed", "1")));
        std::fprintf(stderr, "usage: kgprep dump OUT.tsv [threads=N] [limit=N]   < wikidata-dump.json\n"
                             "       kgprep qa GRAPH.tsv OUT_PREFIX [mem_len=256] [test_frac=0.1] [seed=1]\n");
        return 1;
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}

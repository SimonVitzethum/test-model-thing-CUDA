#pragma once
#include "model.cu"
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <limits>
#include <unistd.h>

// V3 is intentionally incompatible with the old additive-state architecture.
// A streaming checksum detects truncation/corruption before loading GPU weights.
static uint64_t hash_bytes(uint64_t h, const void* data, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) h = (h ^ bytes[i]) * UINT64_C(1099511628211);
    return h;
}
constexpr uint64_t HASH_INIT = UINT64_C(14695981039346656037);
struct Progress {
    uint64_t step = 0, cursor = 0, epoch = 0, carried = 0;
    uint64_t data_size = 0, data_hash = 0;
};
struct Checkpoint {
    FILE* file = nullptr;
    bool writing;
    uint64_t hash = HASH_INIT;
    explicit Checkpoint(const std::string& path, bool write) : writing(write) {
        file = fopen(path.c_str(), write ? "wb" : "rb");
        if (!file) throw std::runtime_error("cannot open checkpoint: " + path);
    }
    ~Checkpoint() { if (file) fclose(file); }
    void bytes(void* ptr, size_t size) {
        size_t got = writing ? fwrite(ptr, 1, size, file) : fread(ptr, 1, size, file);
        if (got != size) throw std::runtime_error("checkpoint I/O failed or truncated");
        hash = hash_bytes(hash, ptr, size);
    }
    template<class T> void scalar(T& v) { bytes(&v, sizeof(v)); }
    void device(void* ptr, size_t size) {
        // Bounded host staging memory, including for large models/caches.
        std::vector<unsigned char> buffer(std::min(size, size_t(1 << 20)));
        for (size_t offset = 0; offset < size; offset += buffer.size()) {
            size_t n = std::min(buffer.size(), size - offset);
            auto* gpu = static_cast<unsigned char*>(ptr) + offset;
            if (writing) CUDA_CHECK(cudaMemcpy(buffer.data(), gpu, n, cudaMemcpyDeviceToHost));
            bytes(buffer.data(), n);
            if (!writing) CUDA_CHECK(cudaMemcpy(gpu, buffer.data(), n, cudaMemcpyHostToDevice));
        }
    }
    void finish() {
        uint64_t checksum = hash;
        if (writing) {
            if (fwrite(&checksum, sizeof(checksum), 1, file) != 1 || fflush(file) || fsync(fileno(file)))
                throw std::runtime_error("checkpoint flush failed");
        } else {
            uint64_t expected = 0;
            if (fread(&expected, sizeof(expected), 1, file) != 1 || expected != checksum || fgetc(file) != EOF)
                throw std::runtime_error("checkpoint checksum/trailer mismatch");
        }
    }
};
static void verify_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot read checkpoint");
    auto size = in.tellg();
    if (size < std::streamoff(16)) throw std::runtime_error("checkpoint truncated");
    in.seekg(0);
    uint64_t remaining = (uint64_t)size - 8, h = HASH_INIT, expected = 0;
    char buffer[65536];
    while (remaining) {
        size_t n = std::min(remaining, (uint64_t)sizeof(buffer));
        if (!in.read(buffer, n)) throw std::runtime_error("checkpoint read failed");
        h = hash_bytes(h, buffer, n); remaining -= n;
    }
    if (!in.read(reinterpret_cast<char*>(&expected), 8) || h != expected)
        throw std::runtime_error("checkpoint checksum mismatch (old V2 files are unsupported)");
}
static void checkpoint_header(Checkpoint& io, Cfg& cfg) {
    uint64_t magic = UINT64_C(0x33544b5043544d54); // TMTCPKT3
    uint64_t read_magic = magic; io.scalar(read_magic);
    if (read_magic != magic) throw std::runtime_error("unsupported checkpoint format/architecture");
    std::string text = config_text(cfg);
    uint64_t length = text.size(); io.scalar(length);
    if (length == 0 || length > 16384) throw std::runtime_error("invalid checkpoint configuration length");
    text.resize(length); io.bytes(text.data(), length);
    if (!io.writing) {
        Cfg decoded;
        std::istringstream lines(text); std::string line, stored_keys;
        while (std::getline(lines, line)) {
            size_t eq = line.find('=');
            if (eq == std::string::npos) throw std::runtime_error("invalid checkpoint configuration");
            set_cfg(decoded, line.substr(0, eq), line.substr(eq + 1));
            stored_keys += " " + line.substr(0, eq) + " ";
        }
        validate_cfg(decoded);
        // Keys added later with a behavior-preserving default may be absent
        // in older V3 files; every other key must round-trip exactly.
        std::string expected;
        std::istringstream canonical(config_text(decoded));
        while (std::getline(canonical, line)) {
            std::string key = line.substr(0, line.find('='));
            if (stored_keys.find(" " + key + " ") != std::string::npos) expected += line + "\n";
            else if (key != "traces") throw std::runtime_error("checkpoint configuration schema mismatch");
        }
        if (expected != text) throw std::runtime_error("checkpoint configuration schema mismatch");
        cfg = decoded;
    }
}
static Cfg checkpoint_config(const std::string& path) {
    verify_checkpoint(path);
    Checkpoint io(path, false); Cfg cfg; checkpoint_header(io, cfg); return cfg;
}
// Architektur-Keys: alles, was Gewichtsformen/State-Layout bestimmt.
// Laufzeit-Keys (batch/seqlen/lr/schedule/loss/steps/...) dürfen beim Laden
// abweichen (Sampler, Eval, Fortsetzen mit neuem Schedule).
static bool is_model_key(const std::string& k) {
    return k == "dim" || k == "layers" || k == "experts" || k == "topk" ||
           k == "gated" || k == "half_min" || k == "half_max" || k == "mla" ||
           k == "mla_heads" || k == "mla_dh" || k == "mla_L" || k == "mla_R" ||
           k == "mla_cache" || k == "mla_every" || k == "mla_cc" ||
           k == "mla_theta";
}
static void require_same_model(const Cfg& a, const Cfg& b) {
    if (config_text(a).size() == 0 || config_text(b).size() == 0)
        throw std::runtime_error("empty configuration");
    // Vergleiche nur Modell-Keys Feld für Feld.
    if (a.dim != b.dim || a.layers != b.layers || a.experts != b.experts ||
        a.topk != b.topk || a.gated != b.gated || a.half_min != b.half_min ||
        a.half_max != b.half_max || a.mla != b.mla ||
        a.mla_heads != b.mla_heads || a.mla_dh != b.mla_dh ||
        a.mla_L != b.mla_L || a.mla_R != b.mla_R ||
        a.mla_cache != b.mla_cache || a.mla_every != b.mla_every ||
        a.mla_cc != b.mla_cc || a.mla_theta != b.mla_theta)
        throw std::runtime_error("checkpoint architecture differs; use a new path for a new experiment");
}
static void checkpoint_payload(Checkpoint& io, Model& m, StreamState& state, Progress& progress) {
    Cfg stored = m.c; checkpoint_header(io, stored);
    require_same_model(stored, m.c);
    io.scalar(progress.step); io.scalar(progress.cursor); io.scalar(progress.epoch);
    io.scalar(progress.carried); io.scalar(progress.data_size); io.scalar(progress.data_hash);
    if (progress.step >= INT_MAX || progress.cursor >= progress.data_size || progress.carried > LONG_MAX)
        throw std::runtime_error("invalid checkpoint progress");
    uint64_t count = m.params.values.size(); io.scalar(count);
    if (count != m.params.values.size()) throw std::runtime_error("checkpoint parameter count mismatch");
    for (auto& p : m.params.values) {
        uint64_t n = p.n; io.scalar(n);
        if (n != (uint64_t)p.n) throw std::runtime_error("checkpoint parameter shape mismatch");
        io.device(p.master, n * 4); io.device(p.m, n * 4); io.device(p.v, n * 4);
        if (!io.writing) copy_bf16_kernel<<<(n + 255) / 256, 256>>>(p.master, p.work, n);
    }
    io.scalar(state.position);
    if (state.position < 0 || (uint64_t)state.position != progress.carried)
        throw std::runtime_error("checkpoint stream position mismatch");
    for (int l = 0; l < m.c.layers; ++l) {
        io.device(state.carry[l], (long)m.c.batch * m.c.dim * 4);
        if (!m.ML[l].use) continue;
        auto& cache = state.cache[l];
        io.scalar(cache.head); io.scalar(cache.base0);
        if (cache.head < 0 || cache.head > cache.Cmax || cache.base0 < 0 || cache.base0 + cache.head != state.position)
            throw std::runtime_error("invalid checkpoint cache position");
        // Save only initialized cache contents, with the per-stream stride.
        for (int b = 0; b < m.c.batch; ++b) {
            io.device(cache.lat + (long)b * cache.Cmax * cache.L, cache.head * cache.L * 2);
            io.device(cache.kr + (long)b * cache.Cmax * cache.R, cache.head * cache.R * 2);
        }
    }
    if (m.c.traces) {  // absent for traces=0, so older V3 files load unchanged
        long bd = (long)m.c.batch * m.c.dim * 4;
        for (int l = 0; l < m.c.layers; ++l) { io.device(state.tdec[l], bd); io.device(state.tgate[l], bd); }
        io.device(state.temb, 256 * bd);
    }
    io.finish();
}
static void save_checkpoint(const std::string& path, Model& m, StreamState& state, Progress& progress) {
    std::string tmp = path + ".tmp." + std::to_string(getpid());
    try {
        { Checkpoint io(tmp, true); checkpoint_payload(io, m, state, progress); }
        if (rename(tmp.c_str(), path.c_str())) throw std::runtime_error("checkpoint rename failed");
    } catch (...) { unlink(tmp.c_str()); throw; }
}
static void load_checkpoint(const std::string& path, Model& m, StreamState& state, Progress& progress) {
    verify_checkpoint(path); // before any model/state mutation
    Checkpoint io(path, false); checkpoint_payload(io, m, state, progress);
}

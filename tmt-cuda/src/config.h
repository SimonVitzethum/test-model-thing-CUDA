#pragma once
#include <algorithm>
#include <climits>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

// One schema drives parsing and checkpoint metadata. Runtime controls are separate.
#define TMT_CONFIG(F) \
    F(int, dim, 256) F(int, layers, 4) F(int, experts, 1) F(int, topk, 1) \
    F(int, batch, 8) F(int, seqlen, 128) F(int, gated, 1) \
    F(float, half_min, 2.f) F(float, half_max, 512.f) \
    F(float, lr, 5e-4f) F(int, warmup, 200) F(int, decaysteps, 8000) F(float, minlr, .1f) \
    F(float, aux, .01f) F(float, zloss, .001f) F(float, latent, 0.f) \
    F(float, ce, 1.f) F(float, var, 0.f) F(float, stop, 0.f) \
    F(float, stopposw, 20.f) F(float, ematau, .99f) F(float, gradclip, 1.f) \
    F(int, maxcarry, 0) F(int, seed, 1234) F(int, traces, 0) F(float, trace_decay, 1.f) F(int, docsep, -1) \
    F(int, mem, 0) F(int, mem_len, 256) F(int, mem_heads, 4) F(int, mem_dh, 32) F(int, mem_every, 2) F(int, mem_rdim, 0) \
    F(int, mla, 0) F(int, mla_heads, 4) F(int, mla_dh, 32) \
    F(int, mla_L, 32) F(int, mla_R, 16) F(int, mla_cache, 4096) \
    F(int, mla_every, 2) F(int, mla_cc, 256) F(float, mla_theta, 10000.f)
struct Cfg {
#define FIELD(type, name, value) type name = value;
    TMT_CONFIG(FIELD)
#undef FIELD
};
static void set_cfg(Cfg& c, const std::string& key, const std::string& value) {
#define FIELD(type, name, initial) if (key == #name) { \
    std::istringstream in(value); type v; \
    if (!(in >> v) || !(in >> std::ws).eof() || !std::isfinite((double)v)) \
        throw std::runtime_error("invalid value for " + key); \
    c.name = v; return; }
    TMT_CONFIG(FIELD)
#undef FIELD
    throw std::runtime_error("unknown option: " + key);
}
static std::string config_text(const Cfg& c) {
    std::ostringstream out; out << std::setprecision(9);
#define FIELD(type, name, value) out << #name << "=" << c.name << "\n";
    TMT_CONFIG(FIELD)
#undef FIELD
    return out.str();
}
static void validate_cfg(const Cfg& c) {
    auto require = [](bool ok, const char* message) {
        if (!ok) throw std::runtime_error(message);
    };
    require(c.dim > 0 && c.layers > 0 && c.layers <= 256, "invalid model dimensions");
    require(c.experts >= 1 && c.experts <= 16 && c.topk >= 1 && c.topk <= c.experts,
            "require 1 <= topk <= experts <= 16");
    require(c.batch > 0 && c.seqlen > 0 && (long)c.batch * c.seqlen <= INT_MAX / 256,
            "invalid batch/seqlen");
    require((long)c.batch * c.seqlen * c.dim <= INT_MAX && (long)c.dim * c.dim <= INT_MAX,
            "tensor exceeds supported index range");
    require(c.gated == 0 || c.gated == 1, "gated must be 0 or 1");
    require(c.half_min >= 1 && c.half_max >= c.half_min && c.half_max <= 100000,
            "require 1 <= half_min <= half_max <= 100000");
    require(c.lr > 0 && c.warmup >= 0 && c.decaysteps > 0 && c.minlr >= 0 && c.minlr <= 1,
            "invalid learning-rate schedule");
    require(c.ce > 0 && c.aux >= 0 && c.zloss >= 0 && c.latent >= 0 && c.var >= 0 && c.stop >= 0,
            "loss weights must be nonnegative, ce must be positive");
    require(c.stopposw > 0 && c.ematau >= 0 && c.ematau < 1 && c.gradclip >= 0 && c.maxcarry >= 0,
            "invalid optimizer/state configuration");
    require(c.traces == 0 || c.traces == 1, "traces must be 0 or 1");
    require(c.trace_decay > 0 && c.trace_decay <= 1, "require 0 < trace_decay <= 1");
    require(c.docsep >= -1 && c.docsep <= 255, "docsep must be -1 (off) or a byte value");
    require(c.mem == 0 || c.mem == 1, "mem must be 0 or 1");
    if (c.mem) {
        require(c.mem_len > 0 && c.mem_len <= 4096 && c.mem_heads > 0 && c.mem_dh > 0 &&
                c.mem_every >= 1 && c.mem_every <= c.layers && c.mem_rdim >= 0 && c.mem_rdim <= 1024,
                "invalid memory configuration");
        require((long)c.batch * c.mem_len * c.dim <= INT_MAX &&
                (long)c.batch * c.mem_heads * c.seqlen * c.mem_len <= INT_MAX &&
                (long)c.mem_heads * c.mem_dh * c.dim <= INT_MAX, "memory exceeds supported index range");
    }
    require(c.mla == 0 || c.mla == 1, "mla must be 0 or 1");
    if (c.mla) {
        require(c.mla_heads > 0 && c.mla_dh > 0 && c.mla_L > 0 && c.mla_R > 0 && c.mla_R % 2 == 0,
                "invalid MLA dimensions (RoPE dimension must be even)");
        require(c.mla_cache >= c.seqlen && c.mla_cc > 0 && c.mla_every > 0 && c.mla_theta > 1,
                "invalid MLA cache/chunk configuration");
        require((long)c.mla_heads * ((long)c.mla_dh + c.mla_R) <= INT_MAX / c.dim &&
                (long)c.dim * c.mla_L <= INT_MAX && (long)c.dim * c.mla_R <= INT_MAX &&
                (long)c.batch * c.seqlen * c.mla_heads <= INT_MAX / std::max(c.mla_dh, c.mla_R),
                "MLA projections exceed supported index range");
        long bh = (long)c.batch * c.mla_heads;
        require(bh <= INT_MAX / c.seqlen && bh * c.seqlen <= INT_MAX / c.mla_cc &&
                (long)c.batch * c.mla_cache <= INT_MAX / std::max(c.mla_L, c.mla_R),
                "MLA cache/workspace exceeds supported index range");
    }
}

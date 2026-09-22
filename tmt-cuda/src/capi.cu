// C API over the CUDA model for the Zig host programs (zig/tmt.zig).
// Opaque handles; C++ exceptions never cross the boundary: functions return
// 0 on success or -1 with the message available from tmt_last_error().
#include "generate.h"
#include "capi.h"
#include <memory>

static thread_local std::string last_error;

template<class F> static int guard(F f) {
    try { f(); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return -1; }
    catch (...) { last_error = "unknown error"; return -1; }
}

struct tmt_gen {
    std::unique_ptr<Generator> g;
    std::string config;
};

extern "C" {

const char* tmt_last_error(void) { return last_error.c_str(); }

tmt_gen* tmt_gen_open(const char* checkpoint) {
    tmt_gen* h = nullptr;
    if (guard([&] {
            auto p = std::make_unique<tmt_gen>();
            p->g = std::make_unique<Generator>(checkpoint, 1u);
            p->config = config_text(p->g->file_cfg);
            h = p.release();
        })) return nullptr;
    return h;
}

void tmt_gen_close(tmt_gen* h) { delete h; }

const char* tmt_gen_config(const tmt_gen* h) { return h->config.c_str(); }

int tmt_gen_feed(tmt_gen* h, int byte, float* logits, float* stop_prob) {
    return guard([&] {
        h->g->feed(byte);
        std::memcpy(logits, h->g->logits, 256 * sizeof(float));
        if (stop_prob) *stop_prob = h->g->stop_prob;
    });
}

int tmt_gen_reset(tmt_gen* h) { return guard([&] { h->g->reset(); }); }

long tmt_gen_fed(const tmt_gen* h) { return h->g->fed; }

}  // extern "C"

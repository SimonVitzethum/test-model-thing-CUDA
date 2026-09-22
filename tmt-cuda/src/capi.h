/* C API of the CUDA model, used by the Zig host programs. */
#ifndef TMT_CAPI_H
#define TMT_CAPI_H
#ifdef __cplusplus
extern "C" {
#endif

/* Message of the last failed call on this thread. */
const char* tmt_last_error(void);

/* Byte-by-byte generator with a persistent recurrent state (B=1, T=1),
   weights loaded from a checkpoint. */
typedef struct tmt_gen tmt_gen;
tmt_gen* tmt_gen_open(const char* checkpoint);          /* NULL on error */
void tmt_gen_close(tmt_gen* h);
const char* tmt_gen_config(const tmt_gen* h);          /* key=value lines */
int tmt_gen_feed(tmt_gen* h, int byte, float* logits /* [256] */, float* stop_prob);
int tmt_gen_reset(tmt_gen* h);
long tmt_gen_fed(const tmt_gen* h);

/* ---- training ---- */
#include <stdint.h>

/* Model configuration (schema, parsing and validation of src/config.h). */
typedef struct tmt_cfg tmt_cfg;
tmt_cfg* tmt_cfg_new(void);                          /* defaults */
tmt_cfg* tmt_cfg_from_checkpoint(const char* path);  /* NULL on error */
tmt_cfg* tmt_cfg_copy(const tmt_cfg* c);
void tmt_cfg_free(tmt_cfg* c);
int tmt_cfg_set(tmt_cfg* c, const char* key, const char* value);
int tmt_cfg_validate(const tmt_cfg* c);
const char* tmt_cfg_text(tmt_cfg* c);                /* key=value lines, valid until the next call */

/* Training progress stored in checkpoints. */
typedef struct {
    uint64_t step, cursor, epoch, carried, data_size, data_hash;
} tmt_progress;

/* A model with its streaming state (one window of batch x seqlen bytes). */
typedef struct tmt_model tmt_model;
tmt_model* tmt_model_new(const tmt_cfg* c);          /* NULL on error */
void tmt_model_free(tmt_model* m);
int tmt_model_load(tmt_model* m, const char* path, tmt_progress* p);   /* full resume */
int tmt_model_load_weights(tmt_model* m, const char* path);            /* init= */
int tmt_model_save(tmt_model* m, const char* path, const tmt_progress* p);
int tmt_model_reset_state(tmt_model* m);
/* ids/targets/ends: batch*seqlen each; targets < 0 are not scored. */
int tmt_model_forward(tmt_model* m, const int* ids, const int* targets, const int* ends,
                      float* loss, float* ce);
int tmt_model_losses(tmt_model* m, float* out);      /* per-position CE of the last forward */
int tmt_model_backward(tmt_model* m, int log_traces);
int tmt_model_optimizer_step(tmt_model* m, int step);
int tmt_model_expert_counts(const tmt_model* m, int64_t* out /* layers*experts */);
/* Trace part vs in-window part of the gradient after a backward with
   log_traces=1: {pp, ww, pw} for decay, gate, embedding (9 values). */
int tmt_model_trace_stats(tmt_model* m, double* out);
/* Mean |state| per half-life bucket (<16, <128, <1k, <8k, >=8k). */
int tmt_model_state_buckets(tmt_model* m, double* sum /* 5 */, int64_t* count /* 5 */);
/* Parameters, for the gradient comparison: `group` is one of decay, gate,
   embedding, norm, router, experts, decoder, mla or "" (ungrouped). */
int tmt_model_param_count(const tmt_model* m);
long tmt_model_param_size(const tmt_model* m, int j);
const char* tmt_model_param_group(const tmt_model* m, int j);
int tmt_model_param_grad(tmt_model* m, int j, float* out);
int tmt_model_param_master(const tmt_model* m, int j, float* out);
int tmt_model_param_set_grad(tmt_model* m, int j, const float* in);
/* Gradient accumulator over several passes of one step (kgtrain). */
int tmt_model_acc_zero(tmt_model* m);
int tmt_model_acc_add(tmt_model* m, int j);   /* j < 0: every parameter */
int tmt_model_acc_store(tmt_model* m);        /* gradients := accumulator */
/* Index of the stage-2 retrieval heads (which: 0 = query, 1 = key), -1 if absent. */
int tmt_model_retrieval_param(const tmt_model* m, int which);
/* Forward with memory bytes (batch*mem_len, -1 = padding) and a state reset. */
int tmt_model_forward_mem(tmt_model* m, const int* ids, const int* targets, const int* mem,
                          float* loss, float* ce);
/* Backward with an external gradient on the representation (batch*seqlen*dim), or NULL. */
int tmt_model_backward_ext(tmt_model* m, const float* dX);
/* Representation rows at (b, last[b]) of the last forward; last[b] < 0 leaves the row as is. */
int tmt_model_read_rows(tmt_model* m, const int* last, float* out /* batch*dim */);
/* Logits of the last forward as float (batch*seqlen*256). */
int tmt_model_logits(tmt_model* m, float* out);
/* ---- device memory and reference kernels, for the Zig kernel tests ---- */
int tmt_dev_alloc(void** p, unsigned long n);
int tmt_dev_free(void* p);
int tmt_dev_upload(void* dst, const void* src, unsigned long n);
int tmt_dev_download(void* dst, const void* src, unsigned long n);
int tmt_ref_emb_forward(const void* W, const int* ids, void* out, int N, int D);
int tmt_ref_emb_backward(const float* dOut, const int* ids, float* dW, int N, int D);
int tmt_ref_ce_fwd(const void* logits, const int* tgt, float* probs, float* loss, int N);
int tmt_ref_ce_bwd(const float* probs, const int* tgt, void* dLogits, float w, int N);
int tmt_ref_stop_fwd(const void* s, const int* end, float* loss, float pos_w, int N);
int tmt_ref_stop_bwd(const void* s, const int* end, void* ds, float pos_w, float w, int N);
int tmt_ref_cast_add(const void* s, float* d, long n);
int tmt_ref_add_f32(float* acc, const float* x, long n);
int tmt_ref_ln_fwd(const void* X, const float* gamma, const float* beta, void* Y,
                   float* mean, float* rstd, int N, int D);
int tmt_ref_ln_bwd(const void* X, const void* dY, const float* gamma, const float* mean,
                   const float* rstd, void* dX, float* dGamma, float* dBeta, int N, int D);
int tmt_ref_copy_bf16(const float* src, void* dst, long n);
int tmt_synchronize(void);

/* SIGINT/SIGTERM set a flag instead of terminating. */
void tmt_install_stop_handler(void);
int tmt_stop_requested(void);

#ifdef __cplusplus
}
#endif
#endif

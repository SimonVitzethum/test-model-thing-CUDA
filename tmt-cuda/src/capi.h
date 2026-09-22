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

#ifdef __cplusplus
}
#endif
#endif

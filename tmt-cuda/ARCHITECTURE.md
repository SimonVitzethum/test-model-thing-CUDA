# TMT-CUDA Architecture (as of commits through `730f6bb`)

Pure CUDA C++ (no PyTorch, no Python at runtime). Byte model:
embedding (256 → dim), per layer recurrence + MoE (+ optional MLA),
byte decoder (256) + stop head (1). Training per window-TBPTT with
AdamW; checkpoints are versioned and exactly resumable.

## 1. Precision and GEMM rules

- **Compute bf16, master fp32.** Every learnable quantity exists as fp32
  master plus bf16 working copy; gradients fp32 (dX paths bf16); Adam
  moments fp32; EMA target encoder fp32.
- **cuBLAS rule (verified element by element):** A and B operand of a GEMM
  must share **dtype** (mixed bf16×fp32 → `NOT_SUPPORTED`).
  Allowed: bf16×bf16→bf16, bf16×bf16→fp32, fp32×fp32→fp32 — all with
  `COMPUTE_32F`. Hence probs/dS live in bf16 on the MLA path, only
  accumulators (O, m, l, dQ, LSE) in fp32.
- **Weight layout like `nn.Linear`:** W(N,K) row-major = [out,in].
  Row-major `Y(M,N) = X(M,K) @ W(N,K)^T` runs as
  `(OP_T, OP_N, N, M, K, W, K, X, K, Y, N)` (llm.c pattern). All
  forward/backward variants (`linear_fwd/dX/dW`, all MLA batched GEMMs,
  router GEMMs) are recomputed in this form element by element; the
  history (`pv` revert, `dq`/`dk` fix, router `ldb`) documents the
  places where alias thinking went wrong.

## 2. Parameterization

- `ParameterStore` (train.cu): all parameters in **one vector**, referenced
  **by index, never by pointer** (vector reallocation invalidates references).
  Each entry: master/m/v/grad (fp32) + work (bf16) + length.
- Init (seed default 1234, host RNG): embed uniform ±0.05,
  linears Kaiming-uniform, router normal(0, 0.02), gamma 1, beta 0,
  decay from half-life schedule (see 3), gate 0.

## 3. Hierarchical gated recurrence (cell.cu)

Per dimension and timestep, with explicit incoming carry:

```
a = sigmoid(decay + gate * x)      # learnable gate per dim (gated=1)
state = a * state + (1 - a) * x    # convex mix -> bounded
```

- `gated=0` switches the gate off (`a = sigmoid(decay)`), as before.
- **Timescale hierarchy:** `decay` is not initialized to 0/2.0 but from
  geometrically staggered half-lives `half_min=2 … half_max=512` across
  dimensions: `a0 = exp(-ln2/half)`, logit init `log(a0/(1-a0))`. Early dims
  forget after ~2 bytes, late ones after ~512 — the net *starts* with sorted
  short-/long-term memory instead of having to learn it first.
- Forward persistent: one thread per (batch, dim) loops the whole sequence
  in registers (S1 bench: ~1000 GB/s, 3 launches per window).
- **Backward exact within the window:** one thread runs backwards,
  `dS_total[t] = dSnorm[t] + a·dS_total[t+1]`, plus exact
  decay/gate gradients incl. contribution of the **incoming carry** (TBPTT
  boundary is constant, but its influence on position 0 is differentiated).
  Verified by finite differences against CPU reference
  (`architecture_test`, incl. nonzero carry, gated on/off).

## 4. MoE (moe.cu)

- Top-k routing (default E=1: dense SiLU path without router).
- Router logits fp32 straight from GEMM; softmax + top-k + renorm per row.
- **Token dispatch:** histogram + prefix offsets on host (E ≤ 16),
  permutation via atomics, exactly one GEMM per expert **over only its
  tokens** (gather/scatter are fused elementwise kernels, no atomics in
  combine: one thread per row sums its k slots).
- SiLU is applied *in* combine on the pre-activations
  (no extra memory); backward recomputes `silu`/`silu'` from them.
- **Differentiable regularizers:** switch aux
  `E·Σ(mean_p·frac)` and z-loss `mean(logsumexp²)` stand not only as
  loss scalars but with **analytic router gradients**
  (`router_bwd_kernel`: softmax backward pass plus aux term
  `aux·E/N·p·(frac−expected)` and z term `zcoef·2/N·lse·p`).
- Verified: forward + all grads (dX, dRouter, dExperts) against
  torch autograd; aux/z gradients by finite differences.

## 5. MLA-128k (mla.cu, default: off)

- DeepSeek principle without absorption trick (exact, verifiable): per position
  only **compressed latent (rank L) + decoupled RoPE key (dim R)**
  in the ring buffer. Example d=1280, L=128, R=64: 128k·192·2 B ≈ 50 MB per
  stream and layer.
- Per layer and heads: Q from `Wq`, up-projections `Wuk/Wuv` per chunk,
  RoPE (NeoX pairs, theta default 10000) on queries (with window position)
  and keys (baked at write time).
- **Chunked online softmax** (chunk default 256/1024/2048): scores and
  P·V as stacked batched GEMMs over all heads×streams (one launch per
  chunk and operation), `(m, l)` accumulators fp32, causal mask over
  global positions, LSE per query for the backward pass.
- **Backward per chunk with recompute** (no P cache across chunks):
  S recompute → dP → softmax backward (P is written into the dead S buffer)
  → dQ accumulation, dKc/dVc, up-grads via `dlat_gemm` with beta accumulation,
  rope backward (negated angles), masked scatter only for window
  positions (past gets no gradient — TBPTT semantics).
- **Reset-on-full:** if the window no longer fits `mla_cache`, the ring
  is reset (linear slot↔position mapping stays trivial);
  the reset position is stored in the checkpoint.
- Verified: forward + **all** grads against independent FP64 CPU reference,
  incl. partial chunks, multi-chunk grads, RoPE positions, prefix eviction.

## 6. Streaming state and shared forward path

- `StreamState` (model.cu) is **separate** from the model: carry vector per
  layer, MLA ring caches per layer, position counter. `build_state` +
  `reset_state` manage allocation/zeroing.
- **Train and eval use the same forward path** (`forward_window`):
  per layer save input snapshot + carry-in snapshot, state loop,
  norm, MoE, optional MLA, residual, extract carry-out.
  `backward_window` requires forward immediately before (workspace sharing).
- Window TBPTT: carry is detached only at window boundaries; `maxcarry`
  (default 0 = off) additionally limits drift; new epoch/file resets.

## 7. Losses and optimizer

- `loss = var·hinge + latent·MSE + ce·CE + stop·BCE + (aux+z)/layers`.
  Defaults: **CE-only** (`latent=var=stop=0`), CE mandatory (`ce>0` validated).
- Stop as raw logits with `BCEWithLogits + pos_weight` (default 20, EOS rare);
  latent MSE against **EMA target encoder** (no self-chasing); variance hinge
  against collapse; all weights individually switchable.
- AdamW (β 0.9/0.999, eps 1e-8, wd 0.01), global grad clip via
  `Snrm2` sum, warmup+cosine schedule, **NaN guard** (non-finite
  gradient → update refused instead of poisoning weights), EMA update
  incl. bf16 refresh of the working copy. Stop head without weight is
  skipped during update.

## 8. Checkpoints V3 (checkpoint.h, incompatible with V2)

Format `TMTCPKT3`: magic + **config as text** (schema roundtrip-checked,
mismatch → abort instead of silently wrong resume) + progress
(`step/cursor/epoch/carried`, data hash/size) + per parameter
(master/m/v) + stream position + carries + MLA cache heads and
initialized contents + **FNV checksum**, written via tmp+fsync+rename.
Loading verifies the checksum first (truncation/corruption → abort),
restores weights **and** stream history. CLI tests prove:
split-vs-whole resume is identical within FP32 tolerance, eval does not
modify the file (`cmp`), wrong config/data/truncation is rejected.

## 9. CLI and data

- `train DATA CKPT [mode=train|eval] [steps=N] [saveevery=N] [key=value …]`;
  an existing checkpoint supplies its config (CLI overrides must
  match). Data via `mmap`, sharded into B streams, windows of length T.
- `mode=eval` writes JSON per window (`bytes/ce/bpb`) without learning;
  SIGINT flag for clean abort.

## 10. Verification (passed locally, RTX 5070)

- `architecture_test`: finite differences for gated/ungated cell (with carry),
  MoE aux/z gradients, learning smoke (CE 5.42→0.75 dense, →0.82 MoE),
  stream separation, resume, corruption rejection, MLA partial chunks/
  multi-chunk grads/RoPE/eviction, MLA-vs-FP64 reference.
- `tests/cli.sh`: resume determinism (dense + MLA), eval tails,
  unmodified checkpoints, rejection of invalid inputs.
- `compute-sanitizer`: 0 memory errors (as of the commit message).

## 11. File overview (tmt-cuda/src)

| File | Contents |
|---|---|
| `config.h` | Single config schema (macro table), parsing, validation |
| `common.h`/`util.h` | bf16 helpers, `CUDA_CHECK` (no shadowing trap), cuBLAS handle |
| `linalg.cu` | bf16 GEMM wrappers fwd/dX/dW (dW with beta accumulation) |
| `cell.cu` | Gated hierarchical cell fwd/bwd + S1 demo path |
| `norm.cu` | LayerNorm with gamma/beta fwd/bwd |
| `moe.cu` | Dispatch, top-k, combine±SiLU, analytic aux/z grads, E=1 path |
| `mla.cu` | Latent ring, chunk attention fwd/bwd, RoPE, position masks |
| `emb.cu`/`loss.cu` | Gather/atomic scatter, CE/stop/latent/var kernels |
| `adam.cu` | Fused AdamW per parameter |
| `train.cu` | `ParameterStore`, `DeviceMemory`, old harness remnants |
| `model.cu` | Model/state construction, `forward_window`/`backward_window`, optimizer |
| `checkpoint.h` | V3 format, checksums, resume validation |
| `train_main.cu` | CLI, mmap data, main loop, SIGINT |
| `architecture_test.cu` | Native regression tests (see above) |
| `bench.cu` | S1 bandwidth demo (~1000 GB/s) |

## 12. Open before the big run

1. **Prod-config memory math** (420M params/128k cache/16 GB VRAM):
   cache alone ≈ 12 GB at 32 layers × 16 streams — doesn't fit together
   with Adam (~5 GB) + activations. Options: fewer streams,
   64k cache, MLA only in every n-th layer (`mla_every`).
2. **Sampler:** eval only measures CE/BPB; free generation (prompt → bytes)
   is missing for quality control in long runs.
3. **FP8** only after measurement (matmul-limited?); **compile/CUDA graphs**
   as throughput levers later.
4. **Data:** enwiki dump (download running on Fisch) unpack/split;
   harness currently reads ≤ 8 GB via `fread` (streaming for 90 GB XML missing).

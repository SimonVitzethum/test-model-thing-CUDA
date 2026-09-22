# What changed relative to the original TMT

Starting point ("TMT classic"): MLX proof-of-concept (~300 lines of Python),
4.5M parameters, byte-level, one decay value per dim (half-life ~1 byte),
JEPA latent loss + CE on the same vector, 1-step online gradient with a
hand-built trace correction covering only embedding/decay, byte-by-byte
training with `stdout` sampling, memory stored in checkpoints, Apple Silicon
only.

Today: pure CUDA-C++ project (`tmt-cuda/`), 420M-parameter runs,
BPC 1.31 (run2) and 2.50 (8h laptop run), reproducibly checkpointed.

## 1. Architecture

| Area | Original | Today |
|---|---|---|
| Recurrence | `state = decay·state + enc`, unbounded growth, single timescale | **Gated hierarchical:** `a = sigmoid(decay + gate·x)`, `state = a·state + (1−a)·x` (bounded); **half-lives geometrically staggered** (2…512 steps across dims), gate init 0 |
| Time credit | 1 step + incomplete traces (main weights got no time credit) | **Exact TBPTT window** (default 128) + exact state backward incl. incoming-carry contribution at the window edge |
| Feedforward | 1× dense linear per layer | **MoE top-k with token dispatch** (only selected experts compute), SiLU inside combine, E=1 dense path |
| Router loss | none (collapse in run2: 114/256 experts dead) | **Switch aux + z-loss with analytic router gradients**, coefficient via CLI (`--aux-coef`; fix verified at 0.1: dead 0/128 over 8 h) |
| Long range | effectively dozens of bytes | **MLA-128k** à la DeepSeek (latent ring cache ~50 MB/stream/layer, chunked online softmax, RoPE, reset-on-full); recurrence stays unbounded (O(1)) |
| Stop head | MSE on line endings, arbitrary threshold | BCEWithLogits + pos_weight (default 20), EOS = `\n` |
| Latent target | own encoder (bootstrapping against itself) | **EMA target encoder**, weights individually switchable (default CE-only) |

## 2. Training and stability

- **Batches instead of byte-by-byte:** (B,T) windows, one backward per batch (before: one sync per byte).
- **State hygiene:** state resets per epoch + optional `maxcarry` cap; checkpoints store the training stream state (recurrent carry + MLA cache) only for exact resume, while eval and sampling always start with a fresh state (before: memory carried into every later run, drift, not reproducible).
- **Optimization:** AdamW + warmup/cosine schedule + grad clip + **NaN guard** (non-finite gradient → update refused instead of poisoning weights); generation never trains weights (before: aimless TTT while sampling).
- **Init:** decay for long memory, router N(0,0.02) for uniform start.
- **Eval discipline:** held-out split (5 MB), `mode=eval` with CE/BPB JSON, CoLA anecdotes replaced by measurement.

## 3. Infrastructure (no more Python/MLX)

- **Pure CUDA C++:** all matmuls cuBLAS (tensor cores), fused elementwise kernels, persistent arenas (no malloc per window), measured near-peak (single GEMM ~100 %, MoE op ~50 % of BF16 roof).
- **cuBLAS rule (learned the hard way):** A and B of a GEMM must share dtype (mixed bf16×fp32 → `NOT_SUPPORTED`). All GEMM layouts verified element by element; probs/dS live in bf16, accumulators only in fp32.
- **StreamState decoupled:** carry/caches/position separate from the model; **same forward path** for train, eval, and sampling.
- **Checkpoints V3** (`TMTCPKT3`, incompatible with V2): config as text, progress (step/cursor/epoch/carried + data hash), master+m+v, stream history, FNV checksum, tmp+fsync+rename. Resume determinism tested; model/runtime key split (sampler may change B/T).
- **CLI:** `train DATA CKPT [mode=train|eval] [steps=N] [saveevery=N] [key=value …]`; SIGINT-safe; JSON stats; router-balance logging (collapse early warning); strict config validation.
- **Sampler** (`sample`): single-step inference B=1 through the training forward path, temperature/greedy, stop head, seed.
- **Tests without Python:** finite differences (cell/MoE-aux/MLA), learning smokes, resume/corruption/CLI tests, `compute-sanitizer` clean.

## 4. Measurements (selection)

- S1 cell: ~1000 GB/s (bandwidth roof), 3 instead of ~2560 launches.
- MoE op: 9 % → **~50 % of BF16 peak performance** (arena, 128-padding, fusion, router occupancy).
- run2 (420M total / 106M active, 5080, enwik8): **BPC 5.11 → 1.31** in 15.6M tokens, then router collapse (trigger for aux fix).
- 8h laptop run (34M, aux 0.1): 2.46B tokens, **held-out BPC 3.15 → 2.50**, router balanced throughout. *Trained with the MoE backward bug below: experts never learned.*
- **MoE backward bug (introduced in `f052b67`, fixed):** the expert-output gradient was zeroed after it was computed, and the expert-input recompute launched 1/D of the needed threads. Expert weights got zero gradient and no gradient flowed through the experts (only residual and router paths). Affects all MoE runs after 2026-09-20 22:35, including run3local and the 8h run; run2 predates it. A new test compares a two-expert MoE with identical experts against the dense path; the small MoE smoke test now reaches CE 0.82 instead of 3.36.
- CUDA vs. torch port: **216× throughput** (106k vs. 0.49k tok/s same scale).

- **Training loop overheads (2026-09-22):** profiling showed the optimizer, not the recurrence or the GEMMs, as the largest single cost (35%): one gradient-norm host sync and several launches per parameter block (~210 per step). A multi-tensor optimizer (global norm, clipping and AdamW in one launch each, one host sync per step), one-launch gradient zeroing, skipping disabled loss terms, and reading all MoE router statistics once per window raised throughput from 82k to 98k bytes/s (34M MoE, batch 16), 104k to 118k (batch 32), and 190k to 242k (dense, batch 16). The recurrence including traces is ~9% of the step. Grouped cuBLAS GEMMs for the experts were tried and gave no measurable gain (and bf16 inputs with fp32 outputs are not supported there).

## 5. Discarded/removed

MLX original (`main.py`), PyTorch port + MoE (`main_torch_moe.py`), both benchmark scripts, old READMEs, V2 checkpoints, dummy-gradient hack, RTRL approximation, memory carried from training into chat/inference, per-byte-stdout training.

## 6. Open

Prod memory math (12 GB cache vs. 16 GB VRAM), sampler extensions, FP8 only after measurement, CUTLASS grouped GEMM + CUDA graphs for the last ~20 points, 128k proof in a long run, Wikipedia-10B, Fisch deploy (tunnel flaky: `ki-pc` intermittently silent).

## 7. Apple Silicon: Zig + MLX port (`tmt-mlx/`)

The CUDA project was ported to a second backend that runs on Apple Silicon and
still needs no Python: **Zig 0.16 against the MLX C API**, with Metal kernels
for the recurrence. Architecture, CLI, defaults, diagnostics, losses,
optimizer, checkpoint format and the whole test suite are the same.

- **Shared forward path, MLX autodiff.** The forward pass is one pure function
  of arrays; training differentiates it with `mlx_value_and_grad`, evaluation
  and generation call it directly. The hand-written CUDA backward passes for
  MoE, MLA, LayerNorm and the losses become graph code; what CUDA derived by
  hand (router aux/z gradients, MLA chunk backward with recompute) is now
  autodiff, checked by the ported finite-difference and FP64-reference tests.
- **The recurrence stays hand-written.** A Metal kernel walks the window per
  (stream, channel) as in CUDA; its exact backward also produces
  λ = dL/d(carry), which the hybrid traces need, and is attached with
  `mlx_custom_vjp`. Trace advance and the embedding trace are two more kernels.
- **Determinism over atomics.** The embedding gradient is a one-hot matmul and
  MoE dispatch sorts its rows, so both backward paths are segment-based instead
  of atomic scatter-adds; otherwise two identical runs would disagree and the
  resume tests could not hold.
- **No host synchronization for MoE dispatch.** Counts and offsets stay on the
  device (`gather_mm`), where the CUDA build read the histogram back per layer.
- **Checkpoints are shared.** V3 is written byte for byte as before. Verified
  on this machine against the CUDA sources: the configuration header
  (including `%.9g` formatting) is identical, and weight initialization from a
  given seed is bit-identical (same LCG, order, and fused multiply-add). A
  CUDA-written file has not been loaded end to end here (no NVIDIA GPU).
- **Numerics.** Working precision is bf16 with fp32 masters as before;
  attention scores are fp32 (MLX has no cuBLAS dtype rule), while weight
  gradients come out of the matmul VJP in working precision instead of fp32.
  `TMT_COMPUTE=f32` runs the whole forward in fp32.
- **Measured (Apple M2, 10-core GPU):** 107k bytes/s for `dim=256 layers=4`,
  37k for `dim=512 layers=8` (50k at batch 32 × 256), 11k with 8 experts,
  7k with MLA and a 4096-byte cache. `bench` reports ~77 GB/s and ~2.2 TFLOPS
  bf16 on the same machine, which is where these numbers come from.
- **Not ported:** the CUDA-specific `profile` and `roofline` tools (their role
  is covered by `bench`) and `TMT_CUDA_WAIT`.

### Apple-Silicon tuning (same session)

Measured, then fixed in this order; throughput at `dim=512 layers=8`, batch
8 × 128 went from 37k to 62k bytes/s, with 8 experts from 11k to 21k, and the
default `dim=256 layers=4` from 107k to 173k.

- **Compiled window graph.** `mx.compile` over the gradient function (not just
  the forward, which leaves the backward unfused) is worth ~20% of the step.
  Compiled plans freeze whatever they capture, so the MLA cache, the traces and
  the RoPE tables became explicit inputs of the window function.
- **Compiled AdamW** over all parameters at once: 6.4 → 2.5 ms dense,
  34 → 10 ms with 8 experts. The CUDA build had hit the same wall (the
  optimizer was 35% of its step) and solved it with a multi-tensor kernel.
- **Recurrence scan split over time.** The kernel measured 10 GB/s at 4k chains
  but 75 GB/s at 256k: short windows starve this GPU. Splitting the scan into
  time chunks that are coupled through threadgroup memory made the kernel
  1.5–1.9× faster in isolation; it is enabled by the window shape.
- **FP32 working copies instead of BF16.** Apple GPUs have no BF16 matmul
  units (2.10 TFLOPS BF16 vs 2.23 FP32 vs 2.57 FP16 on an M2), so the CUDA
  working precision is the slow one here. FP32 is now the default and is also
  the more accurate; `TMT_COMPUTE=bf16` keeps CUDA numerics and half the
  activation memory.

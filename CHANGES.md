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
- 8h laptop run (34M, aux 0.1): 2.46B tokens, **held-out BPC 3.15 → 2.50**, router balanced throughout.
- CUDA vs. torch port: **216× throughput** (106k vs. 0.49k tok/s same scale).

## 5. Discarded/removed

MLX original (`main.py`), PyTorch port + MoE (`main_torch_moe.py`), both benchmark scripts, old READMEs, V2 checkpoints, dummy-gradient hack, RTRL approximation, memory carried from training into chat/inference, per-byte-stdout training.

## 6. Open

Prod memory math (12 GB cache vs. 16 GB VRAM), sampler extensions, FP8 only after measurement, CUTLASS grouped GEMM + CUDA graphs for the last ~20 points, 128k proof in a long run, Wikipedia-10B, Fisch deploy (tunnel flaky: `ki-pc` intermittently silent).

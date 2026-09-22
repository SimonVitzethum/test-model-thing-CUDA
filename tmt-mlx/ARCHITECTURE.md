# TMT-MLX Architecture

Zig + MLX (Metal), no Python. Same byte model as `tmt-cuda`: embedding
(256 → dim), per layer recurrence + Dense/MoE (+ optional MLA), byte decoder
(256) + stop head (1); window-TBPTT with AdamW; versioned, exactly resumable
checkpoints. This document records what the port keeps, what it implements
differently, and why.

## 1. Precision

- Every learnable quantity is an fp32 master. Recurrent state, LayerNorm
  statistics, attention accumulators, optimizer moments and the EMA target are
  fp32. The MLA cache is bf16 (checkpoint format).
- **The working precision of the forward pass is fp32, not bf16 as in CUDA.**
  Apple GPUs have no bf16 matmul units: measured on an M2, bf16 matmul runs at
  2.10 TFLOPS against 2.23 for fp32 and 2.57 for fp16, and the casts cost extra
  passes. End to end fp32 came out ~17% ahead of bf16, and it is the more
  accurate of the two. `TMT_COMPUTE=bf16|f16|f32` picks the format; all three
  pass the full test suite, and the checkpoint format is unaffected.
- MLX has no cuBLAS dtype rule to work around: `matmul` accumulates in fp32
  whatever the operand format is. Attention scores are computed in fp32, where
  the CUDA build had to keep probabilities in bf16 to satisfy cuBLAS.
- Weight gradients are produced by the matmul VJP in the working precision and
  accumulated in fp32; the CUDA build wrote dW directly as fp32. With the fp32
  default the two now agree; under `TMT_COMPUTE=bf16` the gradients are rounded
  once more than in CUDA.

## 2. Parameterization

`model.zig` owns a flat parameter list, referenced by index. Each entry has
master/m/v (fp32) and a shape. The order is the CUDA order — embedding, EMA
target, decoder, stop head, then per layer decay/gate/gamma/beta/router/experts,
then the MLA parameters of the layers that use them — because checkpoints are
shared.

Initialization uses the same LCG, the same consumption order and the same
formulas as the CUDA host initializer, down to the fused multiply-add: for a
given seed both builds produce bit-identical weights (verified).

## 3. Hierarchical gated recurrence (cell.zig)

```
a = sigmoid(decay + gate * x)      # a = 0 at a docsep byte
state = a * state + (1 - a) * x
```

- One Metal kernel per direction. The CUDA layout (one thread per (stream,
  channel), walking the window in registers) leaves this GPU idle: a window has
  only B·D chains, and the kernel measures 10 GB/s at 4k threads against
  75 GB/s at 256k. Because `a_t` depends only on `x_t`, the scan is linear with
  known coefficients and is **split over time**: every thread scans one chunk
  from a zero state and keeps (Π a, local end state), one thread per channel
  turns those into the true chunk-entry states, and every thread re-scans its
  chunk from there. All chunks of a channel sit in one threadgroup, so the
  middle step is a pass over threadgroup memory between two barriers rather
  than extra launches, and a single chunk compiles back down to the plain scan.
  The split only pays while B·D is small, so the chunk count is chosen from the
  shape (override: `TMT_CHUNKS`). Isolated, this made forward plus backward
  1.5–1.9× faster at typical window shapes.
- The backward kernel is hand-written and exact within the window, including
  the contribution of the **incoming carry** to the decay/gate gradients, and
  it returns λ = dL/d(carry). It is attached with `mlx_custom_vjp`, so MLX
  autodiff composes with it.
- **Hybrid traces** (`traces=1`): the trace advance
  `e_out = P·e_in + Σ_t (Π_{k>t} γa_k)·local_t` is a forward quantity, so it is
  a separate kernel whose inputs are stop-gradiented; the credit λ·e_in is
  added to the decay, gate and embedding gradients after the backward pass.
  The embedding trace (layer 0) is a second kernel, as in CUDA.
- Verified by finite differences against an FP64 CPU reference (gated and
  ungated, nonzero carry), and the autodiff wrapper is checked against the
  kernels.

## 4. MoE (moe path in model.zig)

- Top-k routing on fp32 router logits; softmax, top-k, renormalized weights.
- **Token dispatch:** every (token, slot) pair becomes a row, the rows are
  permuted so each expert owns a contiguous segment, and one `gather_mm` runs
  them. MLX then differentiates it with a segmented matmul. Where CUDA computed
  histogram and offsets on the host and issued one GEMM per expert, this needs
  no host synchronization at all — counts stay on the device and are read once
  per window for the router-balance log.
- SiLU and the weighted combine are graph ops in fp32 on the expert
  pre-activations; the residual add is fused into the same expression.
- The **regularizers** (switch aux `E·Σ(mean_p·frac)`, z-loss
  `mean(logsumexp²)`) are written as loss terms and differentiated. The CUDA
  build carried their analytic router gradients by hand; the finite-difference
  test from CUDA is kept and passes against the autodiff version.
- `experts=1` takes a dense path without any routing, exactly as before.

## 5. MLA (mla block in model.zig, default: off)

- Same principle: per position only a compressed latent (rank L) and a
  decoupled RoPE key (dim R) in the stream cache.
- Keys are `concat(stop_gradient(cache), window)`: the past contributes no
  gradient to the window latents (TBPTT), while the up-projections `Wuk/Wuv`
  still receive gradients from the whole cache — the CUDA semantics.
- **Chunked online softmax** over `mla_cc` keys per step with fp32
  accumulators, causal mask on absolute positions, and `m` stop-gradiented
  (the result does not depend on it). The backward is MLX's, not a
  recompute-per-chunk implementation; there is no LSE cache and no dS buffer.
- RoPE is applied on adjacent pairs with host-computed cos/sin tables for the
  window's absolute positions; keys are rotated before they enter the cache.
- **Eviction** drops only the oldest prefix when a window no longer fits, and
  `base0` keeps absolute positions. The cache is a plain array pair per layer
  (append and slice) instead of a ring buffer, which removes the wrap-around
  index arithmetic of the CUDA version.
- Verified against the same independent FP64 CPU reference (forward and all
  six weight gradients plus the input gradient), and chunk-size independence.

## 6. Compilation

The whole window — forward *and* backward — is handed to `mx.compile` as one
function (`mx.compile(value_and_grad(f))`, not `value_and_grad(mx.compile(f))`,
which would leave the backward unfused). MLX keys its plan on the input shapes,
so a compiled plan must not capture anything that changes between windows: the
MLA cache, the hybrid traces and the RoPE tables are therefore explicit inputs
of the window function, next to weights, carries and the batch. An MLA cache
that is still filling changes shape every window, so compilation is held back
until it is full. `TMT_COMPILE=0` runs the graph uncompiled; gradients are
identical either way (`gradcheck` output matches bit for bit).

AdamW is compiled the same way: one call covering every parameter, the gradient
norm, clipping and the EMA target, instead of a dozen op launches per tensor.
That is this port's answer to the multi-tensor optimizer kernel of the CUDA
build, and it cut the optimizer from 6.4 to 2.5 ms (dense) and from 34 to 10 ms
(8 experts).

## 7. Streaming state and the shared forward path

`State` (carry per layer, MLA caches, position, traces) is separate from the
model, as before. `forwardWindow` runs one window for training, evaluation or
generation; `.grad = true` differentiates it and leaves the gradients in
`m.grads`. The per-window graph is built once and evaluated once, so a step
costs one host synchronization (plus one for the gradient-norm check).

## 8. Losses and optimizer

- `loss = var·hinge + latent·MSE + ce·CE + stop·BCE + (aux+z)/layers`,
  defaults CE-only, all terms individually switchable — unchanged, including
  `pos_weight` on the stop head and the EMA target encoder for the latent term.
- AdamW (β 0.9/0.999, eps 1e-8, wd 0.01), global-norm clipping, warmup+cosine
  schedule, EMA update, and the **NaN guard**: a non-finite gradient norm
  refuses the whole step. Built from MLX ops over the parameter list, checked
  element by element against a host reference (the CUDA multi-tensor kernel's
  test, ported).

## 9. Determinism

Resume is checked numerically (`architecture_test`, `tests/cli.sh`), which
requires gradients to be reproducible. Two places would otherwise use atomic
scatter-adds, whose order is not fixed:

- the **embedding gradient** — computed as a one-hot matmul instead;
- **MoE dispatch** — rows are sorted by expert so both backward paths are
  segment-based.

Everything else is deterministic MLX graph evaluation.

## 10. Checkpoints

Identical V3 layout to CUDA (see `checkpoint.zig` and the CUDA
`ARCHITECTURE.md` §8), written via temporary file + fsync + rename, checksum
verified before restore, schema round-trip checked, and tolerant of the four
keys added after the format was fixed (`traces`, `trace_decay`, `docsep`,
`dialog`).

## 11. What is deliberately not ported

The CUDA-specific measurement tools: `profile` (MoE share of the BF16 roof via
cuBLAS) and `roofline` (DRAM and GEMM ceilings). `bench` covers their role here:
recurrence kernel correctness and timing plus the machine's bandwidth and
matmul roof. `TMT_CUDA_WAIT` has no MLX equivalent and is gone.

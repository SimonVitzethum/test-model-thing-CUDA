# TMT-MLX

Apple-Silicon path of the byte model: **Zig + MLX**, no Python at build or run
time. It is a port of [`tmt-cuda`](../tmt-cuda/README.md) with the same
architecture, the same CLI, the same checkpoint format (V3) and the same tests.

## Building and testing

Requires macOS on Apple Silicon, the Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`), Zig 0.16 and MLX with its C
API (mlx-c) — no Python, no PyTorch.

```sh
cd tmt-mlx
zig build                     # expects MLX under ~/.local/mlx
zig build -Dmlx=/path/to/mlx  # other install prefix
zig build check               # architecture tests + CLI tests (needs a GPU)
```

Building MLX and mlx-c once (CMake, no Python bindings):

```sh
git clone --depth 1 -b v0.6.0 https://github.com/ml-explore/mlx-c ~/.local/opt/mlx-c
cmake -S ~/.local/opt/mlx-c -B ~/.local/opt/mlx-c/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$HOME/.local/mlx -DBUILD_SHARED_LIBS=ON
cmake --build ~/.local/opt/mlx-c/build -j8 && cmake --install ~/.local/opt/mlx-c/build
```

`zig build check` requires a GPU but neither Python nor PyTorch. It checks:

- normalized recurrence with/without input gate, nonempty carry, and numeric gradients;
- the autodiff wrapper around the recurrence against its hand-written kernels;
- gradients of MoE load balancing and router z-loss;
- MoE against the dense path, and learning progress;
- the optimizer (clipping, AdamW, EMA, refused steps) against a host reference;
- separate streaming states, checkpoint resume, and corruption detection;
- MLA with multiple/partial chunks, cache eviction, and nonzero RoPE positions;
- MLA forward and all gradients against an independent FP64 CPU reference;
- CLI training, evaluation including tail windows, and invalid inputs.

`bench [B T D]` measures the recurrence kernel against a CPU reference and
prints the machine's memory bandwidth and bf16 matmul roof.

## Model and architecture decisions

```text
Bytes → Embedding → [hierarchical recurrence → LayerNorm → Dense/MoE + Residual] × L
                        optionally after selected blocks: LayerNorm → MLA + Residual
      → final representation → byte logits (256)
```

Per channel, layer, and byte:

```text
a_t = sigmoid(decay + gate * x_t)
s_t = a_t * s_(t-1) + (1 - a_t) * x_t
x_out = x_t + FFN(LayerNorm(s_t))
```

The architecture, the defaults and the semantics are those of the CUDA version;
see [ARCHITECTURE.md](ARCHITECTURE.md) for what the port changed and what it
kept. The default is a small dense model: `dim=256 layers=4 experts=1 topk=1`.

Master weights, recurrent state, optimizer moments and reductions are FP32.
The working precision of the forward pass is FP32 as well, unlike the CUDA
build: Apple GPUs have no BF16 matmul units, so BF16 is not the fast format
here (measured ~17% slower end to end, see below). `TMT_COMPUTE=bf16` restores
the CUDA numerics and halves activation memory, `TMT_COMPUTE=f16` sits in
between. All three pass the full test suite; the checkpoint format does not
depend on the choice.

## Training and evaluation

The data file holds raw bytes, no tokenizer; it is read through `mmap`.

```sh
# New run; steps=0 runs until aborted.
./zig-out/bin/train train.bin model.ckpt steps=1000 saveevery=100

# Resume: configuration, Adam, data position and stream state are restored.
./zig-out/bin/train train.bin model.ckpt steps=1000

# Weights unchanged, fresh stream, no checkpoint writes.
./zig-out/bin/train validation.bin model.ckpt mode=eval

# MoE and MLA experiments.
./zig-out/bin/train train.bin moe.ckpt experts=4 topk=2 steps=1000
./zig-out/bin/train train.bin attention.ckpt mla=1 mla_every=2 mla_cache=4096 steps=1000

# Generation and interactive chat (never modify weights).
./zig-out/bin/sample model.ckpt "The history of" temp=0.7 maxlen=256
./zig-out/bin/chat model.ckpt
```

All options, defaults, resume rules, `maxcarry`, `docsep`, `dialog`, hybrid
traces (`traces=1`, `trace_decay`), the `gradcheck` tool and the JSON result
line behave exactly as documented in the
[CUDA README](../tmt-cuda/README.md#training-and-evaluation); the only
difference is the path to the executables (`./zig-out/bin/...`).

| Options | Default / meaning |
|---|---|
| `dim`, `layers`, `experts`, `topk` | `256`, `4`, `1`, `1` |
| `batch`, `seqlen` | `8`, `128`; TBPTT window in bytes |
| `gated`, `half_min`, `half_max` | `1`, `2`, `512` |
| `lr`, `warmup`, `decaysteps`, `minlr` | `0.0005`, `200`, `8000`, `0.1` |
| `ce`, `latent`, `var`, `stop` | `1`, `0`, `0`, `0` |
| `aux`, `zloss` | `0.01`, `0.001`; MoE only |
| `gradclip`, `ematau`, `seed` | `1`, `0.99`, `1234` |
| `maxcarry`, `traces`, `trace_decay`, `docsep`, `dialog` | `0`, `0`, `1`, `-1`, `0` |
| `mla`, `mla_every`, `mla_cache`, `mla_cc` | `0`, `2`, `4096`, `256` |

## Checkpoints are shared with the CUDA build

Format V3 (`TMTCPKT3`) is written byte for byte as in `tmt-cuda`: magic,
configuration as text, progress, per parameter master/m/v, stream position,
carries, MLA cache, traces, FNV checksum; written through a temporary file
with `fsync` and atomic rename, and verified before restore.

Two properties are checked against the CUDA sources on this machine:

- the configuration header (including `%.9g` float formatting) is identical;
- the weight initialization is **bit-identical** for the same seed (same LCG,
  same consumption order, same fused multiply-add).

Parameter order, shapes and the state layout are the same as well, so a
checkpoint trained with CUDA is meant to continue here and vice versa. That
direction has not been executed end to end (this machine has no NVIDIA GPU).

## Measured on an Apple M2 (10-core GPU, 16 GB)

| Setup | Throughput | Before tuning |
|---|---|---|
| `dim=256 layers=4`, batch 8 × 128 | 173k bytes/s | 107k |
| `dim=512 layers=8`, batch 8 × 128 | 62k bytes/s | 37k |
| `dim=512 layers=8`, batch 32 × 256 | 80k bytes/s | 50k |
| `dim=512 layers=8 traces=1`, batch 8 × 128 | 51k bytes/s | — |
| `dim=512 layers=8 experts=8 topk=2`, batch 8 × 128 | 21k bytes/s | 11k |
| `dim=512 layers=8 mla=1 mla_cache=4096`, batch 8 × 128 | 10k bytes/s | 7k |

Where the time goes at `dim=512 layers=8` (batch 8 × 128): 16.0 ms forward and
backward, 2.5 ms optimizer. Of one layer's 1.8 ms, the recurrence is about 20%
and the rest is the feedforward matmuls plus LayerNorm.

For reference, `bench` measures ~77 GB/s memory bandwidth and ~2.2 TFLOPS
matmul on the same machine; an RTX 5070 laptop GPU is roughly two orders of
magnitude faster in matmul. These numbers say what this hardware does, not what
the architecture costs.

### What the tuning did

- **The window graph is compiled** (`mx.compile` over the gradient function, not
  just the forward): MLX fuses the elementwise chains of forward *and* backward.
  Worth ~20% of the step on its own. Because a compiled plan freezes whatever it
  captures, the MLA cache, the traces and the RoPE tables are passed in as
  explicit inputs. `TMT_COMPILE=0` disables it.
- **AdamW is one compiled call** over all parameters instead of a dozen op
  launches per tensor: 6.4 → 2.5 ms for a dense model, 34 → 10 ms with 8
  experts (the CUDA build hit the same wall and solved it with a multi-tensor
  kernel).
- **The recurrence scan is split over time** (see ARCHITECTURE.md §3): the
  kernel reached 10 GB/s with 4k chains but 75 GB/s with 256k, so short windows
  were starving the GPU. Isolated, forward plus backward got 1.5–1.9× faster.
  `TMT_CHUNKS=N` overrides the split.
- **FP32 instead of BF16 working copies**, as described above.

An MLA run compiles once per cache size while the cache is still filling, so
compilation is held back until the cache is full and the shapes stop changing.

## How the port works

| File | Responsibility |
|---|---|
| `src/mlx.zig` | Zig layer over the MLX C API: arrays, ops, kernels, transforms, lifetimes |
| `src/cell.zig` | Recurrence as Metal kernels (chunked scan, exact backward, traces) with a custom VJP |
| `src/model.zig` | Parameters, state, forward (recurrence, MoE, MLA), gradients, AdamW |
| `src/config.zig` | One configuration schema for CLI, validation and checkpoints |
| `src/checkpoint.zig` | V3 format, checksum, resume validation |
| `src/data.zig` | `mmap` datasets, dataset hash, dialog control bytes |
| `src/train_main.zig` | Data windows, training/evaluation loop, diagnostics, JSON record |
| `src/generate.zig`, `src/sample_main.zig`, `src/chat_main.zig` | Single-step generation and the interactive CLI |
| `src/gradcheck_main.zig` | TBPTT vs hybrid traces against full BPTT |
| `src/architecture_test.zig` | Native numeric and integration tests |
| `tools/dialogprep.zig` | oasst/TSV dialog data (JSON scanner included) |

The forward pass is one pure function of arrays (`forwardCore`). Training
differentiates it with `mlx_value_and_grad`; evaluation and generation call it
directly, so all three share one code path — the CUDA build's `forward_window`
contract, expressed in MLX.

Two pieces are not left to autodiff:

- **The recurrence** is a Metal kernel with a hand-written backward
  (`cell.zig`), because the scan over the window must stay a single kernel and
  because the backward also produces λ = dL/d(carry), which the hybrid traces
  need. It is registered with `mlx_custom_vjp`, so it composes with MLX autodiff.
- **The embedding gradient** uses a one-hot matmul instead of MLX's
  scatter-add, and **MoE dispatch** sorts its rows so both backward paths are
  segment-based. Atomic scatter-adds would otherwise make gradients differ
  between two identical runs, which would break resume determinism.

Everything else — LayerNorm, matmuls, MoE (`gather_mm`), MLA with chunked
online softmax, the losses, AdamW — is MLX graph code; the optimizer is
built from ops rather than a fused kernel.

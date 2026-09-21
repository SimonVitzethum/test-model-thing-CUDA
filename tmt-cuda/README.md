# TMT-CUDA

Active CUDA development path of the byte model. PyTorch (`main_torch_moe.py`,
`benchmark_torch_moe.py`) is deprecated; the MLX model documents the original
prototype. New CUDA checkpoints are compatible with neither.

## Building and testing

Requires Linux, an NVIDIA driver, CUDA with BF16/cuBLAS, and a C++17 host compiler.
The Makefile default is consumer Blackwell (`sm_120a`). Other GPUs need
matching `ARCH` flags and BF16 support, for example:

```sh
cd tmt-cuda
make
make check
# Example for a different target architecture:
make ARCH='-gencode arch=compute_89,code=sm_89'
# After changing ARCH, run `make clean` first.
```

`make check` requires a GPU but neither Python nor PyTorch. It checks:

- normalized recurrence with/without input gate, nonempty carry, and numeric gradients;
- gradients of MoE load balancing and router z-loss;
- dependence of higher layers on lower experts, and learning progress;
- separate streaming states, checkpoint resume, and corruption detection;
- MLA with multiple/partial chunks, cache eviction, and nonzero RoPE positions;
- MLA forward and all gradients against an independent FP64 CPU reference;
- CLI training, evaluation including tail windows, and invalid inputs.

Additional memory checking:

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 ./architecture_test
./bench 2 32 128
```

`bench` measures only the recurrent cell. It is not a throughput benchmark of the
entire training run. Its bandwidth figure is computed from estimated data traffic.

## Model and architecture decisions

```text
Bytes → Embedding → [hierarchical recurrence → LayerNorm → Dense/MoE + Residual] × L
                        optionally after selected blocks: LayerNorm → MLA + Residual
      → final representation → byte logits (256)
```

Each layer receives the residual stream **of the previous layer**. The previous
prototype fed all recurrent layers directly from the same byte embedding.
The new structure enables hierarchical feature processing and routes the
gradient back through all previous layers accordingly.

Per channel, layer, and byte:

```text
a_t = sigmoid(decay + gate * x_t)
s_t = a_t * s_(t-1) + (1 - a_t) * x_t
x_out = x_t + FFN(LayerNorm(s_t))
```

The normalized update bounds accumulation at slow decays. The gate
starts at zero and is learned; `gated=0` allows the static ablation.
Initial half-lives are logarithmically spread from `half_min=2` to `half_max=512`
bytes. With an input-dependent gate, the actual timescale then becomes
context-dependent. A running state does not guarantee unbounded
memory.

The default is a small dense model: `dim=256 layers=4 experts=1 topk=1`.
The dense path performs no routing operations. For `experts>1` only
assigned token-expert pairs are computed. The load-balancing loss uses the
assignment fractions `counts / (N * topk)`; its gradient and the z-loss flow
into the router. MoE dispatch still uses dynamic allocations and
host synchronization. A blanket MoE speedup is not established.

BF16 is used for working weights/activations and GEMMs. Recurrent
states, master weights, Adam moments, and essential reductions are FP32.
Training uses TBPTT: the incoming carry is detached at the window boundary
but correctly contributes to the decay/gate gradient of the first byte.

Next-byte cross-entropy is the default objective. `latent=0 var=0 stop=0`
disables the extra objective terms. They can be enabled for controlled ablations.
The EMA target encoder is updated exclusively by EMA, not by AdamW.
The optional stop loss uses the next newline byte as target; this is not a
general dialog end marker.

## Small shared interface

| File | Responsibility |
|---|---|
| `src/config.h` | One configuration schema for CLI, validation, and checkpoints |
| `src/train.cu` | Shared CUDA operators and model-owned parameter store |
| `src/model.cu` | Model construction, explicit state, forward, backward, and optimizer step |
| `src/checkpoint.h` | Versioned storage, verification, and restore |
| `src/train_main.cu` | Data access, training/evaluation flow, and measurement output |
| `src/architecture_test.cu` | Native numeric and integration tests |

The internal CUDA interface is deliberately small:

```cpp
build_model(model);
build_state(state, model);
forward_window(model, state, loss, ce); // updates only the passed stream
// model.X: final representations [B,T,D]; model.logits: [B,T,256]
backward_window(model, state);         // training only, right after forward
optimizer_step(model, step);
release_window(model);                // release forward scratch after use
reset_state(state, model.c);          // weights stay unchanged
```

Multiple models have separate parameter stores. Multiple `StreamState` objects
can use the same model one after another. The forward workspace belongs
to the model; concurrent calls on the same model are not supported.
Forward must have run on the same model immediately before a backward.
Representations are overwritten by the next forward and must be copied if needed.
This later allows correct classification probes on the actual model output.
A CUDA CoLA adapter is not implemented yet.

## Training and evaluation

The data file holds raw bytes, no tokenizer. `mmap` allows OS-level file paging;
there is no 8 GB file limit anymore and no restriction to the first MiB. The file
must stay unchanged during a run. Resume checks its identity by size and content hash.

```sh
# New run; limited budget recommended, steps=0 runs until abort.
./train train.bin model.ckpt steps=1000 saveevery=100

# Configuration, Adam, data position, and streaming state are loaded.
# steps counts additional steps, not the global target step count.
./train train.bin model.ckpt steps=1000

# Weights unchanged; fresh stream, no checkpoint writes.
./train validation.bin model.ckpt mode=eval

# Standalone MoE experiment, same data/budget comparison required.
./train train.bin moe.ckpt experts=4 topk=2 steps=1000

# Ablations each with own checkpoint path and fixed seed.
./train train.bin static.ckpt gated=0 steps=1000
./train train.bin latent.ckpt latent=1 steps=1000
```

Existing checkpoints supply their configuration automatically. Diverging
model or training parameters are rejected on resume. A different configuration
gets a new checkpoint path. `mode`, `steps`, and `saveevery` are run control
and are not part of the stored configuration. `saveevery=0` saves only at the end.
SIGINT/SIGTERM stops after the current window and saves during training; after a
process crash the last complete checkpoint remains.

Important options:

| Options | Default / meaning |
|---|---|
| `dim`, `layers`, `experts`, `topk` | `256`, `4`, `1`, `1` |
| `batch`, `seqlen` | `8`, `128`; TBPTT window in bytes |
| `gated`, `half_min`, `half_max` | `1`, `2`, `512` |
| `lr`, `warmup`, `decaysteps`, `minlr` | `0.0005`, `200`, `8000`, `0.1` |
| `ce`, `latent`, `var`, `stop` | `1`, `0`, `0`, `0` |
| `aux`, `zloss` | `0.01`, `0.001`; MoE only |
| `gradclip`, `ematau`, `seed` | `1`, `0.99`, `1234` |
| `maxcarry` | `0`: no periodic reset; positive values: reset between windows |

The file is split into `batch` contiguous streams. Each stream
starts with empty state. Training processes complete windows; at the
end of an epoch leftover bytes are discarded and all states reset.
Evaluation also processes tail windows; padding does not enter CE/BPB.
Transitions between streams are not scored. Comparisons must therefore
use the same split and the same reset mode.

The last output line is a JSON object with `mode`, `steps`, `bytes`, `ce`,
`bpb`, `seconds`, and `bytes_per_second`. BPB means bits per byte (`CE / ln(2)`),
not perplexity per subword token. The training measurement includes forward,
backward, updates, and periodic checkpoints where applicable; initialization and the
final checkpoint are outside the timing. Evaluation includes forward and
scoring. A run with `steps>0 mode=eval` scores only a prefix.

## Checkpoints and reproducibility

Format V3 stores the full configuration, FP32 master weights,
Adam moments, EMA encoder, global optimizer step, data position/epoch,
dataset fingerprint, recurrent states, and the valid MLA cache including
absolute positions. BF16 working weights are reconstructed from these.

Files are replaced via a temporary file with flush and atomic rename.
A content checksum is verified before restore. Old V2, MLX, and
PyTorch checkpoints are not silently loaded as new models.
V3 is a native Linux 64-bit binary format, not a portable exchange format.
The hash detects accidental changes, not authentication.

Seed and data progress are reproducible. CUDA atomics and cuBLAS can
cause small rounding differences; bit-identical results across arbitrary
GPUs, drivers, or dispatch orders are not guaranteed. Resume is
checked numerically against an uninterrupted run.

## Optional MLA cache

```sh
./train train.bin attention.ckpt mla=1 mla_every=2 mla_cache=4096 mla_cc=256 steps=1000
```

MLA is disabled by default. The cache stores compressed KV latents plus RoPE keys
per active attention block and stream. The pure cache size is:

```text
batch × number of MLA blocks × mla_cache × (mla_L + mla_R) × 2 bytes
```

Weights, optimizer, activations, and attention workspace come on top.
Attention reads the entire valid cache in chunks. Its compute cost therefore
grows with context length despite compressed storage.

On overflow only the oldest necessary prefix is removed. Eviction happens
before a whole window: the first byte of that window has up to `seqlen-1`
fewer older positions available than with a strictly byte-wise sliding
window. Absolute RoPE positions are preserved. Past cache entries
are detached at the TBPTT boundary; the up-projections still receive gradients.

`mla_cache=131072` is configurable, but neither retrieval quality nor throughput
at 128k bytes is covered by the small regression tests. The cache is not a
lossless archive and there is no guaranteed "exact 128k retrieval".

## Plan: RSI learning across multiple task families

**Status: planned, no automatic RSI controller implemented.** The current
CUDA training and the non-modifying evaluation form the executable foundation.

[Dream-RSI](https://arxiv.org/html/2609.14858v1) improves, in the paper, the executable
exploration strategy with a fixed agent. Historic attempt trees serve
as replay worlds; new strategies decide continuation, branching, and
termination. The method does not replace weight learning. Replay covers only actually
observed continuations. Improvements on old trees guarantee no
transfer to new tasks. The following transfer to TMT is a project plan.

### 1. Reliable tasks and baselines

A fixed evaluator receives only a model artifact, a versioned
task manifest, and a budget. All model changes happen outside the
evaluator. First, the following CUDA adapters are implemented:

| Family | Tasks / protocol | Target metric |
|---|---|---|
| Language | Held-back texts from multiple sources | BPB per source |
| Grammar | CoLA probe on final representation; BLiMP sentence likelihoods | MCC / pair accuracy |
| Memory | Copy, delayed retrieval, key-value mapping | Accuracy by distance |
| Algorithms | Addition, bracket checking, small state machines | Exact solution, longer inputs |
| Continual learning | Domain switches and return to old tasks | Adaptation and forgetting |
| Resources | Fixed shapes and warmup rules | Runtime, bytes/s, peak GPU memory |

[BLiMP](https://github.com/alexwarstadt/blimp) is a collection of grammatical
minimal pairs. For byte models the full sentence likelihood must
be compared with identical state initialization.

Training, development, and locked test data are separated.
Generated tasks get separate seeds plus additionally held-back
lengths/structures. Entire task families are excluded for transfer tests.
Test data must not enter prompts, strategy selection, or weight updates.

Acceptance: reproducible manifests and individual measurements, at least three seeds,
comparison of the dense baseline against static/gated recurrence at equal
budget. The in-repository tests do not replace these quality benchmarks.

### 2. Uniform experiment protocol

A small runner (Rust is intended for process management and storage) starts
CUDA processes with limited GPU time and collects their JSON results. There is
first a local job queue instead of a distributed service architecture.
A task adapter delivers inputs and verifiable
results through the same interface; each benchmark gets no training system of its own.

Each attempt logs:

- attempt ID, parent ID, code/build hash, and full configuration;
- data/task manifest, seeds, and hardware/driver version;
- starting and ending checkpoint including optimizer/streaming state;
- all individual metrics, errors, and aborts;
- GPU time, peak memory, and agent tokens/cost.

The later controller interface reads conceptually:

```text
select(observed_history, remaining_budget) -> parent_ids
execute(parent_id, experiment_spec, budget) -> observation + artifact
```

Empty selection ends a search. Only compatible
training continuations from a checkpoint are supported at first. A changed architecture
starts as a new root attempt; today's strict checkpoint loading is
not relaxed for that. A curriculum extension must log dataset changes explicitly
and gets a separate import/continuation contract.

Acceptance: repeatable resume after abort, immutable evaluator,
equal budgets for all candidates, and fully traceable provenance.

### 3. Fixed search before learned search

First implement random search, a fixed branching strategy, and successive halving as
comparison. Start small: gates, timescales, learning rate,
window length, and loss ablations. Add MoE/MLA only after stable baselines.
An existing coding agent can suggest changes later; whether the small
TMT itself can write such code is unproven so far.

Quality is normalized per task family against pre-set baselines. Families
are weighted equally, not by number of subtests. Individual values stay
visible. A possible selection rule reads:

```text
score = mean(normalized_family_scores) - lambda * normalized_total_cost
```

Costs include online attempts AND strategy development. Resource limits,
correctness, and maximum allowed regressions are hard admission criteria.
Weights/normalization are fixed before the experiment, not adjusted after seeing
test results.

### 4. Historic replay and real validation

Versioned replay trees grow from logged attempts. Every alternative
strategy starts with empty observation state. The controller sees only
already revealed results. Neither future scores nor hidden branches
are accessible as features.

A recorded continuation is reused only if parent artifact,
action, and execution context match. Missing continuations mean
"unknown"; they get no invented success or failure assigned. New
actions must run online. The search keeps adding new
root attempts to extend historic coverage.

Strategies are developed and validated on separate historic trees.
The best replay candidate then competes online against the incumbent strategy.
It is adopted only on robust advantage under equal total budget. Metrics:
quality at fixed budget, cost to a target quality, regressions,
and transfer to held-back tasks.

### 5. Actual learning of the model weights

The selected training runs improve TMT by gradient learning. Verifiable
synthetic tasks deliver input-target pairs; correct agent solutions can
serve as additional training data after review and deduplication. Earlier
domains stay in a fixed mixture to measure
and limit forgetting. A benchmark total score alone is not sufficient
supervised training signal.

Reinforcement learning would be a separate experiment with fixed verifier,
policy/reference checkpoint, and independent evaluation. It is not part of the first
Dream-RSI integration. First, the combination of supervised tasks,
reproducible experiments, and learned search control must beat the fixed baselines.

## Verification status

The native architecture tests and CLI tests ran on an NVIDIA GeForce
RTX 5070 Laptop GPU. CUDA Compute Sanitizer (`memcheck`) reported no errors
for the architecture tests. This confirms the tested small shapes
and code paths; model quality on the planned benchmarks and scaling to
128k bytes remain open.

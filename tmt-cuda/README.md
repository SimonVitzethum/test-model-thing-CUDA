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
gets a new checkpoint path. With `mode=eval`, `seqlen`, `maxcarry`, `docsep` and `dialog` may also differ (e.g. `maxcarry=1024` for a context ablation). `mode`, `steps`, and `saveevery` are run control
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
| `traces` | `0`; `1` enables hybrid traces across window boundaries (see below) |
| `trace_decay` | `1`; per-byte trace decay γ (`e_t = γ·a_t·e_(t-1) + …`), independent of `seqlen` |
| `docsep` | `-1`; byte value that starts a new document (state and traces reset there) |
| `mem`, `mem_len`, `mem_heads`, `mem_dh`, `mem_every`, `mem_rdim` | `0`, `256`, `4`, `32`, `2`, `0`; fact memory and stage-2 retrieval (see below) |
| `dialog` | `0`; `1` scores only assistant turns of dialog data (see below) |

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

## Hybrid traces (optional)

Truncated BPTT cuts the gradient at the window boundary, even though the state
itself is carried on indefinitely. `traces=1` restores gradient credit from
before the window for the per-channel parameters, in the spirit of the original
TMT trace system:

```sh
./train train.bin traced.ckpt traces=1 half_max=65536 steps=1000
```

Because the recurrence is diagonal, `e = ds_(t0-1)/dθ` is exact and cheap for
the decay and gate of every layer and for the embedding feeding layer 0. The
backward pass already computes `λ = dL/ds_(t0-1)` at the window start, so
`λ·e` is added to the gradient, and the trace is advanced in closed form
(`e_out = P·e_in + Σ_t (Π_{k>t} a_k)·local_t`) within the same kernel.
Within a layer these parameters thus get an unbounded, exponentially decaying
gradient horizon. The window length becomes a knob: `seqlen=1` is a pure online
trace system, larger windows add exact lookback and GPU efficiency.

Limits: paths that cross both a layer and the window boundary (layer l changing
layer l+1's carried state) are not traced, and neither are router, expert, and
MLA weights; those keep the window horizon. Traces are computed with the
parameters of earlier windows. Extra state is `(2·layers + 256)·batch·dim`
floats (about 11 MB for dim=512, layers=16, batch=16); measured throughput cost
is below 1 %. Traces advance only during training and reset with the state.
`make check` verifies that, for one layer, two windows with traces reproduce the
full BPTT gradient of one double-length window for all parameters.

### Document resets, trace decay and diagnostics

`docsep=B` resets the recurrence at every input byte equal to `B` by forcing
`a_t = 0` there, so `s_t = x_t`. Carry, gradient flow and traces are cut by the
same recurrence; no separate code path exists. It applies in training,
evaluation and sampling alike, so the context table measures what training
saw. The MLA cache is not reset. enwik8 has no single-byte separator; insert one
before each article, e.g. `perl -pe 's/<page>/\x1e<page>/g'` and `docsep=30`.

`trace_decay=γ` damps old trace contributions per byte. Because it is defined
per byte, the traces after a given number of bytes do not depend on `seqlen`
(tested). `γ=1` keeps exact traces.

Every 100 training steps the log shows:

- `traces:` (with `traces=1`) the norm ratio and cosine between the trace part
  and the in-window part of the decay, gate and embedding gradients. A cosine
  near −1 or wild fluctuations mean the traces fight the window gradient.
- `state:` the mean |state| per half-life bucket (from the decay alone; the
  gate makes the effective half-life input-dependent). It shows whether long
  channels store anything. With `s = a·s + (1−a)·x`, a channel with a 65k
  half-life takes each byte in with weight ~1e-5: it holds a slow average unless
  the gate lowers `a` for selected inputs.

### Gradient comparison (`gradcheck`)

```sh
./gradcheck held.bin model.ckpt len=4096 window=128 seqs=4 [trace_decay=… docsep=…]
```

Using the trained weights, `gradcheck` computes the exact BPTT gradient of
`seqs` sequences of `len` bytes and reports, per parameter type, the cosine
similarity of truncated BPTT (`traces=0`) and the hybrid (`traces=1`) to it.
It trains nothing and is much cheaper and less noisy than training runs. On the
8h checkpoint (half_max=512, experts untrained, see CHANGES.md) at `len=4096`:

| params | cos TBPTT | cos hybrid |
|---|---|---|
| decay | 0.9950 | 0.9956 |
| gate | 0.9791 | 0.9951 |
| embedding | 0.9946 | 0.9942 |
| experts | 0.9941 | 0.9938 |

With half-lives up to 512 bytes, TBPTT is already close to exact and the hybrid
mainly helps the gate. Checkpoints trained with long half-lives are the real
test.

## Knowledge-graph fact memory (`mem=1`, stage 1)

Idea from the TMT community: facts live in a knowledge graph instead of the
weights, and the model reads them through its own state. Stage 1 answers the
first question: **can the model use facts it is given?** Retrieval is done by the
data loader (the subject of each question is known); state-driven retrieval
(stage 2) builds on this.

**Architecture.** The facts of the subject are placed as bytes in a per-stream
memory of `mem_len` slots. Each slot is encoded as
`e_j = E[byte_j] + Eprev[byte_(j-1)] + P[j]` (sharing the model's byte embedding),
followed by a causal gated recurrence over the memory (`m_j = e_j + s_j`, half-lives
1..64 bytes), so a slot knows what precedes it ("Asia" follows "continent: ").
After every `mem_every`-th layer, a cross-attention reads the memory:
`x += softmax(LN(x)Wq (mWk)^T / sqrt(dh)) (mWv) Wo^T`. `Wo` starts at zero, so an
untrained memory, or an empty one, is an exact no-op (tested bit for bit); memory
parameters are created last, so all other weights initialize as with `mem=0`.
Forward and all gradients are tested against an FP64 CPU reference.

**Data (C++ for everything compute-heavy).**

```sh
# Subset through the Wikidata API (network-bound helper, standard library only)
python3 tools/wikidata_kg.py --seed P31=Q6256 --seed P31=Q1549591 \
    --seed P106=Q36180 --per-seed 3000 --out graph.tsv
# ...or the full dump in one streaming, multi-threaded pass
pigz -dc latest-all.json.gz | ./kgprep dump graph.tsv
# QA rows (question, answer, memory), split by subject
./kgprep qa graph.tsv kg mem_len=256 test_frac=0.1
```

`kgprep dump` keeps the item-valued facts of 15 properties (capital, continent,
currency, country, occupation, author, ...) for entities with an English
Wikipedia article, preferred rank over normal, no deprecated statements, no
self-references. It holds every English item label in RAM (several GB for the
full dump). `kgprep qa` renders each subject's facts as
`France: capital: Paris; continent: Europe; ...` (shuffled, whole facts within
`mem_len`) and asks about single-valued facts only. Test subjects never appear
in training, so the test set measures use of the memory, not memorization.

**Training and evaluation.**

```sh
./kgtrain train kg_train.tsv kg.ckpt dim=256 layers=4 batch=32 seqlen=128 steps=20000
./kgtrain eval kg_test.tsv kg.ckpt memory=on        # facts of the subject
./kgtrain eval kg_test.tsv kg.ckpt memory=off       # no memory
./kgtrain eval kg_test.tsv kg.ckpt memory=shuffled  # another subject's facts (control)
```

Each example is one window `question answer\n`; the loss covers the answer only.
Evaluation reports exact match (every answer byte is the argmax given the correct
prefix) and answer CE. The memory helps only if `on` beats both `off` and
`shuffled`; `on ≈ shuffled` means the model ignores the memory's content.

**Stage-1 result** (2026-09-21, RTX 5070 Laptop). Wikidata subset from the API
(seeds: countries, big cities, writers, politicians, films): 18,011 nodes,
59,689 facts; 19,843 training examples (9,098 subjects), 2,269 test examples on
1,008 subjects that never appear in training. Model `dim=256 layers=4`, memory
defaults, `batch=32 seqlen=128`, 20,000 steps (8 minutes):

| Memory | Exact match (test subjects) | Answer CE |
|---|---|---|
| on (the subject's facts) | **90.3%** | 0.07 |
| off | 0.1% | 2.70 |
| shuffled (another subject's facts) | 4.0% | 8.21 |

On training subjects: 99.95% with the memory, 4.2% with shuffled facts, so the
facts are read from the memory rather than memorized in the weights. Most
remaining errors are single-byte copy slips ("Kithuanian litan"). An earlier run
on only 464 examples memorized instead (`on ≈ shuffled`); more subjects made
copying the cheaper strategy.

### Stage 2: retrieval from the model's own state (`mem_rdim > 0`)

No oracle: the model finds the subject itself.

1. It reads the question (no memory). Its final representation at the last
   question byte, through a learned head `Wq` (`mem_rdim` x `dim`), is the query.
2. It reads every node label of the graph, including object-only nodes such
   as "Paris" as distractors. The representation at the last label byte,
   through `Wk`, is the node's key. Keys come from the label bytes, not from
   per-node parameters, so unseen entities get meaningful keys.
3. The node with the highest cosine similarity wins; its facts
   (`kgprep qa` writes them to `PREFIX_nodes.tsv`) go into the memory and the
   model answers as in stage 1.

Training adds an InfoNCE loss (temperature `tau`, default 0.05) between each
question and the batch's distinct subjects plus `negbatches` batches of random
nodes; its gradient flows through both reading passes into the model
(`Model::dXext`, an external gradient on the final representation). The answer
pass keeps the true facts. Tests: head gradients against finite differences, and
an external gradient equal to the CE gradient reproduces the CE backward.

```sh
./kgprep qa graph.tsv kg                      # also writes kg_nodes.tsv
./kgtrain train kg_train.tsv s2.ckpt nodes=kg_nodes.tsv mem_rdim=128 dim=256 layers=4 batch=32 steps=10000
./kgtrain eval kg_test.tsv s2.ckpt nodes=kg_nodes.tsv memory=retrieved
```

**Stage-2 result** (same data and model size as stage 1, `mem_rdim=128`,
10,000 steps, 24 minutes while sharing the GPU with another run). Index: all
18,011 nodes; test questions about 1,008 subjects never seen in training:

| | Test |
|---|---|
| retrieval top-1 (exact node) | **99.25%** |
| retrieval top-1 (same label) | 99.82% |
| retrieval top-5 | 100.00% |
| exact match, retrieved memory | **79.7%** |
| exact match, oracle memory (stage-1 setting) | 79.9% |
| exact match, no memory / shuffled facts | 0.0% / 4.2% |

Retrieval costs almost nothing end to end (79.7% vs 79.9% with the oracle). The
gap to stage 1 (90.3%) comes from the shorter training of the answer pass
(10,000 vs 20,000 steps), not from retrieval. Caveat: the subject's name
appears verbatim in the question, so retrieval is essentially name matching
through the model's state; ambiguous names and paraphrased subjects are not
tested.

`memory=retrieved` reports retrieval top-1/top-5 over the whole index (exact
node; `top1_label` also accepts a different node with the same label) and the
end-to-end exact match. The index is scored exhaustively on the host, which is
fine for tens of thousands of nodes; the full Wikidata graph needs an ANN index.

**Stage 3 (next):** pretraining on plain text, then joint training with the
graph for free-form questions, ambiguous names, combining facts and using them
in sentences. Personalized PageRank over the retrieved node's neighborhood fits
there, when questions need more than one entity.

## Chat and dialog fine-tuning

`chat` is an interactive CLI with a persistent recurrent state: everything said
stays in the model's memory until `/reset`, there is no context window.

```sh
./chat model.ckpt temp=0.7 maxlen=512     # commands: /reset /temp X /maxlen N /mode dialog|raw /quit
```

For plain text checkpoints it runs in `raw` mode: the input is fed as text and
the model continues it until a newline. For dialog checkpoints it runs in
`dialog` mode with a byte-level turn format:

```text
0x1E                 start of a conversation (docsep=30 resets the state there)
0x02 text 0x04       user turn
0x03 text 0x04       assistant turn (the model ends it by emitting 0x04)
```

`tools/dialogprep.cpp` builds such data, e.g. from the OpenAssistant oasst1 trees
(Apache-2.0; every English root-to-leaf path is one conversation, split by tree),
or from `user<TAB>assistant` pairs. `dialog=1` makes the loss cover only the
assistant text and its closing `0x04`; the user turns are context. `init=` starts
a new run from another checkpoint's weights (same architecture, fresh optimizer),
so a text model can be fine-tuned:

```sh
zcat 2023-04-12_oasst_ready.trees.jsonl.gz | ./dialogprep oasst oasst
./train oasst_train.bin chat.ckpt init=text.ckpt dialog=1 docsep=30 lr=0.0002 steps=30000
./train oasst_test.bin chat.ckpt mode=eval
./chat chat.ckpt
```

## Checkpoints and reproducibility

Format V3 stores the full configuration, FP32 master weights,
Adam moments, EMA encoder, global optimizer step, data position/epoch,
dataset fingerprint, recurrent states, the valid MLA cache including
absolute positions, and the traces when `traces=1`. Files written before
`traces` existed load unchanged. BF16 working weights are reconstructed from these.

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

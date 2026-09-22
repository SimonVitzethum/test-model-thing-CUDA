# Test-Model-Thing (TMT) — CUDA

[YouTube Video (original MLX proof of concept)](https://youtu.be/9UERVVwpNew)

This is a small byte-level language model (not an LLM), now developed as a pure
CUDA C++ project. It started as an MLX proof of concept and keeps its core ideas
while reworking the architecture, training, and infrastructure:

* Byte input/output (no tokenizer)
* Gated hierarchical recurrence with geometrically staggered half-lives
* Exact truncated backpropagation through time (TBPTT), optionally combined with exact traces across window boundaries (`traces=1`)
* Optional Mixture-of-Experts feedforward with top-k token dispatch and router balancing
* Optional MLA long-range cache (DeepSeek-style compressed KV latents, up to 128k bytes)
* Optional latent-space prediction with an EMA target encoder
* Continuous data streaming with explicit, checkpointed stream state
* Knowledge-graph fact memory from Wikidata (`mem=1`, stage 1: cross-attention over retrieved facts)

Everything lives in [`tmt-cuda/`](tmt-cuda/README.md). The original MLX and PyTorch
prototypes have been removed; their checkpoints and results are not comparable to
the CUDA model. See [CHANGES.md](CHANGES.md) for a detailed comparison with the
original TMT.

Feel free to fork the code (everything is under MIT). Issues and pull requests
are very welcome. If you have compute lying around, training larger models for
longer is highly appreciated as well, with credit.

## Results

The original proof of concept was a 4.5M parameter MLX model (`dim = 512`,
`layers = 16`) trained on `simplewiki-20260801-pages-articles.xml.bz2` from the
Wikipedia dumps. It tended to misspell characters, but could close
quotes/brackets.

Selected measurements of the CUDA version (bits per byte, lower is better):

| Run | Setup | Result |
|---|---|---|
| run2 | 420M total / 106M active (MoE), RTX 5080, enwik8 | BPC 5.11 → **1.31** in 15.6M tokens (then router collapse, fixed afterwards) |
| 8h laptop run | 34M, `aux=0.1`, RTX 5070 Laptop | 2.46B tokens, held-out BPC 3.15 → **2.50**, router balanced throughout |
| Throughput | same scale, CUDA vs. PyTorch port | **216×** (106k vs. 0.49k tok/s) |

**Caveat for the 8h run:** it was trained with a MoE backward bug (introduced
in `f052b67`, fixed since) that gave the expert weights zero gradient and cut
the gradient path through the experts. The experts stayed at their random
initialization, so this number understates what the architecture can do. run2
predates the bug.

More numbers (kernel efficiency, roofline) are in [CHANGES.md](CHANGES.md).

## Training your own model

Model weights are not provided. You can build and train your own model with
Linux, an NVIDIA GPU with BF16 support, CUDA/cuBLAS, and a C++17 compiler. No
Python is needed.

```sh
cd tmt-cuda
make          # default target: consumer Blackwell (sm_120a)
make check    # native numeric and integration tests (requires a GPU)

# other GPUs, e.g. Ada:
make clean && make ARCH='-gencode arch=compute_89,code=sm_89'
```

The training data is a raw byte file. Train, resume, evaluate, and sample:

```sh
# new run (steps=0 runs until aborted)
./train train.bin model.ckpt steps=1000 saveevery=100

# resume: config, optimizer, data position, and stream state are restored
./train train.bin model.ckpt steps=1000

# evaluate without modifying weights or writing checkpoints
./train validation.bin model.ckpt mode=eval

# MoE / MLA experiments
./train train.bin moe.ckpt experts=4 topk=2 steps=1000
./train train.bin attention.ckpt mla=1 mla_every=2 mla_cache=4096 steps=1000

# generate text (never modifies weights)
./sample model.ckpt "The history of" temp=0.7 maxlen=256

# interactive, the recurrent state persists across turns (/reset clears it)
./chat model.ckpt

# fine-tune a text model on dialogs (answer-only loss), then chat with it
./train oasst_train.bin chat.ckpt init=model.ckpt dialog=1 docsep=30 steps=30000
```

You can safely ^C training; it stops after the current window and saves a
checkpoint. Checkpoints (format V3) contain the configuration, weights, optimizer
state, data position, and stream state, so resumed runs are reproducible. The last
output line of a run is a JSON object with `ce`, `bpb`, and throughput.

All options, defaults, and details are documented in the
[tmt-cuda README](tmt-cuda/README.md).

## Architecture

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

### Changes compared to the original architecture

| Area | Original (MLX) | Now (CUDA) |
|---|---|---|
| Layer structure | all recurrent layers fed directly from the byte embedding | each layer receives the residual stream of the previous layer (hierarchical) |
| Recurrence | `state = decay·state + enc`, unbounded, single timescale (half-life ~1 byte) | gated and normalized (bounded), half-lives staggered from 2 to 512 bytes, input-dependent gate |
| Time credit | 1-step gradient + hand-built traces (embedding/decay only) | exact TBPTT window (default 128 bytes); optional exact decay/gate/embedding traces across windows |
| Feedforward | one dense linear per layer | dense or MoE top-k with token dispatch, SiLU, load-balancing + z-loss |
| Long range | effectively dozens of bytes | optional MLA cache with RoPE, up to 128k bytes |
| Objective | JEPA latent loss + CE on the same vector | next-byte CE by default; latent loss against an EMA target encoder optional |
| Stop head | MSE on line endings | optional BCE with `pos_weight`, newline as target |
| Memory | stored in checkpoints, drifts across runs | explicit stream state decoupled from the model, resets per epoch or via `maxcarry` |

See [ARCHITECTURE.md](tmt-cuda/ARCHITECTURE.md) for the full design and
[tmt-cuda/README.md](tmt-cuda/README.md) for the planned Dream-RSI learning
integration.

## Credits

The original TMT architecture and MLX proof of concept were designed between
July and August 2026 by the original author (see [LICENSE.md](LICENSE.md)).
This repository continues it as a CUDA project.

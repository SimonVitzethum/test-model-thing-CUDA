# Moonshots

Implementation plans for the six long-shot ideas. These are not ranked
against TODO.md and they do not compete with it: the items there are
expected to work, these are expected to *mostly* not, and are written down
so that the ones worth an afternoon can be tried in an afternoon.

Each section gives the concept, the concrete steps against this codebase,
the cheapest test that decides it, and the signal that says stop. None of
them is impossible. All of them are uncertain, and the point of the cheap
test is to find out which kind of uncertain, quickly.

Where a probability appears it is a guess at whether the idea delivers the
*large* factor, not whether it does anything at all. Several of these have
a high chance of delivering something small, which is called out
separately, because a 10% win from a moonshot still counts.

---

## A. Predict the trajectory instead of walking it

Training moves in a small subspace, so estimate that subspace from early
checkpoints and jump forward along it.

**Large factor: ~2%. Something small: ~40%.**

### Steps

1. **Keep a checkpoint series.** `saveevery` currently overwrites one
   path. Add `savekeep=N` to write `<path>.<step>` for the last N, or do
   it in the run script with a copy - the run script is enough for the
   first test.

2. **Dump parameters to a flat file.** A small Zig tool beside `kbench`,
   reusing the checkpoint reader that is already tested, writing one
   parameter's fp32 master as raw bytes. Keeps the format logic in one
   place and lets the maths happen in numpy.

3. **PCA over checkpoints, not over weights.** With K checkpoints of P
   parameters, build the K x P matrix, subtract the mean, and take the
   eigenvectors of the K x K Gram matrix. K is 20, so this is a 20x20
   eigenproblem - the size of the parameter vector never enters the
   decomposition. For P = 34M and K = 20 the matrix is 2.7 GB in fp32,
   which fits in the 31 GB of RAM.

4. **Fit each component in schedule time, not step time.** The cosine
   schedule sets the shape of the trajectory, so fit the component
   trajectories against the integrated learning rate. A straight line in
   step number is the wrong basis and will predict badly for a reason that
   has nothing to do with the hypothesis.

5. **Reconstruct, write a checkpoint, evaluate.** The averaged-checkpoint
   path from TODO item 9 already writes a second checkpoint from swapped
   pointers; the same route works here.

### The cheapest test

One matrix, ten checkpoints, extrapolate 10% beyond the last, evaluate.
An afternoon, and the checkpoints from the night run exist already.

### What makes it work if it works

Not one long jump - **leapfrogging**. Train 10%, extrapolate 5%, refit,
repeat. The known objection is that the subspace rotates, and a short jump
followed by a refit is exactly the shape that survives rotation. Test the
single jump first because it is cheaper, but do not conclude from it.

### The signal to stop

If the extrapolated weights evaluate *worse* than the last checkpoint at
every extrapolation distance including 2%, the components being fitted are
not the ones carrying the progress. That is one afternoon spent.

---

## B. Find the winning ticket at initialisation

A random network contains trainable subnetworks; find one cheaply and
train far fewer parameters.

**Large factor: ~1%. Something small: ~25%.**

The honest starting point: pruning-at-init methods lose to random pruning
at equal density once the comparison is fair. So do not build SNIP or
GraSP. Build the variant where the objection does not apply.

### Why the structured form is different

Unstructured sparsity gives no wall clock on a GPU - a 90% sparse matmul
runs at dense speed. So the usual lottery-ticket work trades a real cost
for a theoretical saving. Structured masks do not have that problem, and
this architecture has two natural structures to mask:

- **experts** (8 of them, the router already scores them),
- **channels**, which carry *staggered half-lives* - so a channel mask is
  a hypothesis about which time constants matter.

### Steps

1. Add a score `s_c` per channel and per expert, initialised randomly.
2. Forward uses the top `d%` by score; the rest are masked out.
3. Backward updates the scores with a straight-through estimator
   (edge-popup), while the weights stay at their initial values.
4. Compare against a dense model **of the same FLOPs** - a narrower one -
   trained normally. That is the comparison the literature usually gets
   wrong, and the one that decides it.

### What would be genuinely new

If the learned channel mask concentrates on a particular *band* of
half-lives, that is a result about the architecture regardless of whether
the lottery ticket story holds: it says which time constants this data
needs, which `half_min`/`half_max` currently guesses.

### The signal to stop

Masked-at-init loses to the equal-FLOP dense baseline across three
densities.

---

## C. Two memories with consolidation

Fast episodic storage in one presentation, slow weights that learn only
from replay of it. Complementary learning systems.

**Large factor: ~2%. Something small: ~35%.** The theoretically most
interesting item here, and the one that fits the existing architecture
best.

### Why this one fits

`mem=1` already builds a fast store: memory projections, `mem_len`,
`mem_heads`, `mem_dh`, retrieval heads through `mem_rdim`, and the KG fact
memory. The structure is there. What is missing is the second half of the
theory: the slow weights currently learn from raw bytes, not from the
contents of the fast store.

### Steps

1. **A fast store with no gradient.** A delta-rule outer-product memory
   updated once per window: `M <- lambda*M + k v^T`. The outer-product
   kernels exist for the memory path already; this one takes no gradient
   at all, which is what makes it fast.

2. **A replay buffer.** Not the data - the *episodes*: corpus offset,
   window length, and the recurrent state at entry (`extract_carry`
   already produces it). A few thousand entries is kilobytes.

3. **Consolidation as distillation from the fast store.** Each step, with
   probability `p`, take a replayed window and train the slow weights
   against what the model-with-fast-memory predicted, rather than against
   the raw byte. This is the actual claim of the theory, and it reuses
   whatever KL machinery TODO item 14 builds.

4. **Prioritised replay.** Sample episodes with priority proportional to
   `loss_slow - loss_with_fast`: replay what the slow weights have not yet
   absorbed. The two losses are already computed.

### The measurement that decides it

**Per unique byte, not per step.** The entire claim is sample efficiency,
so the x-axis has to be bytes the model has never seen. Replay that merely
repeats data will look good per step and flat per unique byte, and that
distinction is the whole experiment. If it is not measured this way the
result means nothing either direction.

### The signal to stop

Equal or worse BPB per unique byte at three replay ratios. Then replay is
data repetition wearing a hat, which is the standing objection.

---

## D. Solve instead of search

Two-layer ReLU networks have exact convex reformulations. A deep
generalisation would replace gradient descent with solving.

**Large factor as stated: ~0.5%.** The deep version is open. But there is
a restriction of it that is convex, cheap and implementable this week, and
it is worth doing on its own merits.

### D1. Solve the output head exactly (do this one)

`logits = h W_dec` under cross-entropy is **convex in `W_dec`** for fixed
`h`. No approximation, no open problem. So:

1. Every N steps, collect hidden states for a few thousand positions.
2. Solve the multinomial logistic regression for `W_dec` to near-optimality
   with a handful of Newton or L-BFGS steps.
3. Write the solution into the decoder and continue training normally.

The decoder is `dim x 256` - for `dim=512` that is 131k parameters and the
solve is small. Known to help under the name last-layer retraining. It
also removes a source of lag: the head is always chasing a representation
that has already moved.

The same applies to the MTP heads, which are additional decoders, and to
the stop head.

### D2. The recurrence is linear in the state

With the gates `a` held fixed, `s = a*s + (1-a)*x` is a **linear** map
from inputs to states. So for fixed gates, state-to-output is a convex
problem. That gives an alternating scheme: solve the readout exactly,
take gradient steps on the gates, repeat.

This is a real structural property of this architecture that a transformer
does not have, and it is the part of D worth pursuing beyond D1.

### The signal to stop

D1 is not a gamble - if exactly solving the head does not beat gradient
steps on it, that is itself a useful and surprising result about the
model. D2 stops if the alternating scheme oscillates rather than
converging, which shows up within a few hundred steps.

---

## E. Integer and binary training

Training without floating point, in 1-2 bit arithmetic with stochastic
rounding.

**Large factor: ~5%. Something small: ~70%** - and the small thing is
worth more here than the large thing is elsewhere.

### The correction that puts this higher than it was

FP8 was rated at 15% earlier on the grounds that this machine is
bandwidth-bound rather than compute-bound. That reasoning was wrong, and
in the informative direction: **quantisation reduces bandwidth**, which is
precisely the measured bottleneck. The argument that kills FLOP-saving
ideas here is the argument *for* this one.

Concretely, `mt_adam` is the largest kernel in the step at 3.7 ms, running
at 322 GB/s - the ceiling - moving 30 bytes per parameter:

```
master fp32  4 + 4 read/write
m      fp32  4 + 4
v      fp32  4 + 4
grad   fp32  4 read
work   bf16      2 write
```

### Staged, each stage independently useful

1. **Adam moments in bf16** with stochastic rounding. 30 -> 22 bytes,
   about -27% on the largest kernel. Established practice, low risk.
2. **Master weights in bf16** with stochastic rounding, which is what
   makes bf16 masters work at all. 22 -> 16 bytes. Cumulative -47%.
3. **int8 expert weights** with per-channel scales. Blackwell has int8
   tensor cores and cuBLAS has IMMA, so this is a kernel change rather
   than a research project. Halves the weight traffic in the GEMMs.
4. **Ternary weights** with an fp master, BitNet-style. Well-trodden.
5. **The moonshot: ternary weights and int8 gradients, no fp master
   anywhere**, stochastic rounding throughout.

Stages 1 and 2 pay for the work on their own. Stages 3 and 4 are ordinary
engineering with known outcomes. Only stage 5 is a gamble, and by then the
infrastructure for it is already built and measured.

### Why this project specifically can do it

Kernels are written here in Zig through LLVM to PTX, so the arithmetic is
reachable. Most groups cannot change the inside of a matmul.

### The signal to stop

Stage 5 stops if the gradient signal disappears into the rounding: the
loss curve flattens at a level well above the fp baseline and stays there
across three rounding seeds. Stages 1-4 do not need a stopping rule,
they need a benchmark.

---

## F. Take someone else's compute

**Large factor: 30-50%.** Not a moonshot - the one item here that plausibly
delivers 100x - and it is already TODO item 14, where the teacher was
measured at 0.853 BPB against this model's 1.63.

Two routes, and they compose.

### F1. Byte-level distillation (TODO item 14)

Measured and planned there. One forward per token, a trie over the
vocabulary, a byte-level target at every position.

### F2. Transplant the layers, not just the outputs

Distillation copies behaviour through a narrow channel - the output
distribution. The wider channel is the *activations*.

1. Run the teacher over a batch and keep each layer's input and output.
2. For each student recurrent layer, fit its parameters to reproduce one
   teacher layer's map on those activations. Per layer this is a
   regression, and by section D it is convex in the readout.
3. Then fine-tune end to end with F1.

The theory behind this is the linear-attention view: attention can be
approximated by a linear recurrence, and there is conversion work in that
direction. This architecture *is* a linear recurrence, so it is on the
receiving end of that literature rather than needing new theory.

### The cost that is not compute

If the model is initialised from a transplanted transformer, it no longer
answers the question "does this architecture learn efficiently by itself".
That matters for a funding argument and for the actual scientific claim.

The answer is not to avoid it but to **keep both tracks and report both**:
one line trained from scratch, one transplanted, same evaluation. The
from-scratch line stays the claim about the architecture; the transplanted
line is the one that produces a usable model.

---

## Where to start

**D1** - solving the output head - because it is convex, small, needs no
new theory and can be built this week.

**A** - because the checkpoints from the night run already exist and the
test costs an afternoon.

**E stages 1 and 2** - because they pay for themselves on the measured
bottleneck whether or not stage 5 ever happens.

**C** - because it is the only idea here with a plausible story for the
large gap, and because `mem=1` already built half of it.

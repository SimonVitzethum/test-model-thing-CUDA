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
   eigenvectors of the K x K Gram matrix - a 20x20 eigenproblem at K=20.
   The parameter count does not enter the *decomposition*, but it does
   enter the work: forming the Gram matrix is O(P*K^2) and reprojecting is
   O(P*K). Cheap, not free. At P = 34M and K = 20 the matrix is 2.7 GB in
   fp32, which fits in the 31 GB of RAM.

4. **Time axis = cumulative learning rate.** Fit the components against
   the integrated learning rate, sum of lr over steps, not against step
   number. The cosine schedule sets the shape of the trajectory, so step
   number is the wrong basis and will predict badly for a reason that has
   nothing to do with the hypothesis.

5. **Extrapolate the averaged weights, not the iterates.** TODO item 9 is
   now implemented: `wavg=0.999 wavg_every=8` writes `<path>.avg` beside
   the checkpoint. Fitting raw iterates fits noise in the valley instead of
   motion along the river.

6. **Reconstruct, write a checkpoint, evaluate.** The averaged-checkpoint
   path already writes a second checkpoint from swapped pointers; the same
   route works here.

### The experiment is a curve, not a leapfrog

An earlier draft proposed leapfrogging - train 10%, jump 5%, refit - as
the shape that survives subspace rotation. It does, but it caps the payoff
at **1.5x**, and this is a moonshot section. For 10x the jump has to cover
nine times the distance already trained.

So the experiment is the curve: **maximum lossless jump distance as a
function of how far training has progressed**. Leapfrogging is the safe
operating mode to fall back to *if* the curve looks good, not the thing
being measured.

### The metric, and why the obvious one is wrong

**Steps or FLOPs to reach a target loss after the jump, including
recovery** - not the loss immediately after landing.

After a jump the Adam moments no longer correspond to the weights, and the
MoE router can redistribute its assignments abruptly. A jump that lands at
a good loss and then takes longer to recover than it saved has bought
nothing. What to do with the moments - keep, zero, rescale - is an
experimental variable in its own right, and the `router:` line in the
training log should be compared before and after.

### The signal to stop

The extrapolated weights evaluate worse than the last checkpoint at every
distance including 2%, after recovery. Then the components being fitted
are not the ones carrying the progress. One afternoon spent.

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

### The trap that would have sunk this

The obvious priority `loss_slow - loss_with_fast`, measured on the episode
the fast store just recorded, is **wrong**, and wrong in a way that would
have produced a convincing-looking result.

On that episode the fast store's advantage is pure retrieval. The priority
is therefore highest for whatever is best memorised, and distilling
against it copies the corpus into the slow weights. That is memorisation
dressed as consolidation, and the loss curve would have looked excellent.

**The advantage has to be measured on other bytes.** Write the fast store
from window W, then evaluate on W' - the *continuation* of W, which the
store never saw:

```
A = loss_slow(W') - loss_with_fast(W')
```

This asks whether the store carries something that *transfers*, rather
than something it can look up. Priority and distillation target both
belong on W', not on W. If the store only memorised W, then A is zero on
W' and the idea is dead - cheaply.

### Phase 0, before building any of it

Measure A. One hour, and it is make-or-break in the same way the teacher's
perplexity was for distillation: if a one-shot episodic store gives no
advantage on continuation bytes, there is nothing for consolidation to
consolidate.

### Episode size, measured

`StreamState` carries per stream: the recurrent carry (layers x D x 4 =
32 KB at dim=512, 16 layers), the hybrid traces (another 64 KB), and the
MLA cache when `mla=1` (`mla_cache * mla_L * 4`, about 512 KB).

So **96 KB per episode without MLA, around 600 KB with it** - not the
kilobytes claimed in an earlier draft, and not megabytes either. A few
thousand episodes is 300 MB without MLA, which is affordable; with MLA it
is not, and the buffer has to store offsets and replay the state instead.

### A structural note

`mem=1` is a *fact* memory over supplied fact bytes, not an episodic store
written from the training stream. The encoder takes arbitrary byte
sequences, so feeding it a window's bytes is natural - but it is a change
of use, not a feature that is already there.

### The measurement that decides it

**Per unique byte, not per step**, and the evaluation must run on **held-out
data**. The per-unique-byte axis only catches the memorisation failure if
the evaluation is on bytes no part of the system has seen. Replay that
merely repeats data looks good per step and flat per unique byte.

### The signal to stop

Phase 0 returns A near zero on continuation bytes. Or, later: equal or
worse held-out BPB per unique byte at three replay ratios.

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

### D2. Withdrawn - the convexity claim was wrong

An earlier draft claimed that because the recurrence is linear in the
state for fixed gates, state-to-output is convex, and that this is a
structural advantage over a transformer. Both halves are wrong.

With the gates fixed the state is linear in the inputs and therefore in
`W_in`. But the logits are `W_dec * (... W_in ...)`, which is **bilinear**,
not jointly convex. And between the recurrence and the head sit the
experts, the norms and further layers. What is convex is the final linear
readout, exactly as in a transformer - which has the analogous property
anyway, being linear in V for fixed attention weights.

So there is no structural advantage here to claim, and a reviewer would
go straight for this sentence. D1 stands on its own; D2 does not exist.

### Implementation notes for D1

"Exactly" means iteratively to convergence - L-BFGS over a buffer of
hidden states - not a closed form. And it needs L2 regularisation:
near-deterministic bytes otherwise send the weights off without bound.

### The one-hour pretest that may end D1 before it starts

Freeze the body at a current checkpoint, solve only the head to
convergence, and measure the loss difference against the head that is
already there. It is probably tiny, because the head is small and is
trained at every step. Then D1 is finished before it is built. If the gap
is large instead, that is a real finding and a reason to build.

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

### Profile before choosing the target

The measurement below was taken with `muon=0`. `isMuonParam` returns false
whenever Muon is off, so in that configuration `mt_adam` really does own
every parameter. Turn Muon on - which TODO item 3 is about to do - and the
expert matrices, the router and the MLA and memory projections leave
`mt_adam` entirely; it keeps the embedding, the decoder, the MTP heads and
the vectors, a small fraction of a byte model. The traffic moves to Muon's
own fp32 momentum and master weights, and the same compression applies
there instead.

So profile under both settings and let that pick the target.

Also, to keep the claims straight: **stages 1 and 2 are optimizer state
compression, not integer training.** Useful, and a different experiment
from stage 5.

With `muon=0`, `mt_adam` is the largest kernel in the step at 3.7 ms,
running at 322 GB/s - the ceiling - moving 30 bytes per parameter:

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

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

**F1 (output distillation): ~60% for a factor of 3-10.
F2 (transplant the layers): ~25% for a factor of 50 or more** - measured as
compute to reach a given quality, not as steps per second.

An earlier draft put a flat "30-50% for the large factor" on the section as
a whole. That was too generous to F1 and too vague about F2, which are
different bets. It is already TODO item 14, where the teacher was measured
at 0.853 BPB against this model's 1.63.

The arithmetic on what is being offered:

```
one night here:  3.8 GB x 34M params x 6   ~ 7.8e17 FLOPs
Qwen3-1.7B:      36T tokens x 1.7G x 6     ~ 3.7e23 FLOPs
                                             ~470,000 nights
```

Half a million nights of this GPU sit in a file on this disk. Almost none
of it can arrive, and the limit is not the teacher but the student: 34M
parameters cannot hold what 1.7G learned. The bound is capacity.

Which is why F2 is the interesting half. F1 copies behaviour through the
narrow channel of the output distribution and buys sample efficiency. F2
starts the weights near a working solution and skips most of the training
rather than speeding it up. At 34M parameters 0.85 BPB from scratch is
probably unreachable at *any* amount of data or time, so F2 is not "100x
faster" - it makes a target reachable that otherwise is not.

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

## G. Retrieval over your own corpus

Not someone else's model - your own training data, made available
non-parametrically instead of being pressed into the weights.

**A factor of 10 or more: ~45%. A factor of 100: ~15%.** The only idea here
with a published double-digit result that takes over no foreign knowledge.

### Why the factor could be large, and why here in particular

Byte prediction on Wikipedia mixes two problems that have little to do with
each other:

1. **Linguistic structure** - syntax, morphology, discourse. Highly
   compressible, learnable from little data.
2. **Factual and verbatim content** - names, dates, infoboxes, templates,
   thousands of near-identical phrasings. Incompressible, needs enormous
   data to memorise, and is exactly what does not fit in 34M parameters.

Gradient descent spends most of its capacity and most of its steps on (2).
Retrieval makes (2) free, and leaves the weights with (1) plus "how do I
copy out of the store" - and (1) is what this architecture should be judged
on anyway.

This also explains the measurement at the top of TODO.md from a second
direction. 24 epochs coming out worse than 14 is capacity being burned on
(2). The data limit is half a shortage of data and half a shortage of
*somewhere to put content* - and the tool for that is an index, not a
weight matrix.

### The evidence

RETRO reached the performance of models with **25x more parameters** on the
Pile. kNN-LM gets large perplexity drops with *no* additional training at
all, purely by looking things up in the training corpus.

With one caveat that has to travel with the number: part of RETRO's
reported gain was later attributed to **overlap between the retrieval
database and the test set**. That is the same leakage trap as below, at
corpus scale, and it means the headline factor should be treated as an
upper bound rather than a result.

That is 25x, not 100x. Stacked with C it might go further, but stacking
rarely multiplies cleanly, and this belongs on the page as a
double-digit-factor idea rather than a hundred-fold one.

### Why this project is unusually well placed to build it

The machinery exists and was measured yesterday. `mem=1` with retrieval
heads (`mem_rdim`) is built, and `memprobe` showed the store is
**content-specific about the continuation**: +0.99 nats against unrelated
text at 14 standard errors, on bytes it never saw.

That measurement is the precondition. It says that *if* the store holds the
right context, it helps. Retrieval is the mechanism that puts the right
context there instead of whatever happened to precede the window.

What is missing is an index over the corpus rather than a fixed fact list.
At byte level something simple suffices: a hashed n-gram index or a suffix
array over the last m bytes, returning the k most similar places in the
corpus, whose continuations go into the existing memory slots.

### The trap, and it is the same one as in C

**Exclude the neighbourhood of the current window from retrieval.**
Otherwise the model looks the answer up instead of predicting it, the loss
curve looks superb, and it means nothing. Same class of mistake as the
consolidation priority in C.

At evaluation: retrieve over the **training corpus only**, never over the
held-out set.

### The honest price

Every step reads k neighbours on top of everything else, and the step is
bandwidth-bound at 82% of the ceiling - the one resource with nothing to
spare.

And the claim changes shape. "This model reaches X BPB" becomes "this model
plus an index over its training corpus reaches X BPB". Still a result from
its own resources, since the index holds nothing that was not in the data,
but it is a different sentence and both numbers should be reported.

---

## H. Give every timescale its own target

The one idea here aimed at the sample-efficiency gap itself, rather than at
routing around it.

**A factor of 10 or more: ~4%. A factor of 100: ~1%.**

Revised down from 25% and 8%, for a reason that should have been in the
first draft: this is not a new idea. Latent targets at multiple horizons
are **Contrastive Predictive Coding** (van den Oord et al. 2018) in
substance. CPC is well studied, it works, and on language it has never
come near a factor of ten. An idea with a decade of prior art and no large
result on this modality does not get a 25% prior.

What survives the revision is that CPC was never applied along the axis
this architecture already has - one horizon per half-life band, with the
band structure chosen by the model rather than by hand. That is the part
that is actually new here, and it is worth a cheap test. It is not worth a
moonshot-sized expectation.

### The gap, stated honestly

A child has strong language competence after something like 10^7 to 10^8
words. A language model needs 10^12 to 10^13 tokens. That is four or five
orders of magnitude, and it is the only evidence anyone has that a factor
this large exists at all.

The comparison is also contested, and it should be. Children get
multimodal grounding, interaction, the ability to ask, and a prior that
took evolution a very long time to find. Nobody knows how the gap divides
between the learning rule and everything else. This section bets on one
identifiable piece of it, not on the whole thing.

### The diagnosis, and it is specific to this architecture

The architecture is multi-timescale. The half-lives are staggered
geometrically from `half_min=2` to `half_max=65536`, which is the whole
point of it - different channels are meant to hold information over
different spans.

The objective is not. It is next-byte prediction, one horizon, and that is
all of it.

So a channel with a half-life of 65536 bytes learns only through whatever
its contribution to the *next byte* happens to be. That signal reaches it
attenuated through the entire stack, and TODO item 8 already measured that
the span it reaches is 512 times the span the window can inform. (That is
a ratio of spans, not a measured gradient fraction - see item 8.)

The architecture says "these channels operate at different scales". The
loss says "all of you, predict the next byte". The mismatch is structural
and it is free to see, which is why this is worth a section.

### The proposal

Give each band of half-lives a prediction target at its own horizon.

A channel with half-life `h` predicts something about the next `h` bytes,
not the next one. Fast channels keep next-byte prediction; slow channels
get targets at their own scale.

**Predicting bytes at those horizons is hopeless** - byte 65536 ahead is
mostly irreducible noise, and forcing the model to predict it wastes
capacity on exactly what cannot be predicted. So the targets have to be
**latent**: something about the pooled content of the next `h` bytes.

Predicting a representation discards the unpredictable surface detail and
keeps the part that is actually learnable, at the scale the channel is
built for.

### Make it contrastive, not regressive

The first draft proposed regressing onto an EMA copy of the model with a
stop-gradient, JEPA-style, and then listed collapse as the main risk. CPC
already solved that problem and the solution should be adopted rather than
rediscovered.

**Use InfoNCE.** The prediction has to distinguish the *real* future of
this position from other futures drawn from the batch. A constant
representation then scores no better than chance, so collapse stops being
a failure mode and becomes impossible by construction rather than by trick.

The alternative is variance and covariance terms in the VICReg style,
which keep the regression formulation and push the representation away
from constants explicitly. Either is defensible; InfoNCE is the one with
the track record at exactly this task shape.

Negatives come free here: the batch is 64 streams, so other streams at the
same offset are already the right kind of negative - same position in the
schedule, different content.

### How it relates to what is already here

- **MTP** (`mtp`, `mtp_weight`) is the special case at horizons 1 to 7,
  already implemented and still unevaluated.
- **Item 11** (latent patch targets) is the special case at patch scale.
- This generalises both along the architecture's own axis: instead of
  picking a horizon or two by hand, every band gets the horizon its
  half-life already implies.
- It attacks the 512x truncation from the opposite side to **item 8**.
  RTRL fixes the *gradient path* so the signal can travel back; this fixes
  the *objective* so there is a strong local signal that does not need to
  travel. They are complementary, and either one alone is a fair test of
  the diagnosis.

### Phase 0, and the obvious version of it is confounded

Before building any of it, measure whether the slow channels are actually
starved.

The obvious test - plot the gradient norm of `decay_c` against the
channel's half-life - **would confirm the diagnosis whether or not it is
true**, and that is worth spelling out because it is an easy trap.

The decay is parameterised, `a = sigmoid(decay + ...)`, so the derivative
of the effective decay with respect to its parameter is systematically
small for channels near `a = 1`. A slow channel therefore shows a small
gradient on `decay_c` *by construction*, fully used or not. A falloff with
`h` is the expected output of the parameterisation, not evidence.

Two tests that are not confounded:

1. **Gradient on the channel's input weights, normalised by their
   magnitude.** A relative update size per channel, which the
   parameterisation of the decay does not distort.

2. **Utility against signal, side by side.** Ablate one half-life band at
   a time from a trained checkpoint and measure the loss increase - that
   is the band's *utility*. Put it next to the band's gradient. **A band
   that contributes a lot and receives little is the actual confirmation**,
   and neither number alone says anything.

The second is the better test and it costs a checkpoint and a handful of
evaluation runs. It is also the same measurement section B wants for its
channel masks, so it pays for two sections at once.

`gradcheck` already groups parameters and reports per-group statistics, so
part of the machinery exists.

### What would make it fail

**Trivial targets**: pooled representations over long spans may be nearly
constant across a corpus, in which case predicting them teaches nothing
even under InfoNCE - the negatives would be indistinguishable from the
positive for reasons that have nothing to do with the model. Measure the
spread of the targets across the batch before trusting a loss that falls.

**Collapse** is handled by the contrastive formulation above and is no
longer the main risk. It remains the main risk for item 11, which is still
written as a regression.

**The prior art**: CPC has been tried on speech, vision and language for
seven years. If per-band horizons were the missing piece, it is not
obvious why nobody found it. The honest position is that the band
structure is a genuinely untried variation on a well-tried idea, and that
untried variations on well-tried ideas usually fail for the same reasons
the original stalled.

---

## I. Thinking steps inside the recurrence

The only idea on this page that explicitly trades FLOPs for sample
efficiency - which is the right direction when data is the limit and
compute is not.

**A factor of 10 or more: ~12%. Something worthwhile: ~40%.**

### The idea

A child processes a sentence far more deeply than one forward pass.
Quiet-STaR and Reinforcement Pretraining let a model think internally
before predicting, and reward the thinking that improves the prediction.

For a recurrent model this is unusually natural, and it needs no new
machinery: **run the recurrence a few extra steps with no new input before
a difficult byte.** The state update `s = a*s + (1-a)*x` is already
defined for any number of iterations; feeding it `x = 0`, or the previous
output, is a well-defined extra step. A small gate decides when to spend
them.

### Why it fits here specifically

Every other idea on this page is constrained by bandwidth. This one is
constrained by *arithmetic*, and arithmetic is the resource this machine
has spare: the step runs at 82% of the streaming ceiling but only **15% of
the matmul ceiling**. Extra recurrence steps are FLOPs on state that is
already resident. They are close to free in the currency that is short.

That is a genuine structural match, and it is the reason this outranks
several better-evidenced ideas here.

### The gate is the whole problem

Deciding *when* to think is the hard part, and the obvious trigger is
circular: spend steps where the loss is high, but the loss is only known
after predicting. Workable versions:

- **Predictive entropy** of the output distribution before committing -
  available, cheap, and does not need the answer.
- **A learned gate** trained by the improvement it produces, which is the
  Quiet-STaR construction and needs a reward signal.
- **Fixed budget, uniformly spent**, as a baseline. Worth running first:
  if uniform extra steps help, the gating question becomes an
  optimisation. If they do nothing, the gate cannot save it.

Run the uniform baseline before building any gate.

### Evidence

Real for token models at scale. Thin for pretraining from scratch, and
thin specifically for the case where the extra computation is recurrence
steps rather than generated tokens.

### The signal to stop

Uniform extra steps at three budgets do not beat the same wall clock spent
on more data.

---

## J. Meta-learn the memory update rule

The moonshot on this page, and the only one aimed at the part of the gap
that comes from prior rather than from learning.

**A factor of 10 or more: ~5%. A factor of 100: ~2%.**

### The idea

Part of what a child brings is not learned in a lifetime - it is a prior
that took evolution a very long time to find. That part cannot be learned
from the corpus, but it can be learned *across tasks*.

The concrete target here is already hand-picked: **the fast memory's
update rule.** Section C proposes a delta rule, `M <- lambda*M + k v^T`,
because that is the obvious choice. Nothing says it is the right one.

Meta-learning asks: what update rule makes this architecture learn fastest
across many small tasks? Learn the rule, then use it. That is evolution's
role, played out on a budget of hours instead of aeons.

### Why the memory rule and not the initialisation

Meta-learning an initialisation (MAML and descendants) needs the inner
loop to be differentiated through, which for a full pretraining run is
hopeless. The memory rule is different: it acts **once per window, without
gradient**, so the inner loop is one step deep. That is what makes this
tractable at all, and it is why the memory rule is the right handle rather
than the weights.

### The cheap version of the same idea

**Item 12, formal-language pre-pretraining.** A few thousand steps on
procedurally generated sequences before real data is the same bet - build
a prior cheaply instead of paying for it in corpus - without any of the
meta-learning machinery. It is rated at 30% and costs an afternoon.

Do that first. If a hand-made prior from bracket languages and cellular
automata measurably helps, a learned one is worth the effort. If it does
nothing, meta-learning a rule is unlikely to rescue the premise.

### The signal to stop

Item 12 shows no effect, or the meta-learned rule fails to beat the delta
rule on held-out tasks it was not meta-trained on - which is the failure
mode learned optimisers have, and this is a learned optimiser wearing a
different hat.

---

## Where to start

Revised after review, and the order changed.

**Item 20 in TODO.md - rephrased data.** Best-evidenced thing on either
page, roughly 3x reported, and it barely touches the trainer. A tenth of
enwik9 is one night of generation.

**H, but with the contrastive objective and the unconfounded phase 0.**
The rating came down a long way once CPC turned out to be the precursor,
but the phase 0 measurement - utility per half-life band against gradient
per band - is worth doing regardless, because section B wants the same
number and it says where the band boundaries belong.

**I - thinking steps.** Architecturally cheap, and the only idea that
spends the resource this machine has spare. Run the uniform baseline
before building a gate.

**D1** - solving the output head - still worth its one-hour pretest: freeze
the body, solve the head, see whether the gap is anything at all.

**A** - the checkpoints exist and the test costs an afternoon.

**E stages 1 and 2** - done, measured at +4.3%, no longer a bet.

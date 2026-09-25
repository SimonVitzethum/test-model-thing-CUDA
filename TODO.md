# TODO

Ranked by what the measurements say, not by how interesting the idea is.

The number in brackets is my estimate that the item actually improves the
held-out BPB (or, for the throughput items, the step rate). They are
estimates, not measurements.

## The finding that orders this list

```
95 MB training data, ~40M parameters  ->  2.4 bytes per parameter
14 epochs -> 1.63 BPB
24 epochs -> 1.82 BPB   (worse)
```

The model memorises enwik8. Anything that only raises steps per second
makes it overfit sooner. Items 1 and 2 address the actual bottleneck;
everything else is downstream of them.

## 1. More data [95%]

`~/tmt-data/enwik8/train.bin` is 95 MB. For 40M parameters that is short by
at least an order of magnitude. enwik9 is the same source at 1 GB and goes
through the existing data tool unchanged.

This is the only item that will move 1.63 BPB substantially, and it is a
download rather than a piece of research.

## 2. Regularisation [70%]

24 epochs coming out worse than 14 is the textbook signature of
overfitting, and nothing in the model currently fights it.

- dropout on the expert outputs
- higher weight decay
- label smoothing on the byte cross-entropy

A day of work, measurable the same night against the held-out set.

## 3. Evaluate Muon [60%]

Implemented (`muon=1`, `muon_lr`), never measured against AdamW. The
reported gains on small models are 1.3-2x in sample efficiency, which acts
directly on the data problem. This is an A/B run, not a build.

## 4. Evaluate MTP [50%]

Same situation: `mtp=1`, `mtp_weight` are in, unmeasured. Multi-token
prediction acts as a regulariser and usually helps exactly in the
data-poor regime, so it belongs next to item 2.

## 5. Chunked parallel scan [45%]

`state_pass` + `state_bwd` are 1.17 ms per step at 89-100% of the
bandwidth ceiling, so they are already efficient; the gain would come from
parallelism, not from moving fewer bytes. High implementation cost and it
changes the numerics, so ktest has to be extended before it lands.

### 5b. Multigrid over the half-lives [40%]

A better-fitting variant of the same goal. The half-lives are staggered
*geometrically*, `half_min=2` to `half_max=65536`, which is literally a
grid hierarchy: the fast channels are the fine grid, the slow ones the
coarse grid. Multigrid is built for exactly that structure and converges
in a number of sweeps independent of the spread, where a flat iteration
needs a number proportional to it.

Concretely: solve the fast channels over short spans, restrict to the slow
channels over long spans, prolong back. The property that makes the
recurrence hard to parallelise - the 32768-fold spread in time constants -
is the property multigrid exploits.

This came out of the Equilibrium Propagation discussion but does not
depend on it; it is a way to parallelise the scan over `T`, and a more
specific one than the generic chunked scan above.

## 6. Model growth [35%]

Start narrow, widen during training. Weak track record in the literature
and the saving is small at 40M parameters.

## 7. Patching at equal FLOPs [40%]

Deferred during the night run and still open. This is the *measurement*
that decides whether patching earns its place, not a new idea - it should
happen before patching goes into any long run.

## 8. RTRL for the recurrent part [40%]

Real-Time Recurrent Learning is normally dismissed as O(n^2) in the state
size. That holds for a dense recurrence. This one is elementwise:

```
s = a*s + (1-a)*x        ds_t/ds_{t-1} = diag(a_t)
```

The Jacobian is diagonal, so for every per-channel parameter - `decay`,
the gate bias, the trace terms - RTRL is exact and costs O(parameters)
rather than O(parameters^2). Carried forward with the state instead of
backward through it, that means:

- no BPTT and no `state_bwd`,
- no activations stored across time,
- and no window: unbounded context without TBPTT.

The last point is what makes it worth trying here specifically, and there
is a number behind it. The night run ran `half_max=65536` at `seqlen=128`:

```
slowest channel:   half-life 65536 bytes
gradient window:              128 bytes
                   ->  the channel reaches 512x further than the
                      gradient that shapes it
```

Read that as a ratio of spans, which is what it is, and not as a measured
fraction of gradient. Nobody has measured how much signal actually arrives;
512 is the mismatch between what the channel is built to hold and what the
window can inform. Establishing the real number is part of the work, not a
premise of it. `traces=1`
carries the *state* across windows but not the gradient - the forward pass
remembers, the backward pass does not. That is a measurable defect in the
current configuration, not a suspected one.

It may also explain the night run's split result: `traces=1` won clearly
with frequent resets (2.1907 vs 2.3643 at 128 bytes) and lost narrowly at
full context (1.6367 vs 1.6292). That is what one expects when the long
channels carry state they were never trained to fill.

The exact trace for a per-channel parameter:

```
d s_t / d a = s_{t-1} - x_t + a * d s_{t-1} / d a
```

Elementwise, carried forward with the state. Do this for the **decay
parameters only** - per-channel scalars, where the trace costs nothing
next to the GEMM. Not for the matrices, where the trace costs about a
forward GEMM per byte.

Two further caveats: across stacked recurrent layers it is an
approximation rather than exact (reportedly a good one), and it interacts
with patching, because the trace then runs over patches instead of bytes.

The honest limit: for the matrix weights `ds/dW` is still a full tensor,
so RTRL stays too expensive there. What is realistic is a hybrid - RTRL
for the recurrent channels, backprop for the matmuls. Roughly 40% that it
runs correctly, 25% that it beats TBPTT once it does.

## More per FLOP

A list of efficiency ideas, filtered against what this machine actually
measures. The filter matters: the step is **host-bound and
bandwidth-bound**, running at 82% of the streaming ceiling but only 15% of
the matmul ceiling. Anything whose payoff is "fewer FLOPs" converts to
close to zero wall clock here, and if it adds kernel launches it makes
things worse, because 361 launches per step is where the 8.3 ms fixed cost
comes from.

Sorted by what survives that filter.

### 9. Weight averaging over training, LAWA [55%]

Cheapest item on this page. Costs no FLOPs in the training loop, the
averaged weights are usually measurably ahead of the current ones, and it
lets a run stop earlier.

Note what already exists and what does not: `ema`/`ematau` averages the
**embedding table only**, as the target for the latent loss. Averaging all
parameters for evaluation is not implemented.

### 10. Hyper-connections [45%]

Several residual streams instead of one, with learned mixing weights
between layers. Reported convergence gains at low extra cost, including a
stabilised variant for large MoE. Small, contained change, and the
evidence is better than for most things on this list.

### 11. Latent patch targets, JEPA-style [30%]

Predict the *representation* of the next patch - against an EMA encoder or
the model's own patch embedding with a stop-gradient - in addition to the
bytes. This gives the upper layers a dense target at patch level instead
of the thinned byte signal seeping up from below. The large sibling of
MTP, and it should be designed together with the patching rather than
after it.

Risk: representation collapse, and the anti-collapse tricks are exactly
the part that is young and unproven for language models.

### 12. Formal-language pre-pretraining [30%]

A few thousand steps on procedurally generated sequences - bracket
languages, cellular automata - before real data. Very cheap to test.

Interesting here specifically: the long half-life channels see almost no
long-range signal early in natural text, and this feeds them long-range
dependencies deliberately. Pairs with item 13.

### 13. Context-length curriculum [25%]

Start with short windows, lengthen. Well established and nearly free, and
the machinery exists already (`maxcarry`, the traces path).

The interaction to watch: long half-lives see no signal at short context,
so either grow `half_max` along with the window or accept that those
channels start late.

Downgraded from 40%. The BabyLM challenge exists to study exactly this gap
and found that child-like curricula - simple material first - barely
helped, while distillation and architecture changes did. Context-length
curriculum is not quite the same thing as a difficulty curriculum, but it
is close enough that the null result should count against it.

### 14. Distillation from a local LLM [55%]

Second only to item 1, and it attacks the same bottleneck: a one-hot byte
carries at most 8 bits, a soft distribution over 256 bytes carries far
more. Distillation multiplies the information per byte, which is exactly
what a data-limited model is short of. A 40M byte-level student from an 8B
teacher is the textbook case for it.

A teacher is already on this machine: `qwen3:8b` at 5.2 GB fits in the
8 GB of VRAM (`qwen3:1.7b` at 1.4 GB is the fast alternative).

**Measured, not estimated.** Qwen3-1.7B scored against the same held-out
set the model is measured on (`enwik8/held.bin`, 2 MB, 1024-token windows,
bf16):

```
Qwen3-1.7B    0.853 BPB      3.08 bits per token, 3.6 bytes per token
this model    1.63  BPB
```

The teacher is nearly twice as good, and that is the *small* one. The
chunk boundaries handicap it slightly, so the true figure is a little
better still.

**Cost, once, not per run.** Scoring runs at 6337 tokens/s on this GPU:

| corpus | teacher | wall clock |
|---|---|---|
| enwik8 (100 MB) | Qwen3-1.7B | **1.2 h** |
| enwik9 (1 GB) | Qwen3-1.7B | **11.8 h**, one night |
| enwik8 | Qwen3-8B | ~5 h, estimated from the size ratio |

Top-8 targets over enwik9 come to roughly 13 GB on disk. Every run after
that reads them for free.

The measurement script is `teacher_bpb.py`; it takes the model name, the
corpus and how many bytes to score.

**The tokenisation is solvable, and more cheaply than it first looks.**
The teacher is BPE, the student is bytes. At a token boundary the teacher
gives `P(t)` over tokens, so

```
P(next byte = b) = sum of P(t) over all t beginning with b
```

and for bytes *inside* a token no second forward pass is needed: restrict
the token set to those consistent with the bytes emitted so far,
renormalise, group by next byte. A trie over the vocabulary, one teacher
forward per *token* rather than per byte, and a byte-level target at every
position from roughly 4x less teacher work.

The one approximation: if the real text falls outside the top-k the
consistent set goes empty, and the target falls back to the hard byte.

**The question that had to be answered first** - whether a model trained
on code and mathematics is any good on 2006 Wikipedia markup - is answered
above, and the answer is yes.

**Caveats.** The student can pick up abilities it could never have learned
from the data, which is the point, but it makes comparisons against the
existing runs a comparison between two different things. And it is several
days of work: trie, marginalisation, target format, loader, a KL term
beside the cross-entropy.

Self-distillation from an earlier checkpoint of this model remains
possible and pairs with item 6, but it is the weak version: the teacher is
capped at what the student already reached.

### 15. Selective loss on learnable bytes, Rho-1-style [20%]

Loss and backward only on positions whose loss is well above a small
reference model's. Strong reported efficiency gains, and it acts on
sample efficiency, which is the actual bottleneck here.

The doubt is specific to bytes: trivial bytes - UTF-8 continuation bytes,
the space after a full stop, word endings - already produce almost no
gradient, so cutting them saves little of what is actually being spent.

### 16. Mixture-of-Depths [15%]

A router decides per position whether to run a block or skip via the
residual. MoE along depth instead of width. For the recurrence, skipped
positions only need the state advanced by decay, which is closed-form here
(`a^dt`) - so the architecture fits well.

Downgraded from what its evidence would suggest, for one reason: its
payoff is **fewer FLOPs**, and FLOPs are not what this machine is short
of. At 15% of the matmul ceiling, halving the FLOPs buys almost no wall
clock, while the extra per-layer router adds launches to the one thing
that is measurably the bottleneck. It would become interesting after the
throughput items below are done, or on a machine that is compute-bound.

### 17. Online data selection [15%]

Weight batches by learning value, roughly model loss minus reference loss.
Item 15 at document level instead of position level. Evidence is good for
multimodal models and mixed for language.

## The optimiser itself

### 18. Modular norm for the embedding and the heads [50%]

Best effort-to-payoff ratio on this page.

Optimisers can be read as steepest descent under a chosen norm: Adam is
roughly sign descent, so the l-infinity norm; Muon is the spectral norm.
The continuation picks the norm per layer *type* - spectral for hidden
matrices, a row norm for embeddings - and derives learning rates that
transfer across width and depth. It may make item mup unnecessary rather
than sitting next to it.

Half of this is already in place, and so is the inconsistency:
`buildOptTable` runs the matrices under Muon (flag bit 4) and the
embedding and heads under AdamW (bit 1). The other half is another flag,
not a rebuild, because the per-parameter `lrmul` machinery already exists.

### 19. Lookahead / outer optimiser [25%]

Every k steps, a Nesterov step on the difference of the weights. DiLoCo
with a single worker reduces to this. Evidence is mixed and the effect is
small, but it is one extra copy of the weights and a few lines in
`optimizerStep` - an hour of work for a fair shot.

### Edge of stability - diagnostics, not an experiment

Sharpness settles around 2/learning-rate. Worth knowing when Muon
learning rates get turned up and instabilities appear. The 2/lr result is
well established for plain gradient descent and blurrier for adaptive and
orthogonalised methods, so treat it as a diagnostic rather than a target.

## Data, beyond just having more of it

### 20. Rephrased data [50%]

The best-evidenced item added since the first draft. WRAP (Maini et al.
2024) had a teacher model rewrite web text in several styles and reported
roughly **3x faster pretraining**. It composes with item 14 from the other
side: there the teacher supplies logits, here it supplies data.

It also connects to the measurement at the top of this page. Muennighoff
et al. (2023) found that repeating data is nearly as good as fresh data up
to about **four epochs**, after which returns decay sharply. The night run
went fourteen and then twenty-four, so both of those sit well past the
point where repetition stops paying - which is a second, independent
account of why 24 came out worse than 14.

**The practical catch, which is specific to this machine.** Generation is
far slower than scoring. `teacher_bpb.py` scores at 6337 tokens/s;
generation on the same model runs at perhaps 500-1500 tokens/s even
batched. enwik9 is about 245M tokens, so rewriting the whole corpus is
30-70 hours. Rewriting **a tenth of it is one night**, and a tenth
rephrased into three styles is already 270 MB of new text over the same
content.

So the plan is a slice, not the corpus, and the experiment is whether a
mixture of original and rephrased beats the same wall clock on original
alone.

### 21. Active selection by learning progress [30%]

Not by loss - selecting by loss selects noise, which is the standing
objection to item 15. By **learning progress**: how much the loss on
similar material has *fallen* over the recent past.

A child seeks out material at the edge of what it can do, which is neither
what it knows nor what is incomprehensible. The measurable version of that
edge is the derivative, not the level.

Related to items 13, 15 and 17, and adaptive where those are fixed.
Evidence is good in reinforcement learning and on small models, open for
pretraining.

## How to measure any of this

**Do this before any of the items above.** The bottleneck on experiments
here is not ideas and not compute, it is that one comparison currently
costs one night per variant - two variants a night on a 55 W laptop GPU,
against seventeen open items.

**Branched cooldowns.** The loss landscape behaves like a river valley: a
high learning rate makes progress along the river, the decay sinks into
the valley. This is why WSD schedules work, and it gives a cheap
experimental method - run one base run, then branch *short cooldown
phases* off its checkpoints, one per variant, and compare those. Ten or
twenty comparisons a night instead of two.

This is not a training method, it is measurement infrastructure, and it
makes everything else on this page an order of magnitude cheaper to
decide.

Beyond that, not at equal step counts. Two axes, answering different
questions:

- **equal FLOPs** for whether a method is more efficient,
- **equal data** for whether it helps the actual bottleneck, since the
  model is data-limited and most of these gains shrink with scale.

The night-run curves are the baseline for both.

## Throughput, separately

These have a high chance of working (>90%) and close to zero effect on
quality. They become worth doing once item 1 is done, because then the
data is no longer the limit.

- CUDA graphs: would remove most of the measured 8.3 ms fixed cost per step
- kernel fusion: 361 launches per step is the reason that cost exists
- ~~Adam state in bf16~~ **done, +4.3%**
- ~~low-precision expert matmul~~ **done: e4m3 +3.6% at dim 512, +5.0% at
  dim 1024; NVFP4 -4.7% and kept only for larger shapes. NVFP4.md.**
- device-side expert counts, then grouped GEMM without padding
- fuse the gather into the matmul: gather, combine, combine_bwd and
  scatter_add are 38% of device time and do nothing but move rows so cuBLAS
  sees contiguous operands. CUTLASS can gather A and scatter D. This is the
  largest single item left in the profile.
- the backward dispatch is still bf16 and re-gathers its own copy, so about
  half the dispatch traffic is untouched by the e4m3 work

**What the quantisation work established, beyond its own numbers.** The step
runs at 82% of the streaming ceiling and 22% of the matmul ceiling, and the
expert forward matmul is only 266 ms of a 3160 ms step. So the whole
compute-side family is worth single digits, and quantisation only pays at
all when the encoding folds into a kernel that had to run anyway - NVFP4's
separate block-scale pass costs more than the matmul it accelerates. That is
the measured argument for spending the remaining effort on the data limit
instead, which is what moonshots G and F attack.

### Batch ramp-up - after the above, not before [35%]

Critical batch size grows mainly with the amount of data, less with model
size, so starting small and ramping up wins per FLOP. On this machine it
currently loses per second, because the 8.3 ms fixed cost per step does
not shrink with the batch:

| | throughput | fixed-cost share |
|---|---|---|
| batch=16 | 87 653 B/s | **35%** |
| batch=64 | 119 265 B/s | 12% |

A ramp starts in the worst throughput regime and eats its own gain. Once
CUDA graphs have removed the fixed cost, the FLOP argument survives into
wall clock and this is worth doing.

## Long shots

Six ideas with a plausible route to a large factor are planned out in
MOONSHOTS.md, with a cheap decisive test for each. They are not ranked
against this list and do not compete with it: the items here are expected
to work, those are expected to mostly not.

Three of them are worth starting regardless of the long shot, because
their first stage pays on its own: solving the output head exactly (it is
convex), Adam moments and master weights in bf16 with stochastic rounding
(it takes 30 bytes per parameter down to 16 on the largest kernel in the
step), and the trajectory extrapolation test, which costs an afternoon
against checkpoints that already exist.

## Considered and rejected

Written down so they do not get reopened in six months.

**Equilibrium Propagation.** The state update is exactly one gradient step
on `E(s) = 1/2 (1-a) ||s - x||^2`, so the energy structure EqProp wants is
there. But EqProp needs relaxation *to the fixed point*, and that fixed
point is `s* = x` - it carries no memory at all. The whole memory of this
model is the distance from equilibrium, which is what the staggered
half-lives control. Relaxing fully sets every half-life to zero.

On top of that: the energy is ill-conditioned by construction (curvature
`ln2/h`, so a spread of 32768 between `half_min` and `half_max`), the
router's top-k has no energy so all 8 experts would have to run instead of
2, and the gradient is a cancelling difference divided by a small beta,
which bf16 cannot represent. Break-even needs K <= 3 relaxation steps;
literature sits at 20-100 on well-conditioned toy problems.

The useful residue is the multigrid idea in 5b, which does not depend on
EqProp.

**Second-order methods** (Shampoo, SOAP, K-FAC). The reported gain over
Muon is narrow and contested, and the cost lands exactly on the measured
bottleneck: `mt_adam` is already the largest single kernel at 3.7 ms and
runs at 322 GB/s, i.e. at the bandwidth ceiling, moving 30 bytes per
parameter. Preconditioner state makes the worst kernel worse.

**Backprop alternatives** (forward gradients, zeroth-order/MeZO, evolution
strategies, Forward-Forward, predictive coding, target propagation,
synthetic gradients). Forward-mode variance grows with the parameter
count; zeroth-order works for fine-tuning, not pretraining; the local
rules buy decoupling of layers, which helps parallelism and not efficiency
per FLOP - and predictive coding is provably equivalent to backprop in the
limit anyway. None of it helps on a single GPU where backprop already runs
at 82% of the bandwidth ceiling. RTRL in item 8 is the one member of this
family that passes, and it passes for an architecture-specific reason.

**Learned optimisers** (VeLO and relatives). Impressive on their training
distribution, poor at generalising to new architectures. A hand-built
model is the worst case for them.

**Grokfast** (amplifying slow gradient components). Unproven for
pretraining, needs an extra filtered gradient state - doubling traffic in
the optimiser that is already bandwidth-bound - and interacts badly with
`accum=4`, since the filter would run over accumulated rather than true
step gradients.

## Machine configuration (done, keep in mind)

Measured on the RTX 5070 Laptop, same run each time:

| | B/s |
|---|---|
| `performance` | 85 067 |
| `max-power` | 112 695 |
| `custom` + `gpu_nv_ac_offset=80` | 123 513 |
| + `nvidia-powerd` (Dynamic Boost) | 131 220 |

Open: `gpu_nv_ctgp` and `gpu_nv_ppab` were zeroed by a bad probe script and
need a reboot to come back from firmware. Until then Dynamic Boost has
almost no budget to hand out and the GPU sits at 55 W of 115 W.

## Measured and closed, 2026-09-25

### 7. Patching - negative at equal wall clock

Four configurations, all worse than the byte baseline on held-out bytes at
matched wall clock. The patched runs were given 10% *more* time.

```
baseline, 16 byte layers, 6000 steps, 275 s      2.0525
separator patches, 2/12/2, 10700 steps, 305 s    2.5465
separator patches, 4/8/4,  8200 steps, 299 s     2.8844
fixed P=4, 4/8/4, 7600 steps, 249 s              2.4565
```

Throughput is real - 1.6x to 2.5x more bytes per second - and the quality
cost is larger. The estimate of 3-4x written into the moonshot discussion
was wrong: it counted the FLOPs saved and not what they were buying. These
are short runs and a hierarchical stack may need longer to pay off, but the
burden of proof is on patching and at this budget it fails it.

### The entropy patcher, and what actually decides a boundary rule

`patch=-2` cuts where the conditional entropy of the next byte given the
previous two crosses a threshold, calibrated to the separator rule's own
boundary density. At equal wall clock, held out:

```
plain byte baseline                       2.0525
fixed stride P=4, 4/8/4                   2.4565
separator rule, 2/12/2                    2.5465
entropy, cut before hard bytes            4.4133
entropy, cut before easy bytes            4.4124
```

The rules are genuinely different - mean patch length 5.12, 5.16 and 4.71,
Tp of 38, 40 and 57 - so this is not a wiring fault. Both entropy directions
fail equally, which rules out the first explanation (that a boundary should
sit where prediction is easy) and leaves a better one:

**consistency of the boundary rule beats content-awareness.** A fixed stride
knows nothing and wins; separators are word-aligned and nearly as good;
entropy boundaries are content-dependent, so the same word is cut
differently depending on what surrounds it and the pooled patch
representation never means the same thing twice. The expensive layers see
only those pooled patches, and they cannot learn a unit that moves.

That is the opposite of the Byte Latent Transformer's argument, and the
reason is structural: BLT's global model reads bytes, this one reads pools.

All of it still loses to the plain byte baseline.

### A (moonshots). Trajectory extrapolation - negative for raw iterates

`w_new = w + a*(w - w_prev)`, from the checkpoint series, evaluated
held-out. A positive control at a=0 reproduces the source checkpoint to five
decimals, so the round trip through `ckpttool` is exact.

```
s88000 (source)              1.70293
control a=0                  1.70292
short baseline, a=0.5        1.76692
short baseline, a=1.0        1.92370
long baseline,  a=0.5        1.78975
long baseline,  a=1.0        2.03426
```

Monotonically worse at every distance and both baselines. The trajectory is
not straight enough for it: cos between the direction over the last 8000
steps and over the last 28000 is 0.605.

Untested, and the variant the moonshot actually recommends: extrapolating
the *averaged* weights rather than the iterates, which needs a run with
`wavg=1`. Also untested is recovery - whether a jump that lands worse
recovers faster than it cost.

### ckpttool patch was broken

It cast parameter blocks to `[]f32`, but they begin at whatever offset the
configuration text ends at - 811 in a current checkpoint - so `@alignCast`
panicked. `dump` never noticed because it writes raw bytes. Fixed to work
bytewise; this is the tool moonshot A needs and it had never run.

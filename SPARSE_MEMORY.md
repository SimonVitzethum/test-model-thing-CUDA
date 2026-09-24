# Sparse associative memory

What the fly's mushroom body computes, whether it scales, what it cannot
do, and the one mechanism by which it could make training cheaper rather
than merely larger.

Grew out of section K of MOONSHOTS.md, which now points here. **Revised
after review**: the capacity formula in the first draft was taken from the
wrong memory model, the cost accounting hid the decision that dominates
it, and a failure mode was missing that would have killed the whole thing
silently. Section 11 lists what changed.

---

## 1. The circuit, measured

The connectome is on this disk - `~/Dokumente/CNS`, the male CNS dataset,
211,577 neurons and 152M weighted edges - so these are measurements, not
figures from a paper. Right hemisphere, connections of at least five
synapses, via `~/tmt-scratch/fly/mb.py` and `mb3.py`:

```
343 projection neurons carrying 62-83 glomeruli    the input channels
                    ->            2045 Kenyon cells   expansion 25x
fan-in per Kenyon cell            4.8 claws           density 1.4%
                    ->              49 MBONs          the readout
                                     2 APL            one per hemisphere
                                   166 dopaminergic   the third factor
```

Three details beyond the headline shape.

**The inhibition is exactly global.** 2081 connections into the single APL
neuron and 2078 back out, for 2045 Kenyon cells. Every cell drives it and
every cell is inhibited by it: a normalisation loop holding activity at a
fixed level, with no structure to learn.

**The modulator addresses the synapse, not the cell.** 1470 dopaminergic
connections land on Kenyon cells against 369 on MBONs, so the third factor
reaches the KC->MBON synapse in compartments. The rule is presynaptic
activity times postsynaptic compartment times modulator.

**The readout is broad.** Each MBON reads a mean of 277 Kenyon cells,
about 14% of the population, with a tail to 1502.

---

## 2. The randomness check, which nearly went the wrong way

FlyHash rests on the expansion being random. A structured expansion is not
a hash, so this had to be verified.

The first measurement said Kenyon cells share **5.25 times** more input
than chance, which would have ended the idea. It was an artifact of the
null model: several projection neurons carry the same glomerulus, and the
glomeruli are not sampled equally. Both inflate overlap above a uniform
null with no wiring structure at all.

Against a **degree-preserving** null - expected overlap
`d_i * d_j * sum(d_p^2) / M^2` for a bipartite configuration model:

```
                  observed   uniform null        degree-preserving
per PN               0.419   0.202  (2.07x)      0.419  (1.00x)
per glomerulus       0.603   0.418  (1.44x)      0.644  (0.94x)
```

**Random, conditional on degree.** The premise holds.

The lesson generalises past this circuit: a uniform null over a graph with
unequal degrees manufactures structure that is not there.

---

## 3. Interference per write

Two random k-sparse codes among `m` units share `k^2/m` units on average.

```
m = 12800, k = 640    ->  32 shared,  5% of the code
dense delta rule      ->            100% of the entries
```

One new write disturbs one existing association by five percent instead of
all of it. That is the argument for preferring this over the hand-picked
delta rule in MOONSHOTS section C.

It is **not** an argument about capacity, which the first draft got wrong
by extrapolating from it. Capacity is section 4.

---

## 4. Capacity, simulated

The first draft used `0.48 (m/k)^2`. That is the capacity of a
**Willshaw** memory: an `m x m` matrix of binary synapses, `m^2` storage
sites. This store is a different object - `m` units each holding a
`d`-dimensional vector, added on write - and the formula does not transfer.

### The derivation

After `N` writes, each row has been touched by about `N*k/m` patterns. A
read sums `k` rows: the signal grows with `k`, the crosstalk adds
incoherently and grows with `sqrt(k * N*k/m)`. So

```
SNR  =  sqrt(m / N)
```

Capacity is **linear in m**, and `k` drops out.

### Simulated, because a second wrong formula would be no better

`~/tmt-scratch/fly/capacity.py`, additive store, random codes:

```
        m      k       N     SNR    m/N   sqrt(m/N)
    65536     64    1638    6.32   40.0        6.33
    65536     64    6553    3.15   10.0        3.16
    65536     64   16384    1.88    4.0        2.00
    65536     64   65536    1.01    1.0        1.00

k at fixed m=16384, N=1638:   k=16 -> 3.16   k=64 -> 2.86   k=256 -> 1.43
```

`sqrt(m/N)` to two digits, and `k` is nearly irrelevant - mildly worse when
the code stops being sparse. **Usable recall needs `N <~ m/10`.**

### The information check, which says the same thing

`10^7` patterns of 512 values is `5*10^9` numbers. A store of `320k x 512`
has `1.6*10^8` slots. Even perfectly coded it is short by a factor of 30.
No arrangement of a sparse code repeals that.

### What that does to the sizing

Store a **narrow value**, not the 512-dimensional state. A compact
prediction for the next byte or patch is what the store is for; the full
state is not.

| associations `N` | units `m = 10N` | 512-d bf16 | 256 int8 logits | 64-d bf16 |
|---|---|---|---|---|
| 10⁵ | 10⁶ | 1.0 GB | 256 MB | 128 MB |
| 10⁶ | 10⁷ | 10 GB | 2.6 GB | 1.3 GB |
| 10⁷ | 10⁸ | 102 GB | 26 GB | 13 GB |

The first draft claimed ten million associations for 467 MB. The real
figure is **ten million for 13-26 GB**, or **100,000 for 128-256 MB** in
8 GB of VRAM beside a training model. That is a different proposition, and
it is the correction that matters most on this page.

---

## 5. Cost, and the decision that dominates it

The first draft quoted per-window costs and then discussed reading, which
hid the only decision that matters: **is the store read once per window, or
at every position?**

Basis: 16 layers, topk=2, three GEMMs each (forward, dX, dW) at 262k MACs,
so about **25M MACs per position** for the expert path.

| | per window of 128 | per position |
|---|---|---|
| hash at `m = 320k` (5m MACs) | 0.05% of the model | **6%** |
| hash at `m = 3.8M` | 0.6% | **76%** |
| read `k=100` rows, bandwidth | 6.4 MB per step | **819 MB per step** |

Per-position retrieval on a step that already moves ~2 GB adds 40% to the
traffic - on a machine at **82% of its streaming ceiling**, which is the
one resource with nothing spare. Plus the top-k, which is `O(m)` per
retrieval and was missing from the first draft entirely.

So per-window retrieval is not a detail, it is the design. Per-position
retrieval is affordable only at small `m`.

---

## 6. The hard limit

**A mushroom body is a memory *over* a representation. It does not produce
one.**

Its generalisation comes from a **fixed random similarity metric**: similar
inputs get overlapping codes, which is the whole of locality-sensitive
hashing. The metric is never learned. Feed it raw bytes and similarity
means byte overlap, which for language is close to worthless - "king" and
"monarch" share no byte.

Learning the similarity metric is what gradient descent on a deep network
provides, and it is what training is for.

The fly learns an **association** in one trial. It never learns a
**concept**, and never generalises to an odour it has not smelled except
through the accidental geometry of its input space.

---

## 7. The mechanism that could matter, and the trap inside it

Capacity alone does not make training faster. This might.

During training, write each window into the store **keyed on the model's
current learned representation**. When similar context returns, the store
supplies the answer, the loss there is already low, so **the gradient there
is small** and the weights are never pushed to memorise it. The gradient
budget goes to what the store cannot supply: structure.

It solves the metric problem, because the key is a learned representation
rather than raw bytes, so the hash improves as the model does. And it is
section C in reverse: keeping content *out* of the weights instead of
consolidating it *in*.

### The trap: self-retrieval across epochs

**This would have failed silently, with an excellent-looking loss curve.**

The corpus is 950 MB and runs go 14 to 24 epochs. From the second epoch,
the store contains every training window together with its exact
continuation. Retrieval is then near-perfect **everywhere**, the loss falls
everywhere, and the gradient vanishes everywhere - including at the
positions where the model was supposed to be learning structure.

The mechanism cannot tell "content that belongs in a store" from
"structure that belongs in the weights". It only sees "encountered
before".

**Same-document exclusion is mandatory, not a safeguard.** Drop retrieval
hits from the document the query came from. The store then helps only on
genuine repetition *across* documents - boilerplate, quotations, recurring
facts - which is exactly the content that should not be in the weights.

This is not a new invention: kNN-LM, Memorizing Transformers and TRIME are
the direct precursors, they all do this, and their reported gains were
**real but moderate**. That is the prior to hold, not a large factor.

### The second known problem: stale keys

The keys come from a representation that keeps changing. Entries written
early are addressed by a metric the model no longer uses. They have to be
re-keyed periodically or aged out, and the same literature says so.

---

## 8. The competing explanation, which is more parsimonious

This page has been leaning on one reading of the measurement at the top of
TODO.md - 14 epochs gives 1.63 BPB, 24 gives 1.82 - namely that capacity is
being burned on content the model cannot hold.

**The simpler reading is ordinary overfitting in a data-limited regime**,
and it should be stated first. Muennighoff et al. found the value of
repeated data falls off sharply after about four epochs and is near zero
past sixteen. Fourteen and twenty-four are both well past that. Nothing
about a memory mechanism is needed to explain the result.

The consequence for priority is direct: **with 950 MB at double-digit
epochs the limit is data, not compute.** More unique data, or rephrased
data (TODO item 20, a reported 3x), is the cheapest lever available and
probably larger than any memory mechanism on this page.

Training on 950 MB is a reasonable choice for fast iteration. But the
epoch findings from that setting should not be extrapolated to larger runs,
and they should not be used as the argument for building this.

---

## 9. What to measure first

Cheap, and it bounds everything above before any of it is built.

Train two identical runs, one with the store and one without, and compare
more than the loss: **how much gradient mass falls on positions the store
could have answered.** That fraction is the ceiling. If it is 5%, the idea
is worth 5%.

**Run it with same-document exclusion**, or it measures self-retrieval and
returns a number near one, which would be meaningless and encouraging in
the worst way.

For the write rule itself the instrument exists. `memprobe` measures loss
against position within a document; build the store once with the delta
rule and once with the FlyHash rule and compare the curves.

---

## 10. What would kill it

- **The gradient-mass measurement comes back small**, and memorisation is
  not where the compute goes.
- **Same-document exclusion removes most of the benefit.** Quite likely:
  it is what separates this from self-retrieval, and what is left is
  cross-document repetition alone.
- **The learned representation makes a poor hash key.** Locality-sensitive
  hashing needs the input geometry to carry the right similarity, and a
  representation trained for next-byte prediction may not.
- **Re-keying costs more than the store saves**, as the representation
  drifts.
- **Capacity in practice falls below `m/10`.** The simulation uses random
  patterns; natural language contexts are correlated, and correlated
  patterns interfere more.

---

## 11. What the review changed

For the record, since four of these were wrong in the first draft.

1. **Capacity is linear in `m`, not quadratic in `m/k`.** The old formula
   was Willshaw's, for a memory made of `m^2` binary synapses. Simulated in
   section 4 rather than swapping one borrowed formula for another.
2. **Sizing is 10-100x worse.** Ten million associations is 13-26 GB with
   narrow values, not 467 MB. The information-theoretic check catches this
   without any simulation and should have been run first.
3. **Compute and bandwidth were quoted per window while the text discussed
   per-position reading.** Per position the hash is 6% of the model at
   `m=320k` and the reads add 819 MB per step, on the bottleneck. The
   `O(m)` top-k was missing entirely.
4. **Self-retrieval across epochs was missing**, and it is the failure that
   would have looked like success. Same-document exclusion is mandatory.
   kNN-LM, Memorizing Transformers and TRIME are the precursors; their
   gains were moderate.
5. **The epoch result has a simpler explanation** than the one this page
   was built on, and the priority consequence is that data beats memory.

---

## 12. Implementation sketch

In the order the pieces would land:

1. **The hash.** A fixed random sparse projection, `m` units of ~5 inputs
   each with a non-uniform degree over channels, plus a top-k. A gather and
   a selection, both with close relatives in `kernels.zig`.
2. **The store.** `m x d` with `d` narrow - a compact next-byte prediction,
   not the 512-dimensional state.
3. **The key.** The layer representation at the window boundary, which
   `extract_carry` already produces.
4. **Retrieval granularity.** Per window unless measurement says otherwise;
   see section 5.
5. **Document exclusion**, section 7. Not optional.
6. **The read.** Same hash, gather the active rows, feed them where
   `mem_rdim` retrieval heads already feed.

Scripts behind sections 1, 2 and 4 are in `~/tmt-scratch/fly/`, reading the
connectome from `~/Dokumente/CNS/fly-llm/data/`.

---

## 13. An application that is not language modelling

DREAM_RSI.md uses this store as the replay simulator of a
recursive-self-improvement loop over kernel search, where the three
properties measured above - one-shot writes, low interference, and
locality-sensitive retrieval - are exactly what such a simulator needs, and
where the inability to extrapolate stops being a limitation because the
method claims the same scope.

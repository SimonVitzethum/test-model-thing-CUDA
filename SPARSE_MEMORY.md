# Sparse associative memory

What the fly's mushroom body computes, whether it scales, what it cannot
do, and the one mechanism by which it could make training substantially
cheaper rather than merely adding capacity.

This started as section K of MOONSHOTS.md and outgrew it. K now points
here.

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

Three details matter beyond the headline shape.

**The inhibition is exactly global.** 2081 connections into the single APL
neuron and 2078 back out, for 2045 Kenyon cells. Every cell drives it and
every cell is inhibited by it. That is a normalisation loop holding
activity at a fixed level, with no structure to learn.

**The modulator addresses the synapse, not the cell.** 1470 dopaminergic
connections land on Kenyon cells against 369 on MBONs, so the third factor
reaches the KC->MBON synapse in compartments. The rule is presynaptic
activity times postsynaptic compartment times modulator.

**The readout is broad.** Each MBON reads a mean of 277 Kenyon cells,
about 14% of the population, with a long tail up to 1502.

---

## 2. The randomness check, which nearly went the wrong way

FlyHash rests on the expansion being *random*. A structured expansion is
not a hash, so this had to be verified rather than assumed.

The first measurement said Kenyon cells share **5.25 times** more input
than chance. That would have ended the idea.

It was an artifact of the null model. Several projection neurons carry the
same glomerulus, so a cell sampling "DA1" may connect to several DA1
neurons; and the glomeruli are not sampled equally. Both inflate overlap
above a uniform null without any wiring structure at all.

Against a **degree-preserving** null - expected overlap
`d_i * d_j * sum(d_p^2) / M^2` for a bipartite configuration model:

```
                  observed   uniform null        degree-preserving
per PN               0.419   0.202  (2.07x)      0.419  (1.00x)
per glomerulus       0.603   0.418  (1.44x)      0.644  (0.94x)
```

**Random, conditional on degree.** The premise holds. The only structure
is that some glomeruli are sampled more than others, which a degree
distribution reproduces.

The lesson generalises past this circuit: a uniform null over a graph with
unequal degrees will manufacture structure that is not there.

---

## 3. Why one trial is enough

Two random k-sparse codes of size `k` among `m` units share `k^2/m` units
on average.

```
m = 12800, k = 640    ->  32 shared,  5% of the code
dense delta rule      ->            100% of the entries
```

A new write disturbs an existing one by five percent instead of all of it.
That is the entire mechanism. Sparsity buys non-interference, and
non-interference is what makes a single presentation sufficient.

This is an arithmetic argument, not an appeal to biology, and it is the
reason to prefer this over the hand-picked delta rule in MOONSHOTS
section C.

---

## 4. Expanding it

### Capacity

Interference accumulates across writes, so capacity scales with the square
of the sparsity ratio. For Willshaw-type sparse associative memory,

```
capacity  ~  0.48 * (m/k)^2
```

Order of magnitude, not a guarantee. Checked against the animal: the fly
has `m = 2000, k = 100`, giving `0.48 * 400 = 192` associations, which is
the right size for what a fly learns.

### Sizing for this project

The store writes **once per window of 128 bytes**, not once per byte. That
single decision is what makes the large sizes affordable, and it is the
natural granularity anyway, since an episode is a window.

| units `m` | capacity | store, 512 dims bf16 | MACs per position |
|---|---|---|---|
| 12,800 | ~8×10³ | 13 MB | 500 |
| 456,000 | ~10⁷ | 467 MB | 18,000 |
| 3,800,000 | ~7×10⁸ | 3.9 GB | 148,000 |

Against a per-position cost of one 512×512 GEMM at 262,144 MACs, and a
step that spends roughly 8.4M MACs per position on the expert path, even
the largest store is **under 2% of the step**. The 467 MB row is the
practical one: ten million associations for 0.2% of the arithmetic.

Reading touches `k` rows: 100 × 512 × 2 = 100 KB, against the ~2 GB the
step already moves. Negligible.

### What the expansion costs in the currency that is short

This machine runs at 82% of its streaming ceiling and **15% of its matmul
ceiling**. The expansion is arithmetic on data that is already resident,
and it is sparse. It spends the resource there is spare.

The store itself is memory, and the 3.9 GB row does not fit beside a
training model in 8 GB of VRAM. 467 MB does.

---

## 5. The hard limit

**A mushroom body is a memory *over* a representation. It does not produce
the representation.**

Its generalisation comes from a **fixed random similarity metric**: similar
inputs get overlapping codes, which is the whole of locality-sensitive
hashing. But the metric is never learned. Feed it raw bytes and similarity
means byte overlap, which for language is close to worthless - "king" and
"monarch" share no byte.

Learning the similarity metric is exactly what gradient descent on a deep
network provides, and it is what training is for.

The fly learns an **association** in one trial. It never learns a
**concept**. It does not generalise to an odour it has not smelled, except
through the accidental geometry of the input space.

So an expanded mushroom body makes the *storage* half cheap and leaves the
*representation* half entirely intact.

---

## 6. The mechanism that could matter: redirecting the gradient

Capacity alone does not make training faster. This does.

During training, write each window into the hash store **keyed on the
model's current learned representation**. When similar context returns, the
store supplies the answer, the loss at those positions is already low, and
therefore **the gradient there is small**. The weights are never pushed to
memorise it.

The gradient budget then goes to what the store cannot supply: structure.

Two things make this the right version of the idea.

**It solves the metric problem.** The key is the learned representation
rather than raw bytes, so the hash improves as the model does. The fixed
random projection sits on top of a learned space instead of on top of
bytes.

**It is C in reverse.** Section C consolidates memory *into* the weights.
This keeps content *out* of them.

And it explains the measurement at the top of TODO.md from a third
direction. 24 epochs coming out worse than 14 is the model spending
capacity on content it cannot hold. This mechanism removes the pressure
that produces that.

---

## 7. The honest ceiling

Not a factor of 100.

The gain is bounded by **the fraction of training compute that goes into
memorising**, and nobody knows that number. For a 34M model the amount
retained is small - on the order of 2 bits per parameter, so about 8 MB -
but the compute *wasted* attempting to hold more can be large. That is
precisely what the 14-versus-24-epoch result measures, and it has never
been quantified.

---

## 8. What to measure first

Cheap, and it bounds everything above before any of it is built.

Train two identical runs, one with the hash store and one without, and
compare more than the loss:

**How much gradient mass falls on positions the store could have
answered.** That fraction is the ceiling on what this can buy. If it is 5%,
the whole idea is worth 5% and should be dropped. If it is half, it is the
most important thing on any of these pages.

The instrument for the write rule itself already exists. `memprobe`
measures loss against position within a document; build the store once with
the delta rule and once with the FlyHash rule and compare the curves. A
rule that writes without interference should hold up further into the
document.

---

## 9. What would kill it

- **The gradient-mass measurement comes back small.** Then memorisation is
  not where the compute goes and the premise is wrong.
- **The learned representation makes a poor hash key.** Locality-sensitive
  hashing needs the input geometry to carry the right notion of similarity;
  a representation trained for next-byte prediction may not.
- **The store is read but not used.** A model can learn to ignore an input
  that is unreliable early in training, and then never revisit it. Watch
  whether the retrieval heads carry gradient at all.
- **Capacity in practice falls far below `0.48 (m/k)^2`.** That bound is
  for random patterns. Natural language contexts are correlated, and
  correlated patterns interfere more.

---

## 10. Implementation sketch

Against this codebase, in the order the pieces would land:

1. **The hash.** A fixed random sparse projection, `m` units of ~5 inputs
   each drawn with a non-uniform degree over channels, plus a top-k. One
   kernel; it is a gather and a selection, both of which have close
   relatives in `kernels.zig` already.
2. **The store.** `m x dim` in bf16, written at the `k` active rows.
3. **The key.** The layer representation at the window boundary, which
   `extract_carry` already produces.
4. **The read.** Same hash, gather the `k` active rows, feed them where the
   memory path already feeds - `mem_rdim` retrieval heads exist.
5. **The leakage guard.** Exclude the current window from its own
   retrieval, the same trap as in MOONSHOTS C and G. At evaluation, the
   store holds training data only.

Scripts behind the measurements in section 1 are in `~/tmt-scratch/fly/`.
They read the connectome from `~/Dokumente/CNS/fly-llm/data/`.

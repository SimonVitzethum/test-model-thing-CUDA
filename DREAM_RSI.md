# Dream-RSI and the sparse store

Using the sparse associative memory of SPARSE_MEMORY.md not *beside* the
model but *around* the work on it: as the replay simulator of a
recursive-self-improvement loop over kernel and configuration search.

The first application on these pages where the store accelerates the
research rather than the language model.

---

## 1. What Dream-RSI is

[arXiv:2609.14858](https://arxiv.org/abs/2609.14858), September 2026,
[code](https://github.com/zhengkid/Dream-RSI). Three stages in a loop:

1. **Online explore** - run real, expensive evaluations.
2. **Construct a replay simulator** from the accumulated discovery history.
3. **Dream** - evaluate candidate *policies* inside that simulator for cheap
   off-policy feedback, refine, and redeploy online.

The claim underneath is that the history of what has already been tried is
itself a simulator over the part of the search space that has been
realised. Reported across algorithm engineering, mathematical optimisation
and **GPU kernel engineering**, with competitive discovery quality at
substantially lower cost.

---

## 2. Why the mushroom body is the right memory for it

A replay simulator needs a store with three properties, and they are the
three that were measured in SPARSE_MEMORY.md section 1-3:

| Dream-RSI needs | the sparse store gives |
|---|---|
| write each trial **once** | one-shot write, no gradient |
| trial 1000 must not overwrite trial 1 | `k^2/m` interference, 5% not 100% |
| predict an **untried** candidate | locality-sensitive hashing: similar candidates, overlapping codes |

**"Dreaming" is then literally a memory read.** Hash the candidate, gather
the active rows, and what comes back is the averaged experience of similar
candidates. The interpolation over code overlap *is* the simulator's
prediction function - that is not an analogy.

And the hard limit of section 6 stops being a limit here. A memory over a
fixed metric cannot extrapolate, only interpolate between what it has seen.
Dream-RSI claims exactly the same scope: the simulator is valid over the
**realised** search space and nowhere else. The two assumptions match
instead of fighting.

---

## 3. Where it fits this project, and where it does not

### GPU kernel engineering: yes

One of the paper's own three domains, and it is the daily work here. The
infrastructure exists: `kbench` with measured ceilings, 83 kernels, and a
Zig-to-PTX pipeline that turns a variant into a benchmark in minutes.

**Minutes per trial is the regime a replay simulator is for.** Hundreds to
thousands of trials are affordable, which is what it takes for the
simulator to be worth more than a lookup table.

The loop:

```
kernel variant -> feature vector -> hash -> measured throughput, stored
dream: predict throughput of untried variants from the store
compile and benchmark only what the dream ranks highly
```

Features are available without new instrumentation: block size, tile shape,
unroll factor, bytes per thread, shared memory per block, register
pressure, the arithmetic intensity of the kernel. `kbench` already reports
the outcome side.

### Training configurations: no

A trial costs **hours**. Dozens are affordable, not thousands. With twenty
points a "replay simulator" is a small regression, and dreaming adds
nothing that reading the table would not.

This is the honest boundary, and it is about trial cost rather than about
the method.

---

## 4. The convergence worth noticing

Three independent threads arrive at the same place.

The review of the moonshots concluded that the only 100x worth writing down
without hesitation is **faster research, not faster training** - proxy
models, branched cooldowns, one-hour pretests that kill ideas before they
are built.

Dream-RSI is the formalised version of that.

And the sparse store is the mechanism underneath it.

None of the three was aimed at the other two.

---

## 5. What could not be verified

The abstract gives neither the number of online trials the simulator needs
before it pays, nor concrete figures for the kernel-engineering domain, nor
the authors' own stated limitations. The full PDF is required reading
before building on it, and **how many trials it takes before the simulator
carries** is the number that decides whether this is a tool here or a
footnote.

---

## 6. A warning that is not technical

"Recursive self-improvement" attracts overclaiming, and this project has
already taken criticism in public for exactly that.

If something comes of this, the thing to describe is what it is: **a
prediction model for kernel benchmarks that saves the expensive
measurements.** That is a useful piece of engineering and it does not need
a grander name.

---

## 7. First step

Not an implementation. Read the PDF and find the trial count. If the
simulator needs thousands of trials before it beats random search, the
kernel domain still qualifies and the configuration domain still does not,
and that is the whole decision.

After that, the cheapest test is offline and needs no new loop: take the
kbench results that already exist, hold out a third of them, and see
whether a sparse store over the remaining two thirds predicts the held-out
throughputs better than the mean. That is an afternoon and it says whether
the hash carries any signal about this search space at all.

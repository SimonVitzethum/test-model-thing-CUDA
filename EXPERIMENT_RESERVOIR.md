# Does the connectome contribute anything?

An experiment on `fly-llm`, designed to answer one question with a number:
**when a fly connectome is used as a fixed recurrent layer in front of a
language model, does the specific wiring matter, or only its graph
statistics?**

Everything downstream of `fly-llm` rests on the answer, and nobody has
measured it for a language task.

---

## 1. Why this is the question

`FlyBigLM` is

```
Embedding -> FlyReservoir -> 4 transformer blocks -> LayerNorm -> head
```

and the connectome enters as `register_buffer("crow"/"col"/"val")`, not as
`nn.Parameter`. **It is not trained.** Only `W_bias` is learnable. So the
fly is a fixed recurrent feature map and the four attention blocks do the
language modelling.

Which means the contribution of the connectome is currently unknown. It
could be doing real work, or it could be an arbitrary sparse random graph
with a good spectral radius.

### What the public fly sims say, and why it raises the stakes

The viral demonstrations - Doom, Beat Saber, Mario 64 - do **not** learn.
The DOOMFLY validation report fails its own vision, conditioning and
survival gates; 6,385 Doom attempts never beat the game. The Beat Saber
demo is overfit to a single track and barely uses visual input. Patrick
Mineault's analysis calls the sims noise-driven random generators with
decorative sensory systems, built from leaky-integrate-and-fire units with
uniform parameters and connectome weights scaled by one fitted constant.

That is a qualitative critique. **This experiment is the quantitative
version of it, in a domain where it has not been run.** A null result is
publishable in the informal sense that it settles something; a positive
result is more interesting still.

---

## 2. The design: a ladder, not a baseline

A single "random graph" control answers almost nothing, because a random
graph differs from the connectome in a dozen ways at once. Instead,
preserve progressively more of the fly and find the rung where the benefit
appears - if it appears.

| # | Null model | Preserves |
|---|---|---|
| 0 | Erdos-Renyi | edge count only |
| 1 | Configuration model | in and out degree sequences |
| 2 | + sign-matched | excitatory/inhibitory identity per neuron |
| 3 | + weight-matched | the synapse-count distribution |
| 4 | Within-class rewiring | the coarse class/superclass block structure |
| 5 | Within-region rewiring | neuropil-level structure |
| 6 | The real connectome | everything |

Sign comes from `body-neurotransmitters.feather`, class and superclass from
`body-annotations.feather`. Both are already on disk.

**The shape of the curve is the result.** A jump between rungs 1 and 2 says
the excitatory/inhibitory balance is what matters. A jump only at 6 says
something in the specific wiring does. A flat line says the connectome is
decoration - which is a real answer and the one the public sims predict.

### The targeted variant

One more condition, because it tests the claim this project already
measured: **the mushroom body real, everything else rewired at rung 3.**

SPARSE_MEMORY.md found the mushroom body's expansion is random conditional
on degree, with a sparse code and a k-winner - a motif that transfers.
If circuit motifs are what matter rather than whole brains, this condition
should beat rung 3 and approach rung 6.

---

## 3. The control that decides whether the experiment is valid

**Match the spectral radius across every condition.**

Reservoir performance is dominated by the spectral radius of the recurrent
matrix - the edge-of-chaos argument - and it varies wildly between random
graph models at fixed edge count. Without rescaling, this experiment
measures the spectral radius and nothing else, and it would produce a
confident, wrong answer.

So: build each graph, compute its leading eigenvalue with power iteration
on the sparse operator, rescale `val` so that every condition has the same
rho. Then sweep rho over a few values per condition, because the optimum
may differ.

Also held fixed:

- unit count `N` and edge count,
- `alpha`, the leak,
- embedding, transformer blocks, head, optimiser, schedule, data order,
- the number of trainable parameters, which the rewiring does not change.

And: **the same seeds, several of them.** Reservoir results are noisy.
Five seeds per condition, compare held-out loss with error bars, and treat
anything inside one standard error as a tie.

---

## 4. What to measure besides the loss

The loss says whether. These say why, and they are cheap.

**Memory capacity** (Jaeger): train linear readouts to reconstruct the
input `k` steps back, sum the squared correlations over `k`. One number for
how far the reservoir remembers.

**Separation**: feed pairs of inputs at a controlled distance and measure
the distance between the resulting states. A reservoir that collapses
different inputs to similar states cannot help downstream.

**Activation statistics**: `mean_act` and `mean_sat` already exist in
`autonomous.reservoir_stats`. A saturated reservoir is a constant function
wearing a disguise.

If a condition wins on loss, one of these should explain it. If none does,
suspect the spectral radius control.

---

## 5. Implementation

Small, because the code is already shaped for it. `FlyReservoir` holds the
matrix as CSR buffers, so a variant only has to build different
`crow/col/val`:

```python
class FlyReservoir(nn.Module):
    def __init__(self, N, alpha, graph="connectome", rho=None, seed=0):
        W = build_graph(graph, N, seed)     # the ladder lives here
        W = rescale_spectral(W, rho)        # the control
        self.register_buffer("crow", W.crow_indices())
        ...
```

`build_graph` needs one function per rung. Rungs 1-3 are standard
configuration-model sampling; 4 and 5 shuffle edges within blocks defined
by the annotation columns. All of them read from
`~/Dokumente/CNS/fly-llm/data/`.

`rescale_spectral` is power iteration on the sparse matrix, twenty
iterations, then divide.

Nothing else in `train_big.py` changes.

---

## 6. Cost

At the sizes `fly-llm` already trains: seven conditions plus the mushroom
body variant, five seeds each, times a few spectral radii. That is a few
dozen short runs - an overnight job at the scale of `fly-big.pt`, not a
research programme.

Run rungs 0, 1, 3 and 6 first with three seeds. If those four are within
noise of each other, the ladder is flat and the remaining rungs are not
worth the electricity.

---

## 7. What each outcome means

**Flat.** The connectome contributes graph statistics and nothing else.
Then expanding "the whole fly brain" is unnecessary: generate a larger
graph with matched statistics and skip the connectome entirely. It also
confirms, with numbers, what the critique of the public sims says with
words.

**Rises at rung 2 or 3.** Sign structure or weight distribution matters.
Those are easy to reproduce at any size, so expansion is straightforward
and the fly is still not needed.

**Rises only at rung 6.** Something in the specific wiring matters and
nobody knows what. Then the mushroom body condition becomes the important
one, because it distinguishes "a motif transfers" from "the whole thing
does" - and the first is buildable at scale while the second is not.

**The mushroom body variant approaches rung 6.** The best outcome for this
project, because it is the one that connects to SPARSE_MEMORY.md and to
something that can be scaled deliberately rather than copied.

---

## 8. What this experiment cannot say

It tests the connectome as a **fixed recurrent feature map in front of a
transformer**, which is what `fly-llm` does. It says nothing about the
connectome as a trained sparsity mask, nothing about spiking dynamics, and
nothing about the fly's own tasks.

And the missing biology stays missing either way: synaptic strengths, time
constants, delays and intrinsic electrophysiology are not in the volume,
which is the deeper reason the public sims do not work. No graph-level
experiment recovers them.

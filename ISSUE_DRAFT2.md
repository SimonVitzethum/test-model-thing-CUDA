## Correction: the "capacity collapse" was a kernel bug, and the capacity question is still open

In the comment above I reported that doubling the experts from 8 to 16 —
which keeps the arithmetic per token identical at `topk=2` — collapsed:
110 of 256 expert slots dead, training loss oscillating up to 15.5 BPB, and
I concluded that "the next limit after data isn't compute, it's router
stability."

That was wrong. It was a bug in my router kernel, and it had been there the
whole time.

**What it was.** `router_topk` is launched with 64 threads and indexes its
experts as `te / 8`, so it computes logits for experts 0 through 7 and no
others. The row below it reads `moe_lp[0..E]` regardless. At eight experts
that is exactly right and at sixteen half the logits are uninitialised
shared memory.

It does not crash. It routes on garbage, quietly, and the result looks
convincingly like a training instability. The arithmetic even fits: 8
broken experts across 16 layers is 128 slots that should never be chosen,
and 110 were observed dead.

A second bug sat behind it and would have surfaced the moment the first was
fixed. The warp shuffle that reduces each expert's dot product uses the mask
`0xFF << e*8`, which is warp-local — beyond four experts per warp it shifts
past 32 and produces an empty mask. Both are fixed: the launch now provides
`E*8` threads and the mask is computed from the lane's position inside its
own warp.

**After the fix**, same configuration, same data, same seed:

```
step    before (bug)    after (fix)
4000      15 / 256        0 / 256
6000      47 / 256        0 / 256
8000      69 / 256        0 / 256
9000      67 / 256        0 / 256

BPB at step 9000:   2.414  ->  1.677
```

No dead experts at any point, and the sixteen-expert model is now well
ahead of where the broken one was.

**How it was found**, since the route there is the useful part. Believing
the collapse was a real training dynamic, I added noisy routing with a
decaying warm-up — expert selection reinforces itself from the first step,
before any expert has earned a preference, and noise early on is the
standard remedy. It crashed with an illegal memory access after 7080 steps:
uninitialised logits reaching the top-k selection and producing an
out-of-range expert index.

So the fix for the problem was found by building a fix for a different
problem that did not exist. The noise stays in — it is sound on its own
terms and costs nothing when disabled — but it fixes nothing here.

**What this means for the earlier claim.** The capacity question is
reopened and currently unanswered. Everything I said about the auxiliary
loss being too weak at sixteen experts, about tokens per expert halving,
about recovery rates falling below starvation rates — none of it was
measuring what I thought. The auxiliary loss share rising from 0.12 to 0.36
over that run was the balancing term straining against garbage, not against
a real imbalance.

The data-limited finding from the original post stands; it came from a
different measurement and an eight-expert configuration, where the kernel
was correct.

**The lesson I'd pass on**, because it cost a night and an incorrect public
claim: a plausible mechanism is not evidence. I had a coherent story —
positive feedback in expert selection, halved tokens per expert, a
balancing term diluted across twice as many experts — and every piece of it
was consistent with the numbers. What I did not have was a reason to
believe the kernel was correct at a configuration it had never been run at
before.

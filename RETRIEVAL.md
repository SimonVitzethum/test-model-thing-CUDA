# Retrieval over the training corpus

Moonshot G, being built. This file is the experiment design and the running
record; MOONSHOTS.md keeps the argument for why it is worth doing.

The claim under test: byte prediction on Wikipedia mixes linguistic
structure, which is compressible and learnable from little data, with
factual and verbatim content, which is not and does not fit in 34M
parameters. Gradient descent spends most of its capacity on the second. An
index over the corpus makes it free.

## What "15x speedup" has to mean, and the trap in it

There are two different claims people call a retrieval speedup:

- **Data efficiency.** The baseline reaches bits-per-byte X after B bytes of
  training; the retrieval system reaches X after B/15.
- **Parameter efficiency.** A retrieval model with N parameters matches a
  baseline with 15N. This is RETRO's claim, and it is not a speedup.

Only the first is a speedup, so that is what gets measured.

**The trap is that the answer depends entirely on where you stop the
baseline.** Retrieval contributes a roughly constant offset in BPB - it does
not get better as the model trains. The baseline's own curve flattens. So
the same constant offset is worth a great many steps late in training and
very few early on, and a ratio quoted without the baseline's budget means
nothing. An undertrained baseline makes any improvement look enormous.

Therefore: **report the curve, not the ratio.** The ratio is one number read
off it, and it is quoted with the budget it was read at.

## Protocol

One training run, evaluated at many points.

1. Train the baseline, keeping checkpoints along the way - `saveevery`
   overwrites one path, so a copier keeps `s<step>.ckpt` beside it.
2. For each checkpoint, `train mode=eval lossout=...` writes the per-byte
   cross entropy back in corpus order.
3. `ngram DATASTORE HELD LOSSFILE` mixes in the retrieval distribution and
   reports BPB against lambda. Two curves come out: BPB_base(step) and
   BPB_retr(step).
4. Let X be the baseline's BPB at its final step. Find the smallest step
   where the retrieval curve reaches X. The ratio of the two steps is the
   speedup **at that budget**.

### Rules that keep it honest

- **The datastore is the training corpus only.** Never the held-out set.
  Near-duplicate passages across Wikipedia articles are not leakage -
  exploiting them is the whole point - but retrieving over the evaluation
  data would be.
- **Lambda is chosen on one half of the held-out data and reported on the
  other.** Picking the mixing weight on the test set inflates the result by
  exactly the amount the reader cannot see.
- **The baseline has to be trained far enough to flatten**, or the speedup
  is measuring the steep part of the curve rather than anything retrieval
  did.

## Why output interpolation first

What G actually proposes is feeding retrieved passages into the model's
memory slots, so the model learns when to trust them. That is a training
change and it costs bandwidth on a step that already runs at 82% of the
streaming ceiling.

Interpolating the *output* distribution needs none of that. It is the
kNN-LM construction, it changes nothing inside the model, and it bounds what
any retrieval system built on the same index can deliver. If it does not
lower held-out BPB, nothing built on top will.

It is also cheap in an unobvious way. Mixing two distributions and reading
the result at the true byte needs only both of their values there:

```
p = (1-lambda) * p_model + lambda * p_retrieval
```

so the model never has to emit a full 256-way distribution. Its per-byte
loss is `-log p_model(y_true)`, which the evaluator already computes. That
is what `lossout` writes.

## The index

An n-gram store at several context lengths, queried longest first - the
infinite-gram construction. Each entry packs a 56-bit hash of the context
with the byte that followed it into one u64, so sorting groups equal
contexts *and* orders their continuations within the group. Both the total
count and the count for one specific byte are then binary searches rather
than scans, which matters: an eight-byte context in a hundred-megabyte store
occurs millions of times, and scanning per query would dominate everything.

Two retrieval distributions are scored side by side because they are
different claims:

- **longest**: raw maximum likelihood at the longest context that occurs.
  Sharp, and zero for anything unseen, so it cannot stand alone.
- **backoff**: the recursive interpolation
  `p_m = (c_m(ctx,y) + a*p_{m-1}) / (c_m(ctx) + a)` from a unigram upward.
  No zeros, and a language model in its own right.

## Status

Built: `lossout` in `train` (per-byte cross entropy in corpus order) and the
`ngram` tool. Both committed.

**Calibration, not yet a result.** A first pass with a 50 MB datastore
against an old checkpoint that evaluates at 5.27 BPB:

```
lambda   longest-match mix
  0.00   5.2730
  0.50   2.9447
  0.80   2.8077   <- best
  1.00  11.7565   (the zeros)
```

This says almost nothing about G. The model was bad, and the number mostly
measures the n-gram: at lambda 0.9 the mix is 2.84, so **an infinite-gram
over 50 MB of Wikipedia bytes is worth roughly 2.8 BPB on its own**. That is
the useful part of this run - a calibration of the retrieval component
against which the real question can be posed.

The real question is what it adds to a model that is *already better than
it*. A baseline at 1.4 BPB and a retrieval component at 2.8 overlap on
exactly the content the model has already learned; the gain comes only from
where they disagree and the retrieval is right.

Baseline training is under way at dim 512, 16 layers, `fp8=1`, on enwik9;
it passed 1.362 BPB at step 17860 before being paused.

## What would make this fail

- The offset is small. If interpolation buys 0.05 BPB against a flattened
  baseline, the speedup is a small factor and the section is over.
- The gain is all on markup. Wikipedia bytes include a great deal of XML and
  template boilerplate, which an n-gram predicts almost perfectly and which
  is not what anyone means by knowledge. **Score prose separately** before
  believing any headline number - the split is worth building into the
  evaluation rather than arguing about afterwards.
- The index cannot be afforded at step time. Interpolation is free here only
  because it happens offline against a loss file. A system that retrieves
  during training pays bandwidth it does not have.

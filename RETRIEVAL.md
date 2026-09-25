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
`ngram` tool, which takes several loss files at once so a whole training
curve is scored against one build of the index.

### First result, against a model better than the index

Checkpoint at step ~20000 of the fresh baseline, 2 M held-out bytes, a
50 MB datastore - one nineteenth of the corpus:

```
                  BPB      vs model
model alone     1.8853
best mix        1.6869      -0.1984   (-10.5%)   at lambda 0.45
```

Lambda was chosen on a disjoint tuning half, which picked **the same 0.45**,
so the number is not the product of tuning on what it reports.

The split that mattered most:

```
                model   mixed
prose (73%)    1.9433  1.7660    -0.1772
markup (27%)   1.7289  1.4736    -0.2553
```

Markup gains more, as expected - and prose gains nearly as much. **This is
not an artefact of the dump's format**, which was the single largest risk to
the whole section.

98.7% of bytes had some context in the store. Longest-match lengths: 7.6% at
32 bytes, 19.1% at 16, 21.7% at 12, 30.1% at 8, 20.2% at 4.

### It scales with the datastore

Same checkpoint, same lambda, varying only how much of the corpus is
indexed:

```
 12 MB   1.8853 -> 1.7816   -0.1037   97.0% covered
 25 MB   1.8853 -> 1.7275   -0.1578   98.1%
 50 MB   1.8853 -> 1.6869   -0.1984   98.7%
```

Every doubling of the store is worth roughly another -0.05 BPB, decreasing
slowly. Extrapolating that to the full 950 MB puts the gain near -0.3.

### Calibration

An infinite-gram over 50 MB of Wikipedia bytes is worth about 2.8 BPB on its
own - worse than the model it is being mixed into, which is the point: the
gain comes from where the two disagree and the index is right.

### The speedup, measured

One doubling of training multiplies held-out BPB by 0.9767 at this point on
the curve - measured, not assumed:

```
step 20000   1.8593
step 40000   1.8159
```

Against that ruler, with a 50 MB store:

```
step    model    mixed    offset    baseline doublings to match    speedup
20000   1.8593   1.6783   -0.1810              4.34                 20.2x
40000   1.8159   1.6535   -0.1624              3.97                 15.6x
```

**The offset shrinks as the model trains** - 0.1810 to 0.1624 over one
doubling. The model learns part of what the index supplies. So a single
measurement produces a number that decays, and quoting one without the
budget it was taken at is exactly the error this file warned about before
any of it was measured.

What is honest to say: **retrieval is currently worth 15 to 20 times the
training compute, and the factor declines slowly.** With a 100 MB store the
offset is a quarter larger, so the factor is higher again.

### Caveats that travel with those numbers

- Two points on the curve. The slope rests on a single doubling, and the
  speedup is an extrapolation of four doublings from it. More points are
  coming from the running baseline; the factor should be re-read then.
- The offset declines, so this cannot be extrapolated far. Whether it
  approaches a floor or keeps decaying is the next thing to measure, and it
  decides whether retrieval is a constant advantage or a head start.
- This is "retrieval replaces training", not "training runs faster". The
  index has to be there at inference, and it is 950 MB of corpus.

## Withdrawn: the speedup numbers above

Every speedup in this file - 20x, 15.6x, the seven-point fit's 5.3x and the
~3x plateau - was read off a baseline that trained at **10% of its peak
learning rate from step 10000 on**. `decaysteps` was not set, its default
is 8000, and the cosine finished there. The curve between steps 20000 and
88000 is therefore a crippled schedule creeping along, not what the model
does with compute, and a flat curve makes any constant offset look worth
many doublings. This is exactly the failure the external review predicted
under its first point.

The retrieval *offsets* - how much the index lowers held-out loss at a given
checkpoint, the prose/markup split, the near-duplicate analysis, the scaling
with the store - stand, because they compare two predictors on the same
checkpoint. What does not stand is the conversion of an offset into a
speedup.

The baseline is being redone as warmup-stable-decay: a nearly constant rate,
raw and averaged weights kept at 24k, 48k, 100k, 200k, 400k and 480k steps,
and a cooldown branch from each, so every point on the compute axis is a
fairly cooled model. The retrieval mix is then measured at each of them.
Script: `~/tmt-scratch/wsd/run.sh`.

## Runbook, for when the GPU is free again

The baseline was paused at step ~17900; `train` resumes from an existing
checkpoint path, so step 1 continues rather than restarts.

```bash
cd ~/Schreibtisch/test-model-thing/tmt-cuda

# 1. resume the baseline, keeping a checkpoint series beside it
nohup ./zig-out/bin/train ~/tmt-data/enwik9/train.bin \
  ~/tmt-data/runs/retr-base/base.ckpt \
  dim=512 layers=16 experts=8 topk=2 batch=64 seqlen=128 traces=1 \
  half_max=65536 warmup=2000 aux=0.1 fp8=1 steps=300000 saveevery=4000 \
  > ~/tmt-scratch/retr/base.log 2>&1 &
nohup ~/tmt-scratch/retr/keep.sh > ~/tmt-scratch/retr/keep.log 2>&1 &

# 2. split the held-out data: tune lambda on one half, report on the other
cd ~/tmt-scratch/retr
head -c 4000000 ~/tmt-data/enwik9/held.bin > tune.bin
tail -c +30000001 ~/tmt-data/enwik9/held.bin | head -c 4000000 > test.bin

# 3. per checkpoint, dump losses and mix
for c in ~/tmt-data/runs/retr-base/s*.ckpt; do
  ./zig-out/bin/train ~/tmt-scratch/retr/test.bin "$c" mode=eval \
      lossout=~/tmt-scratch/retr/$(basename $c).f32
  ./zig-out/bin/ngram ~/tmt-data/enwik9/train.bin ~/tmt-scratch/retr/test.bin \
      ~/tmt-scratch/retr/$(basename $c).f32 store=200 ms=32,24,16,12,8,4,2 \
      limit=2000000 maxgb=12
done
```

Step 3 is GPU-light (one eval per checkpoint) and CPU-heavy (the index is
rebuilt each time). Building the index once and reusing it across
checkpoints is the obvious next change to `ngram` if this becomes the
bottleneck.

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

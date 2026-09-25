# NVFP4

What is built, what it measures, and the recipe it has to follow. The recipe
is not guesswork: it is from *Pretraining Large Language Models with NVFP4*
(arXiv 2509.25149), which trained a 12B model on 10T tokens and matched an
FP8 baseline - MMLU-pro 62.58 against 62.62.

## The format

e2m1 - four bits, eight magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6 - with one
e4m3 scale per sixteen values, and **one fp32 scale for the whole tensor**.

That last one is not optional and is the first thing to get wrong. e4m3
spans 2^-6 to 2^8, so a block scale below 0.016 underflows; weights of
magnitude 0.02 want block scales near 0.008. Without the outer scale every
block quantises to zero. The paper computes it as `s_enc = 6*448/amax`.

## The recipe, exactly

| | |
|---|---|
| Fprop | both operands NVFP4 |
| Dgrad | both operands NVFP4 |
| Wgrad | both operands NVFP4 |
| Random Hadamard | **Wgrad inputs only**, 16x16, one random sign vector shared by every layer and fixed for the whole run. Transforming Fprop or Dgrad inputs *degrades* quality. |
| Stochastic rounding | **gradients only**. On forward tensors it is detrimental; weights and activations use round-to-nearest-even. |
| Weight scaling | **2D, 16x16 tiles** - 16 input channels by 16 output channels - with the tile's scale replicated into each 1x16 block for the tensor cores. A weight is used untransposed in Fprop and transposed in Dgrad, and a scale grouped along one axis only is consistent in one direction. |
| High precision | first 2 and last 8 transformer blocks in BF16, 16% of linear layers. The paper calls this conservative and reports stability with only the last 4. |

One of these was measured here before it was read: stochastic rounding made
the forward matmul worse, 0.195 to 0.244 relative error, which is the
paper's own finding arrived at from the other direction.

## What is built

`zig/kernels/kernels.zig`: `fp4_quant` (1D blocks, with round-to-nearest or
stochastic rounding and optional random Hadamard), `fp4_quant_w2d` (the 2D
weight form), `fp4_dequant`, `fp4_absmax`.

`zig/model/fp4.zig`: the cuBLASLt matmul, single and strided-batched, with
the layout that matches `linalg.linearFwd` so it is a drop-in.

`zig/fp4test.zig`: the measurements below.

## Measured on the 5070 laptop at 115 W

Speedup over the bf16 matmul it replaces:

```
shape (M,N,K)        bf16 us   nvfp4 us   speedup
 1024, 512, 512         16.1        9.0     1.79x
 2048, 512, 512         28.7        9.6     3.00x   <- the expert shape
 4096, 512, 512         50.9       12.3     4.15x
  512, 512,2048         26.6        6.4     4.14x
 2048,2048,2048        363.6       66.4     5.48x
```

Quantiser round-trip, relative L2: 0.095 gaussian, 0.068 gaussian with
outliers, 0.109 lognormal - the expected range for four bits with sixteen
element blocks.

The Hadamard transform earns its place only where it is meant to. On
gaussian data with outliers it halves the matmul error, 1.81 to 0.85. On
uniform data it makes things worse, because rotating uniform values produces
gaussian-ish ones with heavier relative tails against a fixed eight-level
grid.

Batched matmul: algorithms exist for batch 2, 8 and 16, which is what the
expert path needs.

## Three traps, each of which cost a debugging round

**The heuristic reports NOT_SUPPORTED until the scale *pointers* are set**,
not merely the scale modes. That alone looks exactly like "this GPU cannot
do FP4".

**cuBLASLt applies the e4m3 block scales and nothing else**, so the outer
fp32 scale has to come back through `alpha`. Leaving it out made the result
too large by 1/(gA*gB) - a factor of 8e9, which is how it was found.

**0x7F is NaN in e4m3** and the encoder could emit it at the top of the
exponent range. A NaN block scale turns the whole matmul into NaN. It only
appeared once the Hadamard rotation started pushing block maxima against the
ceiling, so it would have shipped quietly otherwise.

And one about measurement rather than the format: the first benchmark used
uniform data, on which neither stochastic rounding nor a rotation can help.
It reported both as harmful, correctly and uselessly.

## What is not built

Integration into the training path. That means: a config flag, quantising
the expert weights after each optimizer step and the activations each
forward, routing `linearFwd`/`linearDX`/`linearDW` through `fp4.Gemm`,
applying the three modes by tensor role rather than globally, and excluding
the first and last blocks.

MXFP4 is not available through cuBLASLt on sm_120 - zero algorithms - though
the PTX assembles, so it would need hand-written MMA. NVFP4 does not.

The optimizer stays out. Adam writes about `lr*m/sqrt(v)` per step, roughly
2.5% of a weight's magnitude, against e2m1's ~15% relative resolution: the
update would round away entirely. Stochastic rounding removes the bias, not
the variance. The master copy is where the precision has to live, and bf16
with stochastic rounding is already the floor.

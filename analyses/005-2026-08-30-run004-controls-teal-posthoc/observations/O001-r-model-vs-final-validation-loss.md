# O001 - TEAL-style R_model versus final validation loss

## Question

How does uniform TEAL-style post-hoc magnitude clipping change measured
`R_model` versus complete-validation loss for the full-pass Pythia-14M GeLU and
ReLU controls used in Analysis 003 Figure 1?

## Sources and coverage

- Run 004's valid verified GeLU attempt
  `001-20260829-221007-bb5288c8` and ReLU attempt
  `002-20260829-231305-195b22c6`, each at its retained step-712 checkpoint.
- Checkpoint content SHA-256 identities, source manifests, Run 004 verification,
  and both MiniPile token-cache byte counts and SHA-256 identities are verified
  before evaluation.
- Per-matrix thresholds are calibrated separately for `a`, `m`, `h`, and `z`
  in each of six blocks on the first 10 complete source-order training blocks
  (20,480 input tokens). Calibration and validation use disjoint dataset splits.
- Uniform target sparsity `p` varies over `{0.0, 0.1, ..., 0.9}`. For each
  matrix input, the threshold is the smallest empirical absolute-value order
  statistic reaching the target, and evaluation zeros `abs(x) <= t`.
- Every endpoint uses FP16 CUDA autocast with FP32 parameters and covers all 500
  MiniPile validation documents, all 338 complete 2,048-token blocks, and
  692,224 input tokens; the 1,444-token tail is excluded and reported.
- One initialization/data-order seed is evaluated. There is no replicate or
  calibration-sample uncertainty estimate.

`01_evaluate.py` performs the hash checks, calibration, complete evaluation,
activation capture, and six-operation logical-product capture. It durably
records every point before publishing `teal_frontier.json`. `02_plot.py`
rejects incomplete coverage or unreconciled counts, derives the global
nondominated envelope, writes `tables.md`, and reads only the verified reduced
artifact for plotting.

## Figure caption and legend

**Figure 1. Final validation quality versus measured model-level logical
opportunity for uniform TEAL-style post-hoc clipping of the Run 004 controls.**
Blue circles and a solid line show the GeLU checkpoint; orange squares and a
dashed line show the ReLU checkpoint. Labels give the uniform per-matrix target
sparsity `p`. Filled points lie on the global nondominated envelope over all 20
points under lower loss and higher `R_model`; open points are dominated. The
black dotted line traces that envelope. Panel (a) shows the complete sweep and
panel (b) is an explicitly labeled low-loss detail through target `p=0.5`.
Horizontal position is measured exact-zero `R_model` as a percentage, a logical
zero-product opportunity rather than measured speedup. Vertical position is
loss over all 338 complete validation blocks; lower is better. Lines connect
increasing targets for readability and do not imply interpolation. No
uncertainty bars are shown because only one seed was evaluated.

## Observed pattern

Both curves are monotonic over the tested grid: increasing `p` increases both
measured `R_model` and validation loss. The most conservative GeLU target,
`p=0.1`, moves from effectively zero to `1.278706%` `R_model` with a paired loss
increase of `+0.005859`; `p=0.2` reaches `2.554786%` with `+0.041676` loss.
For ReLU, the target-zero point already has `2.714133%` `R_model` from natural
`h` zeros. Its `p=0.1` and `p=0.2` points reach `3.568403%` and `4.418365%`,
with paired loss increases of `+0.004439` and `+0.038307`.

At common target `p=0.4`, GeLU gives loss `5.605384` and `5.125544%`
`R_model`; ReLU gives loss `5.602965` and `6.129325%` `R_model`. ReLU's `h`
exact-zero fraction stays near its natural `63.45%` through this target, while
`a`, `m`, and `z` reach about `39.6%`, `39.6%`, and `41.8%` exact zeros. Thus
the mild-target gains combine new clipping at the other matrix inputs with
natural ReLU sparsity at `h`.

The global envelope contains GeLU targets `0.0--0.2`, ReLU targets `0.0--0.6`,
both controls at `0.7` and `0.8`, and GeLU at `0.9`. GeLU targets `0.3--0.6`
and ReLU target `0.9` are globally dominated on these two coordinates. The
high-target points exceed loss `8`; they establish the full measured curve but
are not low-degradation operating points.

## Caveats and nonclaims

This is a descriptive single-seed, single-scale comparison of two separately
trained activation checkpoints. A same-target GeLU/ReLU difference does not
isolate the activation function because the final learned parameters differ.
The deterministic 10-block calibration sample is not varied. TEAL's original
setting uses separate matrices in Llama-like blocks and focuses on batch-one
autoregressive decoding; Pythia uses a fused QKV matrix and this analysis uses
full-sequence pretraining loss. This is uniform target allocation, not TEAL's
optional block-wise greedy allocation. Logical zero products are not removed
FLOPs or measured runtime acceleration. The arithmetic frontier has no
replicate uncertainty and is not promoted to a finding or manuscript claim.

## Provenance

- Evaluation: `../01_evaluate.py`
- Plot and frontier reduction: `../02_plot.py`
- Reduced artifact: `../teal_frontier.json`
- Generated table: `../tables.md`
- Figure: `../figures/01-r-model-vs-final-validation-loss.pdf`

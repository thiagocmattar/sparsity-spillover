# Figure 05-v3: signed activation distributions

Status: design approved on 8 September 2026; the user selected RunPod because
the local GPU is busy. Implementation and CPU verification are in
`runs/031-2026-09-08-signed-activation-density/`. The concrete cloud launch
envelope awaits the separate launch confirmation. No Pod or full measurement
has started. This design addresses signed densities; the earlier
ACTIVATION-DIAGNOSTIC.md concerns finer magnitude bins.

## Question and manuscript purpose

How do the signed activation distributions differ between A0, A4-OL1 and
A7-OL1 as the trained threshold increases? Figure 05-v3 should make their
distribution reshaping visible in the activation case study, using full
distributions rather than the two summary fractions of Figure 05-v2.
The original Figure 05 and Figure 05-v2 remain available. No manuscript
claim or text is changed by this measurement.

## Retained evidence and missing information

The seven selected checkpoints all exist locally (56,279,344 bytes each).
Their activation diagnostics retain moments, exact-zero counts and cumulative
magnitude counts at .001/.01. They contain no signed histograms or raw samples.
Neither these summaries nor Run 030's additional clipping summaries determine
a signed density. A fresh diagnostic pass is necessary; fitted distributions
must not stand in for measurements.

## Fixed models and evaluation

Use the same seven Pythia-14M step-712 checkpoints as O005: A0 from Run 004,
A4-OL1 from Run 015, and A7-OL1 from Run 014 at kappa=0,.05,.5. Exact source
attempts and source identities are in figure_data.json. The conditions were
randomly initialized, with model/data-order seed 1234, 712 AdamW updates and
1,493,172,224 training input tokens. This new task loads their final weights
for evaluation; there is no optimizer, backward pass, continuation training,
new seed, or new post-hoc clipping.

Retain the saved gate configurations: A0 stock GELU; A4 one-sided G+ at
a,m,h,z; A7 the same G+ sites plus symmetric Gpm at post-RoPE q,k and v.
The OL1 conditions were trained with pressure at their gated sites. Pressure
is inactive during evaluation. Capture h,m,q_post,k_post,v after configured
gates, using their actual signed values without RMS normalization.

Use FP32 weights, FP16 CUDA autocast, eager uncached attention, batch one,
T=2048 and the pinned MiniPile validation cache. Cover all 500 documents,
338 complete blocks and 692,224 input tokens per checkpoint, excluding the
1,444-token tail. Seven passes total 2,366 block evaluations. Recompute loss
and the original activation moments/counts to reconcile with retained results.

## Signed measurements and aggregation

Collect per-site/layer signed histogram counts on a common grid with .001
bin width over [-8,8], plus underflow/overflow counts and extrema. Record
exact-zero counts separately, and retain the original near-zero thresholds,
finite/nonfinite counts and RMS/L2 accumulators. The grid includes the gate
boundaries 0, +/-.05 and +/-.5. Verify boundary equality and dtype handling;
never interpolate or smooth across the excluded gate region.

Pool integer bin counts and totals across layers, blocks and sites, then
divide. FFN activations pool h,m: this gives h an 80% and m a 20% element
weight because h has four times as many coordinates. Attention activations
pool q,k,v with equal element weights. Retain individual site/layer counts
so pooling remains auditable. These are distributions of activation elements,
not an equal-site mixture or an average of site densities.

For nonzero bins, density is count/(total activation count * bin width).
Its area is the nonzero probability mass within the displayed range, not
renormalized to one. Show exact-zero probability separately in each panel;
zero is a point mass and must not become a broad artificial density peak.
Retain and disclose any mass outside the displayed x range. Preserve complete
histogram counts, checkpoint/cache/code identities and validation coverage
in a compressed structured artifact; do not archive full activation tensors.

## Figure

Save a separate figures/05-v3-activation-density-grid.pdf in this analysis.
Use three rows (kappa=0,.05,.5) and two columns: FFN activations and Attention
activations. Signed activation x is horizontal and density is vertical.
Overlay A0, A4-OL1 and A7-OL1 with the existing colors, colored density outlines
and low-opacity fills. Use common binning and comparable axes across rows;
the exact x view is chosen from measured tail coverage, not estimated moments.
Keep the existing title font/size, thin grey grids and a compact legend below.

FFN panels mark +kappa, applicable to A4/A7. Attention panels mark both
-kappa and +kappa, explicitly for A7: A4 does not gate q,k,v. At kappa=0,
the two symmetric guides coincide. Markers refer to saved train-time gate
settings; no additional clipping is applied during these measurements.

## Interpretation, checks and execution boundary

Shifts in signed density, depleted gate regions and increased zero mass would
support a descriptive reshaping result. Similar distributions or contrary
changes would limit that result. Pooling can hide opposing site-level changes;
retained site counts allow checking that limitation. The complete recipes
change both gates and pressure sets, so this does not isolate their causal
contributions, demonstrate a training trajectory, or establish speedup.

Run 031 tests signed-bin accounting, zero exclusion, boundary equality, exact
hooks, group weights, coverage and serialization. Local checks are CPU-only.
The RunPod launch includes a three-checkpoint, eight-block CUDA calibration
before full evaluation; use its measured timing and peak memory to determine
whether the full pass fits the approved envelope. The separate launch proposal,
tests, current prices and transfer inventory are in Run 031's README. The
existing checkpoints and cache remain retained throughout.

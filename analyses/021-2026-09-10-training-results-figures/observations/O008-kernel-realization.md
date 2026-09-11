# K050 native-relative speedup and exploitable projection sparsity

## Question and scope

How does model-wide sparsity relate to native-relative full-model speedup, and
how does projection MMA bypass relate to the gain from enabling projection
sparse paths? This author-approved revision replaces the cohort-ablation
lollipop with a second 30-checkpoint scatter after the
[investigation](../investigation/README.md). It remains **analysis-only**;
no manuscript changes, model evaluation or benchmarking were performed.

## Sources, method and coverage

- [Analysis 018 figure data](../../018-2026-09-08-results-materials/figure_data.json)
  and [Run 029 retrospective](../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/results/matched-retrospective-001.json)
  provide the qualified c01-c30 cohort, canonical integer sparsity counters and
  all three implementations' replicate identities. Their counts, speedups and
  source hashes are reconciled as before.
- The [investigation checkpoint table](../investigation/data/checkpoints.json)
  supplies projection counters and raw geometric-mean latencies; its
  [analysis](../investigation/data/analysis.json) independently supplies the
  projection regression. The figure reduction matches checkpoint identities,
  full speedups, all-skips-off and attention-dense paired speedups, then derives
  the projection ratio from the raw candidate latencies.
- Both panels contain the same 30 Pythia-14M checkpoints, ten per visual family.
  Historical h-only A4 pressure checkpoints are excluded. All 90 underlying
  implementation/checkpoint comparisons qualify with three complete processes.
- Timings use RTX5090, BF16, batch one, uncached 2,048-token inference with all
  50,304 output logits. Each process times the same 64 validation inputs with
  seven paired native/candidate passes; three fresh processes are geometrically
  pooled. Native-relative speedup is native PyTorch/SDPA CUDA-graph latency
  divided by full K050 latency for that checkpoint.
- Canonical S_model uses FP16 integer zero-product counts divided by the common
  block-plus-output-head denominator, excluding future-masked attention pairs.
  Qualification and counters cover all 338 complete validation blocks from 500
  documents, with 692,224 input tokens and the 1,444-token tail excluded.
- Projection bypass pools bypassed/potential MMA instructions over QKV,
  attention output, FFN-up and FFN-down. The underlying full-validation BF16
  diagnostics reconstruct QKV/FFN-up A-fragment decisions and instrument the
  FFN-down/output hybrid kernels. The latter include padded instruction work
  and replacement by scalar/SIMT products. MMA bypass is not eliminated FLOPs.
- Projection-path gain is the all-skips-off candidate geometric-mean latency
  divided by the attention-dense/projection-on candidate geometric-mean latency.
  The implementations retain the same fusion and use the matched input/timing
  protocol in separate processes. This is the investigation's raw-latency
  ratio, not the ratio of separately native-normalized speedups formerly used
  for the lollipop increments. Both fits are unweighted OLS with intercepts.

## Figure and proposed caption

[Publication PDF](../figures/07-kernel-realization.pdf)

**Model-wide sparsity tracks native-relative speedup, while projection
instruction bypass tracks projection-path gain.** (a) Full K050 speedup
relative to each checkpoint's native PyTorch/SDPA implementation versus
model-wide sparsity for all 30 Pythia-14M checkpoints. The labeled seven-site
+ OL1 checkpoint at ? = 0.5 achieves 1.78? speedup. (b) Gain from enabling
projection sparse paths versus the fraction of projection matrix-multiply-
accumulate (MMA) instructions bypassed: gain is all-skips-off latency divided
by projection-on/attention-dense latency, with identical fusion. Gray circles,
blue diamonds and orange triangles identify baseline/local, four-site and
seven-site recipes in both panels. Dashed lines are descriptive OLS fits
(R? = 0.817 and 0.946); dotted lines mark gain 1?. Projection bypass includes
SIMT substitution and padded h/z instruction work, so it is not a fraction
of arithmetic eliminated. Canonical sparsity uses FP16, whereas the kernel
counters and timings use BF16. Counters cover 338 validation blocks; timings
use the matched 64-input subset. Associations across distinct trained
checkpoints do not identify causal effects or equal-quality acceleration.

The 7.4-by-3.4-inch PDF uses a 60:40 horizontal layout and one shared legend
below both panels. Only the highest-sparsity checkpoint is labeled; regression
equations, recipe-level legends, lollipops and additional diagnostics are absent.
Axes show all points, with small left margins so zero-valued markers remain
visible. The small projection-axis explanation names instructions bypassed,
rather than implying that all bypassed work disappears.

## Results and interpretation limits

Panel (a): Pearson r = 0.903724 and R? = 0.816718. Panel (b): r = 0.972679 and
R? = 0.946105. All checkpoint values remain unchanged from the investigation.
The c30 endpoint is 27.482684% S_model and 1.783175? native-relative speedup.

Native references differ across recipes. In the matched positive-? pairs,
seven-site has a larger native-relative speedup but a higher absolute K050 latency. The figure's explicit native-relative label
preserves that distinction; the investigation retains absolute latencies,
within-family fits, scalar-projection comparisons and attention diagnostics.
Projection instruction bypass combines different sparse execution paths,
including SIMT substitution. Its correlation with gain is implementation- and
cohort-specific, not a general causal law or an inference claim for other
model sizes, hardware or cached decoding.

## Reproduction and verification

Source script: [07_kernel_realization.py](../07_kernel_realization.py).
The [reduction](../data/kernel-realization.json) retains source hashes, all 90
qualified timing comparisons, the 30 projection points with integer counters
and raw latency denominators, and both fitted coefficient sets.

```powershell
.venv/Scripts/python.exe -X utf8 analyses/021-2026-09-10-training-results-figures/07_kernel_realization.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_kernel_realization.py -q
```

Three focused tests pass: complete cohort/count reconciliation, independent
OLS and all 30 plotted points in each panel, and projection gain from matched
raw latencies with integer-pooled instruction counts. The PDF was rendered
and visually checked. The manuscript remains unchanged.

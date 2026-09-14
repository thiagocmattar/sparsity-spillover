# O014 - Figure 07 v3: one native A0 reference

## Question

How does model-wide sparsity relate to K050 full-model speed when all 30
checkpoints share the same native A0 latency reference? The author requested
this separate v3 after identifying that recipe-specific native overhead changes
the interpretation of v2's speedup ratios. This is retained-data analysis only.

## Method and coverage

Panel (a) plots `T_native_A0 / T_full_K050_checkpoint`. Its fixed numerator is
**0.65544239800561 ms**, the geometric mean of the c01/A0 native controls in
the full-K050 timing processes (`native_baseline_gm_ms`, equal to
`full_native_gm_ms` in the investigation). Every denominator is the original
checkpoint's `full_candidate_gm_ms`. These are geometric means of 1,344 host
latencies: 64 validation inputs x seven passes x three processes. We do not
average native references from the other ablation processes or force the
optimized A0 point to 1. The latter is 1.005938x against native A0.

The same 30 qualified Pythia-14M checkpoints, BF16 batch-one uncached
2,048-token inference, full-vocabulary output, RTX5090, and timing inputs are
retained. Model-wide sparsity uses the original pooled integer FP16 counters
over all 338 complete validation blocks (692,224 tokens from 500 documents;
the 1,444-token tail is excluded). No training, evaluation or timing is run.

Panel (b) is unchanged: all-skips-off K050 latency divided by
projection-on/attention-dense K050 latency **for the same checkpoint**.
Changing this ablation's reference to A0 would cease to measure the gain from
enabling projection skipping. Its points, four threshold curves, two local
pressure-weight paths, marker styles, annotations and absent regression are
preserved exactly.

## Legend and proposed caption

**Full-model speed relative to a common native A0 reference.** (a) Model-wide
sparsity versus native A0 geometric-mean latency divided by each checkpoint's
full K050 geometric-mean latency. All 30 points share the same 0.65544 ms
numerator; the dotted line is native A0 execution time. The dashed line is a
descriptive OLS fit (R-squared = 0.510). Seven-site + OL1 at kappa = 0.5 reaches
1.38x against this reference. Black squares identify A0, gray circles the
one-site recipes, blue diamonds four-site recipes and orange triangles
seven-site recipes. (b) Matched projection-skipping gain versus model-wide
sparsity, with the same definition as v2. For four-/seven-site recipes, solid
lines/open markers denote no OL1 and dashed lines/filled markers denote OL1;
muted gray paths connect local L1/OL1 weight settings from ReLU. Lines connect
separately trained conditions. The shared A0 reference compares execution
times across different trained models, not equal-quality or isolated
sparsity-induced acceleration.

## Results

- Full-model association: Pearson r = **0.714084**, OLS R-squared = **0.509917**
  (previous same-checkpoint native-reference R-squared = 0.816718).
- At kappa = 0.5 with OL1, four-site c20 reaches **1.426501x**, whereas
  seven-site c30 reaches **1.384642x**. Their original full K050 latencies are
  0.459475664 and 0.473366013 ms, respectively.
- Four-site c20 is the highest-speed point. Seven-site c30 remains the
  highest-sparsity point, and its retained annotation identifies that condition
  without claiming it is the fastest.
- The common-reference ratios range from 1.005938x to 1.426501x. A fixed
  numerator makes their ordering exactly inverse to optimized absolute latency.

## Caveats

The reference removes varying native checkpoint cost from the plotted ratio.
It does not separate fusion, dense implementation changes and sparse skipping,
or match validation quality across checkpoints. A0 and each candidate were
timed in separate retained processes; these are not newly collected,
interleaved common-A0/candidate timing pairs. No uncertainty or causal claim is
added. Panel (b)'s matched ratio remains the more specific projection-ablation
measurement. The manuscript still contains v2 and its old reference definition;
this task does not adopt v3 or rewrite those claims.

## Files, reproduction and verification

- Figure: [07-kernel-realization-v3.pdf](../figures/07-kernel-realization-v3.pdf).
- Source script: [07_kernel_realization_v3.py](../07_kernel_realization_v3.py),
  reusing the existing v2 layout without changing either previous builder.
- Reduction: [kernel-realization-v3.json](../data/kernel-realization-v3.json).
  It retains source hashes, original native ratios, an explicit common A0
  reference, all 30 new ratios and the refitted OLS. The previous PDF hashes
  remain recorded; original and v2 outputs are preserved.
- Evidence: [investigation checkpoint table](../investigation/data/checkpoints.json)
  and [timing definitions](../investigation/METHODS.md).

```powershell
.venv/Scripts/python.exe -X utf8 analyses/021-2026-09-10-training-results-figures/07_kernel_realization_v3.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_kernel_realization_v3.py analyses/021-2026-09-10-training-results-figures/test_kernel_realization_v2.py -q
```

All five focused tests pass. They check the single A0 numerator, all 30
denominators and plotted coordinates, inverse-latency ranking, independent OLS
recalculation, unchanged source data, and unchanged panel (b) paths/markers.
The single-page PDF was rendered and visually checked; text fits within the
page and fonts are embedded. No manuscript file is modified.

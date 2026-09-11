# O001 - K050 projection structure and the native reference

## Question

Why can seven-site checkpoints have larger full-model speedup despite a lower
projection contribution to S_model? The author requested this retained-data
investigation on 11 September 2026. The original investigation was analysis-only;
[O010](../../observations/O010-kernel-manuscript-adoption.md) records the later
approved manuscript adoption and supplementary-manifest provenance refresh.

## Method and coverage

Trace the frozen K050 dispatch and sparse predicates; reconcile the existing
Analysis 018 and Run 029 cohort with the manuscript supplementary runtime data.
Pool the six operation families and structural counters from 30 complete BF16
diagnostics, each covering all 338 validation blocks and six layers. Reconstruct
latency geometric means from 270 qualified timing files: 64 identical validation
inputs, seven paired passes and three fresh processes per implementation.
Canonical logical sparsity remains the archived FP16 measure. Compute matched
latency gains, all ten four-/seven-site contrasts, and Pearson/OLS fits across
the cohort and its three recipe families. See [METHODS](../METHODS.md) for code
links, denominator definitions and unavailable quantities.

## Figure and caption

[Five-page diagnostic figure](../figures/kernel-investigation.pdf).

**K050 speedup depends on structural execution and its native reference.**
Page 1 compares the five requested scalar/counter predictors with full-model,
projection and attention gains, and additionally shows absolute K050 latency.
Each point is one checkpoint; gray-green circles identify baseline/local
recipes, blue diamonds four-site recipes and orange triangles seven-site
recipes. Dashed lines are unweighted OLS fits with intercepts; the attention
MMA predictor has only two distinct values, with c30 alone at the higher value.
Page 2 shows within-family S_model/speedup fits, with separate axes to expose
variation. Pages 3 and 4 compare four-site with seven-site at the five matched
thresholds, without pressure and with OL1 respectively. Lines connect trained
conditions, not trajectories or interpolated checkpoints. Page 5 separates
native, all-skips-off and full-K050 absolute latencies; column ranges differ
but are shared between pressure settings. Gains are ratios of reconstructed
geometric-mean host latencies. Projection bypass includes SIMT substitution
and h/z padding; attention counts include masked/padded instruction work.
The 338-block counters and 64-input timing subset are distinct estimands.
These are descriptive associations, not causal estimates or seed uncertainty.

## Result and caveats

In all eight positive-threshold pairs, seven-site has higher relative speedup
but higher absolute K050 latency. At κ = 0.5 with OL1, four-/seven-site latencies
are 0.45948/0.47337 ms, while native latencies are 0.73944/0.84409 ms; their
speedups are 1.609/1.783. Four-site has more projection MMA bypass in every OL1
pair. Only c30 has additional attention MMA skips beyond the common padded-PV
baseline, and enabling attention skipping is slower for every checkpoint.

The [report](../README.md) gives the conclusion, measured/reconstructed/missing
distinctions and full association summary. Counters restricted to timing inputs,
pure-zero versus SIMT-replacement bypass, unmasked-only MMA counts and isolated
operator timing require additional instrumentation; no such measurements were
invented or launched.

## Sources and verification

- [01_extract.py](../01_extract.py): cohort, source hashes, scalar/counter pooling and raw timing reconstruction.
- [02_analyze.py](../02_analyze.py): matched pairs and descriptive fits.
- [03_plot.py](../03_plot.py): all five PDF pages.
- [test_investigation.py](../test_investigation.py): five focused tests covering exact retained evidence, timing pairs/GM, denominator conservation, missing data, matched latency versus speedup, and constant-predictor handling.

All five tests pass. All five PDF pages were rendered and visually inspected.

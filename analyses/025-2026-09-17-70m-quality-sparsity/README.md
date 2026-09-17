# Analysis 025 - All 70M trained quality-sparsity endpoints

The user requested a plot of model-wide sparsity versus validation loss for
all available 70M results on 17 September 2026. This analysis includes all
22 completed training conditions from Runs 018 and 034, using existing artifacts.

[Publication figure](figures/01-70m-quality-sparsity.pdf) |
[Full-precision plotted data and source hashes](data/70m-quality-sparsity.json) |
[Observation and caption](observations/O001-70m-quality-sparsity.md)

Included: GeLU baseline, ReLU, A4 + OL1(all), A4 + OL1(h), A7 + OL1(all),
and A7 + OL1(h). Each OL1 family contains kappa = 0, 0.01, 0.05, 0.1, 0.5.
There are no completed pressure-free A4/A7 or ReLU+OL1 training grids at 70M.
Runs 016/017 stopped before scientific training and supply no plotted endpoint.
Kernel runs are measurements of existing checkpoints, not additional trained
models. Post-hoc clipping is outside this trained-endpoint overview.

All points use ordinary reloaded final-checkpoint validation loss from
`metrics.validation.final.loss`. Logical diagnostic pass loss is retained
separately in the reduction for audit, and is never substituted into this plot.
The x coordinate is `100 * R_model`, recomputed from pooled integer zero-product
and full-model product counts. It includes the dense LM-head denominator;
it is not mean site sparsity, an architectural ceiling, or measured speedup.

The canonical initialization, data-order hash, seed 1234, 712 updates and
1,493,172,224 training input tokens are matched. Every evaluation covers all
500 validation documents / 338 complete 2,048-token blocks; the 1,444-token
tail is excluded. The single seed does not support uncertainty intervals.
Hardware differs between historical and new executions (H200, with one H100
for Run034 A4/h at kappa=0.5); device provenance remains in the plotted data.

This is a descriptive update to the evidence behind the manuscript's
quality-sparsity and scale discussion. No manuscript text or finding is changed.

Reproduce locally from the repository root:

```powershell
.venv/Scripts/python.exe analyses/025-2026-09-17-70m-quality-sparsity/01_plot.py
```

The builder reconciles source verification records, condition identities,
training budget, validation coverage, count sums and the complete five-kappa
grids. It verifies that all 22 points lie within the displayed axes. The PDF
is rendered separately for visual inspection; preview rasters remain in `tmp/`.

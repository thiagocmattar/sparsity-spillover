# 012: Matched quality-sparsity panel with post-hoc control frontiers

## Question

Add baseline and 1-site post-hoc clipping frontiers for both model sizes to
Figure11, retaining the original figure and all of its trained recipes.

## Method and coverage

Keep Figure11's 44 trained checkpoints and eight pressure-family curves
unchanged. Add four retained ten-point sweeps: A0 and A1-H at 14M and 70M,
with calibration targets p = 0, 0.1, ..., 0.9. These are 40 evaluation-only
measurements of four existing checkpoints, not 40 additional trained models.
All 84 displayed records are retained, including the four independent p=0
measurements. No target is removed for high loss, overlap or domination.

The clipping protocol calibrates absolute-value order-statistic thresholds
at a,m,h,z from the first ten complete training blocks in source order. One
target p is shared, with separate thresholds per site and layer. During
evaluation, values with abs(x) <= threshold become zero after existing gates.
Weights remain fixed. The 1-site label describes the trained A1-H ReLU
checkpoint; its clipping sweep also acts at all four sites a,m,h,z.

Clipping uses FP32 parameters, FP16 CUDA autocast, eager uncached attention,
batch one and sequence length 2,048. Each point covers all 500 validation
documents / 338 complete blocks / 692,224 input tokens, excluding the
1,444-token tail. X is 100 times pooled integer block zero-product counts
divided by full-model product counts, including the dense LM-head denominator.
It is measured logical sparsity, not the calibration target p.

The 40 points come from Run030's verified consolidated `clipping-points.json`.
That release reused the original control measurements from Analysis006 and
Run018. The builder checks the release hash against Run030's verification,
checks the original source hashes, and reconciles losses, integer counters,
coverage and calibrated thresholds with the original raw records. Checkpoint
content identities and weight-file hashes match Figure11's exact controls.
No new training, calibration, checkpoint evaluation or cloud compute is used.

## Loss-pass alignment

Trained points preserve Figure11's ordinary final-checkpoint losses. Clipping
points preserve their own measured FP16 eager losses, including the actual
p=0 reference. These evaluation paths differ slightly. Across the four
controls, the maximum absolute p=0 versus ordinary-final difference is
0.0000125473067607 nats. No loss offset or sparsity shift is applied to force
the sweeps to meet the ordinary endpoint. The four exact differences and
paired sweep deltas are retained in `data/quality-sparsity-clipping.json`.

## Legend and caption

**Validation loss versus model-wide logical sparsity, including post-hoc
clipping of the baseline and 1-site controls at 14M and 70M.** All 44 trained
points and eight pressure-family curves from Figure11 remain unchanged.
Open markers indicate 14M and filled markers indicate 70M. Gray stars identify
baseline A0, green circles 1-site A1-H, blue diamonds four-site recipes and
orange triangles seven-site recipes. Short/long dashes identify OL1(h)/OL1(all).
Dotted gray and green paths add the four complete post-hoc sweeps on A0 and
A1-H, connecting increasing p. Smaller markers identify their ten measured
targets, including p=0. Both controls are clipped at a,m,h,z. Both axes are
linear, and the full measured loss range is visible. Lower loss is better.

## Result

The additional curves show the evaluated trade-off available by clipping
fixed control checkpoints. The highest displayed clipping loss is
9.179397 nats (rounded), so the y range expands relative to Figure11 to
retain every measured target. Training curves remain at their original
coordinates; their within-group variation is visually compressed by the
expanded range. Model-size labels sit above each size's complete cohort.

All ten points in each selected sweep are nondominated within that checkpoint's
retained sweep. This does not imply that they lie on a joint frontier across
all trained and clipped recipes. Connecting segments are visual guides;
intermediate clipping settings were not evaluated.

## Caveats

Clipping is an evaluation-only four-site intervention, not training with
threshold gates or pressure. Targets are repeated measurements of fixed
weights, not independent seeds; no uncertainty intervals are inferred.
Natural zero ties can affect achieved sparsity. Architecture-dependent
logical-product denominators, optimization-regime differences, hardware
differences and single-seed limitations from Figure11 remain. This plot
does not establish a causal sparsity effect, a scaling law or runtime gains.
No manuscript text or finding is updated.

## Sources and reproduction

Run `12_plot_quality_sparsity_clipping.py` to regenerate
[Figure12](../figures/12-14m-70m-quality-sparsity-clipping.pdf) and
[`data/quality-sparsity-clipping.json`](../data/quality-sparsity-clipping.json).
The JSON retains all trained and clipping coordinates, eight trained and four
clipping connection lists, checkpoint identities, integer counters, coverage,
source hashes, p=0 comparisons, labels, axis limits and PDF hash. Original
calibrated thresholds and per-operation counters remain in the hash-checked
Run030 release and raw sources. See [Figure11's observation](011-matched-quality-sparsity.md)
and [Run030's data release](../../../runs/030-2026-09-08-all-models-posthoc-clipping/results/README.md).

Verification checks the complete four-by-ten target grid, exact trained-point
preservation, control checkpoint identity, raw/reduced equality, count pooling,
paired deltas, full validation, source/PDF hashes and all-point axis coverage.
The single-page PDF was rendered and visually inspected; all fonts are
embedded. Figure11 and its underlying data remain unchanged.

# Manuscript results data

## Pressure-placement extension, 17 September 2026

`pressure-placement.json` adds all six multisite 14M recipes, including ten
h-only endpoints, to the original 54-endpoint release: 64 trained models in
total. It contains both full-validation loss passes, matched contrasts, pooled
integer site/layer and operation counts, and all 14,240 OL1 step records.
The approved reported losses use ordinary final validation for h-only and the
logical diagnostic pass otherwise; their maximum difference is 0.000106 nats.
Both uniform-pass alternatives are retained explicitly.

The extension also restores all three matched K050 implementations and skip
counters for five historical four-site h-only checkpoints, from qualified raw
timing pairs in the same RTX5090 session. The current figure uses 35 points
and recomputes its fits. `kernel/` retains the original 30-point reduction;
these files are not silently overwritten. The five seven-site h-only timings
and h-only scaling remain pending. Copy hashes appear in `SOURCES.json`.
Source: [Analysis 021 pressure scope](../../../analyses/021-2026-09-10-training-results-figures/pressure-scope/README.md).

The following catalog describes the original release.

This release accompanies the manuscript results, including the 11 September
kernel revision. The 18 copied measurement files are byte-identical to retained evidence;
SOURCES.json records the original repository path, SHA-256 and byte size.
The additional protocol.json is a locally composed record with its own
hashed source map. No evaluation
was launched for manuscript assembly.

- `figure-data.json`: 54 trained endpoints, 29 paired contrasts, 15 scale
  comparisons, integer operation counts, protocol/source identities and the
  corrected 30-checkpoint runtime reduction. Its legacy clipping subset has
  190 points; use the complete clipping release below for all 540.
- `density-figure-data.json`: signed-density plot normalization, zero masses
  and displayed/native tail fractions for each panel.
- `histograms/*.json.gz`: seven checkpoint measurements, with per-site and
  pooled signed counts, exact zeros, grid tails, coverage and source hashes.
- `clipping/clipping-points.csv`: all 540 checkpoint/target coordinates.
- `clipping/clipping-points.json`: the same points with integer counts,
  calibrated thresholds, coverage, checkpoint hashes and trained endpoints.
- `clipping/raw-points.json.gz`: lossless underlying measurements, including
  per-layer statistics, retained without requiring the original source paths.
- `clipping/frontiers.json`: evaluated frontier memberships, preserving ties.
- `clipping/verification.json`: source release checks and hashes; paths in it
  refer to the originating run, not this copy.
- `protocol.json`: pinned data/tokenizer revisions and packing, cache hashes,
  training settings, implementation/environment identities and known kernel
  limits. Its source paths are repository-relative and refer to retained
  records, not to files packaged inside this measurement directory.
- `kernel/checkpoints.csv` and `kernel/checkpoints.json`: the same 30 K050
  checkpoints, with 140 fields covering scalar operation counts, structural
  counters, raw-latency geometric means, matched gains, and missing quantities.
- `kernel/associations.csv`: pooled and within-family descriptive Pearson/OLS
  fits, including constant-predictor cases marked undefined.
- `kernel/matched-pairs.csv`: all ten four-/seven-site comparisons at matched
  thresholds, with and without OL1.

The kernel files come from
[Analysis 021's investigation](../../../analyses/021-2026-09-10-training-results-figures/investigation/README.md).
Its [field guide](../../../analyses/021-2026-09-10-training-results-figures/investigation/METHODS.md)
defines counters and traces the actual implementation. The complete five-page
[diagnostic PDF](../figures/appendix/kernel-investigation.pdf) is included with
the manuscript figures; its first page appears in the appendix.

## Fields and interpretation

`R_model` is the fraction plotted as S_model (multiply by 100 for percent).
`counts` contains pooled integer product counts. `loss` is validation
cross-entropy on all 338 complete 2,048-token blocks from 500 MiniPile
validation documents; the 1,444-token tail is excluded. Each size uses one
training seed (1234), and each trained endpoint is at update 712.

`training_parameter` (or legacy `dose` on trained records) denotes lambda
for A1-H-L1/A1-H-OL1 and kappa for A4/A7 variants; A0/A1-H use null.
`clipping_target_p` (legacy `dose` on clipping records) is a calibration
quantile, not achieved model-wide sparsity. It takes 0,.1,...,.9 with fixed
checkpoint weights. `delta_loss_from_p0` uses that sweep's measured p=0.
The stored `U_arch` uses recipe/site-union reach and is NOT the common A7
normalization used in the manuscript's cross-scale figure and endpoint table.
For the latter compute R_model / the same-size A7 ceiling explicitly.

Histogram exact zeros are separate from nonzero bins. Divide bin counts by
all captured elements and actual bin width; do not normalize the nonzero
curve to unit area. Runtime speedups are separate BF16 measurements against
each checkpoint's native SDPA CUDA-graph baseline; canonical logical sparsity
is measured in FP16. Neither logical sparsity nor frontier membership is a
measured runtime gain. Threshold targets are not independent training seeds.

Kernel `*_model_contribution_pp` fields use the common model denominator;
`*_scalar_zero_fraction` fields use their operation or operation-group denominator.
MMA bypass counts instructions, including padded projection rows and scalar
substitution; it is not eliminated arithmetic. Attention MMA counts include
masked/padded probability work. Counters cover 338 validation blocks using BF16
operands, while timing covers 64 inputs with seven passes in three processes.
`projection_sparse_gain` divides all-skips-off by projection-on candidate
latencies; `attention_sparse_gain` divides projection-on by full-K050 latency.
Both are full-model ratios with other implementation choices retained.
Blank CSV fields/null JSON fields denote unavailable measurements, not zeros.

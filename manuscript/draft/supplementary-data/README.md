# Manuscript results data

## Main/appendix quality split, 18 September 2026

`quality-main-appendix.json` is copied unchanged from Analysis024. After
the 19 September figure polish, it retains the original 74 trained and
60 Base/ReLU clipping records: 58/40 in the 14M/70M main figure, 12/20 in
the complete 410M appendix panel, and four unplotted historical naive-L1
records under `excluded_trained_points`.
Coordinates, source identities, loss conventions and pooled counts are
unchanged. Every 410M point is visible at the expanded loss range.
Main-panel loss scales are now independent, with 8/7 clipping points above
their respective views. The original `all-model-quality-sparsity.json` remains
the measurement and protocol source. Current figures are quality Figure 3, speedup Figure 4
and 410M Figure 15; earlier numbering below records earlier revisions.
See the [placement and preservation audit](../reviews/2026-09-18-quality-scope-placement/README.md).

## Base-model speedup adoption, 18 September 2026

`base-model-speedup.json` is copied unchanged from Analysis024's
`data/14m-70m-sparsity-base-speedup.json`. It supports Figure 3 with all 44
trained checkpoints, 40 control-clipping settings, original latencies,
size-specific A0 references, computed ratios, pooled logical counts,
checkpoint identities, timing sessions and source hashes. The caption and
main text distinguish the optimized-base reference from native execution.
See the [adoption and numerical audit](../reviews/2026-09-18-base-speedup-adoption/README.md).
Later figure numbers below describe their earlier adoption states.

## Operation-sparsity/runtime integration, 18 September 2026

The existing `paired-pressure-figure-data.json` also contains the complete
`04-operation-sparsity-changes.pdf` record: twelve 14M checkpoints, six
operation numerators per checkpoint, full-model denominators and stack totals.
That record now supports manuscript Figure 5 and the QK/PV contribution in
Section 4.3. `kernel/operation-bypass-summary.json` continues to support the
unchanged instruction-bypass artwork, now Figure 6. Logical zero products and
matrix instructions have different units and denominators; neither is a
measured fraction of running time saved. No new measurements were collected.

## Paired pressure figure adoption, 18 September 2026

`paired-pressure-figure-data.json` is a byte-identical copy of Analysis024's
`data/paper-derived.json`. The entry for `03-pressure-scope-threshold.pdf`
contains all twenty all-site-minus-h-only contrasts across model size,
threshold scope and kappa, their forty checkpoint keys and timing sessions.
Those checkpoints and their endpoint measurements are already included in
`figure1-quality-latency.json`. The copied export also retains the other
Analysis024 figure records; this adoption does not replace those paper figures.
See the [adoption and numerical audit](../reviews/2026-09-18-paired-pressure/README.md).

## All-model quality overview, 18 September 2026

`all-model-quality-sparsity.json` is copied unchanged from Analysis024. It
contains all 74 trained paper conditions (40/22/12 at 14M/70M/410M), sixty
Base/ReLU post-hoc clipping settings, pooled counts, analytic ceilings,
checkpoint identities, protocol checks, plotted recipe styles and source
hashes. All trained losses use ordinary reloaded final-checkpoint validation;
historical logical-pass losses remain separately recorded. Thus the 410M
values may differ slightly from the older endpoint tables below. The 62
existing 14M/70M coordinates are preserved exactly. The export also records
the eight kappa=.5 optimized-base-relative speedup calculations cited in
Section 4.1. See the [adoption record](../reviews/2026-09-18-all-model-quality-overview/README.md).

## Two-size Figure 1 extension, 18 September 2026

`figure1-quality-latency.json` is copied unchanged from Analysis024's
`data/14m-70m-quality-sparsity-latency.json`. It records 44 trained checkpoints
(22 each at 14M/70M), 40 post-hoc control settings with matched measured
latencies, the panel membership and visibility limits, analytic ceilings,
checkpoint identities, timing sessions, and source hashes. Trained loss is
uniformly ordinary final-checkpoint validation; clipping retains its original
FP16 loss/count pass. Timing uses BF16 and is a separate measurement.

The ten added 70M h-only endpoints supplement the historical 64 below, giving
74 conditions across the paper. The historical skipping-ablation and clipping
sweeps remain unchanged. See the
[Figure 1 adoption record](../reviews/2026-09-18-two-size-main-figure/README.md).
The following entries describe their original release scopes.

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

`kernel/operation-bypass-summary.json` supplies manuscript Figure 7's four
selected checkpoints: T4/Pall and T7/Pall at kappa=0.5 for 14M/70M. It retains
pooled issued/bypassed instruction counts for all six operations, scalar
replacement counts, checkpoint identities and source hashes. The bars measure
instruction bypass, including padding and scalar replacement, rather than
latency savings. The originating [observation](../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/029-operation-bypass-summary.md)
and `21_plot_operation_bypass_summary.py` document the selection and figure.

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

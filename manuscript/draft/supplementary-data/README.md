# Manuscript results data

This local release retains the 8 September 2026 measurements and adds the
9 September existing-results audit.
The 14 copied measurement files are byte-identical to retained evidence;
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
For the latter compute R_model / the same-size A7 architectural reach explicitly;
this equals block-only sparsity S_block in the declared Pythia graph.
Architectural reach R_arch is the manuscript alias for the unchanged
R_model_max fields. It is not a quality-constrained upper bound on measured sparsity.

Histogram exact zeros are separate from nonzero bins. Divide bin counts by
all captured elements and actual bin width; do not normalize the nonzero
curve to unit area. Runtime speedups are separate BF16 measurements against
each checkpoint's native SDPA CUDA-graph baseline; canonical logical sparsity
is measured in FP16. Neither logical sparsity nor frontier membership is a
measured runtime gain. Threshold targets are not independent training seeds.

## Existing-results audit added on 9 September

- `revision-audit/training-audit.json`: all 54 canonical endpoints with exact
  A0-relative losses, 29 paired and 15 cross-size differences, operation sums,
  actual-p=0 sensitivity and retrospective .05/.10/.20 quality budgets over
  all 594 evaluated trained/clipped points. Empty sparse-trained regimes remain
  explicit. No interpolated endpoints or imputed clipped-checkpoint latencies.
- `revision-audit/runtime-audit.json`: 30-checkpoint raw-timing reconciliation,
  actual BF16 losses and host-millisecond summaries, native-normalized ratios,
  incremental sparse-path factors, all leave-family/within-family regressions,
  and retained BF16 h/z row-NNZ summaries by layer. Absolute latency is the
  geometric mean of process medians; the reported speedup instead averages
  raw paired ratios geometrically. The BF16 scalar counter is only a lower
  bound and does not replace canonical FP16 counts.
- `revision-audit/historical-audit.json`: the five verified h-only-pressure
  controls, their exact identities and matched four-site differences. They
  remain outside the main 30-checkpoint cohort and are not fixed-coefficient
  placement controls.

These are reductions of existing measurements. The accompanying repository's
Analysis 019 README distinguishes regenerating visuals from re-auditing raw
run artifacts. This directory alone does not include training checkpoints,
executable GPU kernels or every original timing sample. No anonymous public
code or checkpoint URL is asserted here.

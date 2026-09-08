# Manuscript results data

This local release accompanies the 8 September 2026 manuscript results.
Every data file is a byte-identical copy of retained evidence; SOURCES.json
records the original repository path, SHA-256 and byte size. No evaluation
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

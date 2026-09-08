# Complete clipping measurements

The release contains 540 evaluations: every one of the 54 current manuscript
checkpoints at p=0,.1,...,.9. It combines 350 new points with 190 retained points
matched by exact checkpoint content hash. No point is discarded for domination,
overlap, or high loss. This excludes the historical h-only A4-OL1 cohort.

| Scale | Checkpoints | Clipping evaluations | Joint frontier records | Distinct frontier coordinates |
| --- | --- | --- | --- | --- |
| 14M | 30 | 300 | 29 | 18 |
| 70M | 12 | 120 | 35 | 21 |
| 410M | 12 | 120 | 20 | 12 |

- [clipping-points.csv](clipping-points.csv): one row per checkpoint/target, with source recipe, original training parameter, paired loss, logical sparsity, normalization and within-checkpoint frontier flag.
- [clipping-points.json](clipping-points.json): the same 540 points with integer counts, coverage, calibrated thresholds, checkpoint hashes, normalization ceilings, original source paths, 54 trained endpoints and pooled frontier memberships.
- [raw-points.json.gz](raw-points.json.gz): lossless compressed raw measurements for all 540 points, including retained per-layer activation statistics and timing fields. Raw data are bundled so the original artifact paths need not be present to inspect the measurements.
- [frontiers.json](frontiers.json): exact evaluated membership, separately for clipping-only and training-plus-clipping pools at each model size. IDs join to clipping-points.json; exact ties are preserved.
- [verification.json](verification.json): cross-format equality, coverage/grid checks, p=0 differences, actual plot-coordinate checks, execution identities and output hashes.
- [transfer-receipt.json](transfer-receipt.json): final SHA-256 inventory verified locally before Pod deletion.

`R_model` is a fraction; multiply by 100 for the plotted S_model percentage.
`U_arch` is a fraction of the analytic reach ceiling for the union of trained
gate and clipping sites. `clipping_target_p` is the calibration target; the JSON
retains the legacy field name `dose` for compatibility. `training_parameter` is
lambda for local L1/OL1, kappa for A4/A7 recipes, and blank/null for A0/A1-H.
`delta_loss_from_p0` compares against that checkpoint's actual measured p=0.
Each point uses all 338 validation blocks from 500 documents. Targets are
repeated evaluations of fixed weights, not independent seeds. No runtime
speedups or continuously attainable intermediate models are inferred.

Full-range PDFs and self-contained captions are in [observations](../observations/INDEX.md).
Analysis 018 Figure 01 uses all 300 matching 14M clipping evaluations.

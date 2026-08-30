# Analysis 006 - TEAL post-hoc clipping of every full-pass A1-H/A4-Z checkpoint

## Status

Complete and verified. The approved local evaluator ran from
2026-08-30T18:42:27.7802292-03:00 to 2026-08-30T19:18:13.0615690-03:00 and
wrote all 130 new full-validation points. The combined artifact contains the
130 new points plus the 20 frozen Analysis 005 control points. The one-panel
PDF was rendered and visually inspected with its displayed loss axis capped at
6.

## Question

Does applying uniform TEAL-style post-hoc clipping to every trained A1-H and
A4-Z full-pass checkpoint extend the measured quality--`R_model` frontier
beyond post-hoc clipping of the GeLU and ReLU controls alone?

## Approved source set

Analysis 004 supplies the canonical 15 endpoints and their matched identities.
Analysis 005 supplies the already-complete 10-target sweeps for the Run 004
GeLU and ReLU controls. This analysis evaluates the remaining 13 checkpoints:

- Run 004 A1-H naive L1 at `lambda in {0.05, 0.1, 0.5, 1}`;
- Run 009 A1-H OL1 at `lambda in {0.05, 0.1, 0.5, 1}`;
- Run 011 A4-Z one-sided thresholding at
  `kappa in {0, 0.01, 0.05, 0.1, 0.5}`.

All are retained step-712 checkpoints from random Pythia-14M initialization,
with the same seed-1234 initial-parameter hash, data-order/schedule hash,
1,493,172,224 training input tokens, and complete-validation workload. Released
Pythia weights are not used.

## Operational method

The method exactly reuses the frozen Analysis 005 implementation and records
its code hash. For each checkpoint, the first 10 complete source-order training
blocks calibrate a separate empirical absolute-value threshold for each of the
24 `site.layer` tensors at `a`, `m`, `h`, and `z`. The same target sparsity in
`{0.0, 0.1, ..., 0.9}` is assigned to all tensors, while thresholds remain
tensor-specific. During evaluation only, `abs(x) <= threshold` becomes exact
zero. Natural zero ties can exceed the requested target.

The clipping ports are the actual inputs to fused QKV, MLP W1, MLP W2, and the
attention output projection. On A4-Z checkpoints, this post-hoc symmetric
magnitude clipping acts after the trained one-sided gates at the same four
sites. Targets below a checkpoint's natural exact-zero mass may therefore be
no-ops. This is uniform TEAL-style clipping, not TEAL's optional block-wise
greedy allocation; it evaluates full-sequence pretraining loss and logical
zero-product opportunity without a sparse kernel.

Every point covers all 500 validation documents, all 338 complete 2,048-token
blocks, and 692,224 input tokens; the 1,444-token tail is excluded. Evaluation
uses batch size 1, FP16 CUDA autocast, FP32 parameters, eager uncached attention,
and count-first aggregation. The `p=0` loss must reproduce the source endpoint
within `5e-4`.

## Measurements and retention

Each new point stores per-layer exact-zero/RMS/L2 activation statistics for
`a,m,h,z`, per-site pooled exact-zero counts, thresholds, all six actual-operand
logical-product counter families, `R_block`, `R_model`, validation loss,
throughput, duration, and full coverage. Calibration and progress are durable
and protocol-locked so an interrupted local run can resume without repeating
completed points.

Weights are unchanged by post-hoc clipping, so source weight diagnostics are
linked rather than recomputed. Gradient interaction is training-time-only and
cannot be reconstructed; existing Run 004/009 records remain authoritative.
All source final checkpoints are already retained and content-hash verified.

## Outputs

- `teal_all_variants.json` - 150-point combined artifact: 130 new points plus
  20 reused control points.
- `artifacts/calibration/*.json` - 13 per-checkpoint calibration artifacts.
- `artifacts/progress.jsonl` - durable new-point completion records.
- `figure_data.json` and `tables.md` - count-reconciled figure/table reduction.
- `figures/01-all-full-pass-teal-frontiers.pdf` - one-panel trained A1/A4 and
  post-hoc TEAL comparison with validation loss capped at 6.
- `observations/O001-all-full-pass-teal-frontiers.md` - result, caption,
  provenance, and limits after execution.

The figure retains all 15 trained endpoints, shows the visible portion of
every complete post-hoc trajectory, and highlights the pooled TEAL-augmented
nondominated envelope. Above-cap points remain in JSON/table and are omitted,
not clamped, in the PDF.

## Result

All 15 conditions have the complete target grid `p in {0.0, ..., 0.9}` and
all 150 rows cover the same 338 complete validation blocks. The new run has no
non-finite activation rows or logical-count discrepancies; all 13 `p=0` losses
match their source endpoints exactly within the `5e-4` guard. Median evaluation
time was 16.141 seconds per point (42,887 input tokens/second), and the new
evaluation completed in 2,145.35 seconds.

In the pooled descriptive frontier, A1-H naive L1 at `lambda=1` supplies the
low-loss segment through `R_model=6.4902%` at loss 5.2456. A4-Z one-sided
thresholding at `kappa=0.1` extends it to `R_model=9.3424%` at loss 5.4466;
`kappa=0.5` reaches `R_model=10.5631%` at loss 5.7058 while remaining under the
display cap. The GeLU and ReLU control trajectories are explicitly shown as
evaluation-only post-hoc clipping. Fifty-four higher-loss points are outside
the displayed range but remain in `teal_all_variants.json`, `figure_data.json`,
and `tables.md`. See `observations/O001-all-full-pass-teal-frontiers.md`.

## Post-run verification

- Structural reconciliation: 150 rows, 15 conditions, 10 targets per
  condition, 13 new zero-threshold guards passed, 130 durable progress rows,
  13 calibration artifacts, zero validation-coverage failures, zero non-finite
  activation rows, and zero logical-count reconciliation failures.
- Focused Analysis 006 tests: 5 passed.
- Full bootstrap suite: 159 passed.
- PDF QA: one 800.167 by 447.134 point page, rendered to PNG and visually
  inspected; extracted text confirms the control/evaluation-only wording,
  loss-6 cap, and 54-point disclosure.
- Deterministic PDF regeneration reproduced SHA-256
  `1217ea4ca69d30c66912c8f00b4a20772ec488c58c9fb0ff0d5e27f4218fcdf9`.

## Pre-launch evidence

`prelaunch/smoke.json` records a non-evidence smoke on checkpoint
`relu-l1n-0p05`: all 13 source checkpoint identities were reconciled, 24
site-layer thresholds were calibrated over 10 blocks, and `p=0.2` completed 32
validation blocks with count-reconciled logical products. On the local RTX
5070 Ti Laptop GPU it reserved 1.776 GB, evaluated 65,536 input tokens in
1.706 seconds (38,422 tokens/second), and remained finite. The source checkpoint
content hash and both cache hashes match the approved inputs.

The repository `.venv` currently contains CPU-only Torch 2.11. The passing
smoke used the system Python 3.12 environment with Torch 2.11.0+cu128 and
Transformers 5.12.1, matching Analysis 005. The proposed detached command pins
that interpreter and adds the repository `src/` directory to `PYTHONPATH`.
`prelaunch/launch-plan.json` contains the local resource fit, ETC arithmetic,
monitoring warnings, storage inventory, and exact execution definition.

## Interpretation limits

This is one seed and one Pythia-14M scale. A1-H and A4-Z differ in topology,
trained operator, and pressure method. TEAL clipping compounds those learned
interventions rather than isolating them. `R_model` is exact-zero logical-product
opportunity, not removed FLOPs or measured speedup. No observation is promoted
to a finding or manuscript claim automatically.

## Approval record

- 2026-08-30: the user confirmed the 13-checkpoint, 130-new-point design, reuse
  of the two complete control sweeps, complete target grid, full validation,
  diagnostic scope, and one-panel loss-capped figure.
- 2026-08-30T18:40:31.6483242-03:00: the user approved launch of the exact
  local 130-point evaluation described here, with a 50-minute maximum-duration
  cap.

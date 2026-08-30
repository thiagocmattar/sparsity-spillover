# Analysis 006 - TEAL post-hoc clipping of every full-pass A1-H/A4-Z checkpoint

## Status

Design approved on 2026-08-30. Implementation, focused tests, and the bounded
local smoke are complete and pass. The 130-point scientific evaluation has not
been launched and still requires the separate launch confirmation.

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

## Planned outputs

- `teal_all_variants.json` - 150-point combined artifact: 130 new points plus
  20 reused control points.
- `artifacts/calibration/*.json` - 13 per-checkpoint calibration artifacts.
- `artifacts/progress.jsonl` - durable new-point completion records.
- `figure_data.json` and `tables.md` - count-reconciled figure/table reduction.
- `figures/01-all-full-pass-teal-frontiers.pdf` - one-panel trained A1/A4 and
  post-hoc TEAL comparison with validation loss capped at 6.
- `observations/O001-all-full-pass-teal-frontiers.md` - result, caption,
  provenance, and limits after execution.

The figure will retain all 15 trained endpoints, show the visible portion of
every complete post-hoc trajectory, and highlight the pooled TEAL-augmented
nondominated envelope. Above-cap points remain in JSON/table and are omitted,
not clamped, in the PDF.

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
- Scientific execution remains pending a separate launch confirmation after
  implementation tests, bounded smoke, resource fit, and ETC are reported.

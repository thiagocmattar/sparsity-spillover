# Analysis 005 - TEAL-style post-hoc clipping of Run 004 controls

## Status

Completed as a verified local post-hoc analysis of the two Run 004 controls.
All 20 target/control points cover complete validation, all activation and
logical fractions reconcile to integer counts, and the requested PDF has been
rendered and visually verified. No centralized finding was promoted. On
30 August 2026, the user approved reporting the analysis as a strong
descriptive result in
`manuscript/reports/01-2026-08-30-status-update/status-update.tex`.

## Question

How does uniform TEAL-style post-hoc magnitude clipping change the measured
`R_model` versus complete-validation loss frontier for the full-pass Pythia-14M
GeLU and ReLU controls used in Analysis 003 Figure 1?

## Operational interpretation

TEAL applies magnitude sparsity to every matrix input. In this Pythia mapping,
one threshold is calibrated for each of four input tensors in each of six
blocks: `a` feeds the fused QKV projection, `m` feeds MLP W1, `h` feeds MLP W2,
and `z` feeds the attention output projection. Values satisfying
`abs(x) <= t` become exact zero at evaluation only. This is joint clipping; it
does not attribute the response to an individual site.

Thresholds are calibrated separately for every site-layer tensor on the first
10 complete 2,048-token blocks of the MiniPile training split. The sweep assigns
the same target sparsity in `{0.0, 0.1, ..., 0.9}` to every matrix input, while
the corresponding absolute threshold is tensor-specific. Natural ties at zero
can make realized sparsity exceed the target, especially for the ReLU control.
The target-zero row is included and every loss delta is paired to that row.

This is the uniform TEAL variant, not the paper's optional block-wise greedy
allocation. Pythia has one fused QKV matrix rather than separate Q/K/V matrices.
The evaluation measures full-sequence pretraining loss rather than batch-one
autoregressive decoding, and it measures logical zero-product opportunities
rather than sparse-kernel runtime.

## Sources and coverage

- Run 004 verified GeLU control: attempt
  `001-20260829-221007-bb5288c8`, final step 712 checkpoint.
- Run 004 verified ReLU control: attempt
  `002-20260829-231305-195b22c6`, final step 712 checkpoint.
- Calibration: a disjoint training split, 10 complete blocks and 20,480 input
  tokens in deterministic source order.
- Evaluation: all 500 validation documents, all 338 complete 2,048-token blocks,
  692,224 input tokens, with the 1,444-token tail excluded.
- FP16 CUDA autocast with FP32 parameters, matching Run 004 evaluation.
- One Pythia-14M initialization/data-order seed; no replicate uncertainty.

`01_evaluate.py` verifies both checkpoint-content hashes and both token-cache
hashes, calibrates empirical per-matrix thresholds, evaluates every point with
count-first activation and logical-product counters, and durably appends each
complete point to `artifacts/progress.jsonl`. It can resume without repeating
completed points only when the complete protocol hash matches.

## Results

Both controls trace a monotonic quality--logical-opportunity tradeoff over the
tested uniform targets: every higher target increases both `R_model` and
validation loss. For GeLU, target `p=0.1` raises `R_model` from effectively zero
to `1.278706%` at a paired loss increase of `+0.005859`; `p=0.2` reaches
`2.554786%` at `+0.041676` loss. For ReLU, the zero-target checkpoint already
has `2.714133%` `R_model` from its natural `h` zeros. Targets `p=0.1` and `0.2`
reach `3.568403%` and `4.418365%`, with paired loss increases of `+0.004439`
and `+0.038307`.

At the common `p=0.4` target, GeLU reaches `5.125544%` `R_model` at loss
`5.605384`, while ReLU reaches `6.129325%` at loss `5.602965`. This is a
descriptive checkpoint comparison, not an activation-isolated effect. The
ReLU control's `h` exact-zero fraction remains near its natural `63.45%`
through target `p=0.4`; its mild-target opportunity gains therefore come from
clipping `a`, `m`, and `z` while preserving those existing `h` zeros.

Across both curves, the global nondominated envelope contains GeLU targets
`0.0--0.2`, ReLU targets `0.0--0.6`, both controls at `0.7` and `0.8`, and the
GeLU target `0.9`. The GeLU `0.3--0.6` points and ReLU `0.9` point are globally
dominated on these two measured coordinates. This arithmetic ordering has no
replicate uncertainty and is not promoted to a finding.

## Outputs

- `teal_frontier.json` - count-reconciled machine-readable 20-point sweep.
- `artifacts/calibration/*.json` - per-control empirical threshold calibration.
- `artifacts/progress.jsonl` - durable point-completion record.
- `tables.md` - endpoint, site-zero, loss-delta, and global-frontier table.
- `figures/01-r-model-vs-final-validation-loss.pdf` - requested full-range and
  low-loss-detail figure.
- `figure_data_02.json` - reduced, source-hashed data for the combined
  full-pass/post-hoc figure, including the points omitted above its display cap.
- `figures/02-full-pass-frontier-with-posthoc-controls.pdf` - single-panel
  comparison of the Analysis 004 A1/A4 trained endpoints with the GeLU/ReLU
  control curves under evaluation-only post-hoc clipping; displayed loss is
  capped at 6.
- `observations/O001-r-model-vs-final-validation-loss.md` - caption,
  provenance, result, and limits.
- `observations/O002-full-pass-frontier-with-posthoc-controls.md` - caption,
  cross-analysis provenance, display-cap rule, result, and limits for Figure 2.

## Limits

This analysis evaluates one seed and one 14M-scale model family. Calibration
uses a deterministic 10-block sample and does not estimate calibration-sample
uncertainty. The full-sequence workload differs materially from TEAL's focal
decode setting. `R_model` is exact-zero logical-product opportunity, not
removed FLOPs or measured acceleration. The result is reported in Status
Report Number 1 with these limits; no centralized finding is promoted.

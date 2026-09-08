# Run 030: complete post-hoc clipping for the manuscript cohort

Status: production evaluation running. The user explicitly authorized the runs
and RunPod access on 8 September 2026, then requested the complete retained
data and an update of Analysis 018 Figure 01.

## Question and scope

How does evaluation-only magnitude clipping extend each trained checkpoint's
quality versus logical-sparsity trade-off? Cover all 54 current manuscript
checkpoints: 30 Pythia-14M, 12 Pythia-70M and 12 Pythia-410M. Reuse the 190
verified evaluations for 19 checkpoints and evaluate the remaining 35 checkpoints
at all ten targets, producing 540 evaluations in total. The historical A4-OL1
with pressure only at h remains outside the manuscript cohort.

Weights, trained gates, pressure history, seed 1234, step 712, initialization,
and training schedule remain those of each source checkpoint. No optimizer,
backward pass or additional training is used. This addresses the draft's
train-time intervention versus post-hoc clipping comparison, without changing
manuscript prose or promoting a finding.

## Fixed method

Reuse the existing uniform TEAL-style protocol. Calibrate each site/layer's
absolute-value order-statistic threshold on the first ten complete source-order
training blocks. Targets are p=0,.1,...,.9, with one common target but separate
thresholds at a,m,h,z. Apply abs(x)<=threshold after the existing trained gates;
keep the A7 query/key/value gates unchanged. Preserve each sweep's measured p=0
point and require agreement with its canonical eager endpoint within 5e-4 loss.
Natural zero ties can make several targets identical.

Evaluate every point with FP32 parameters, FP16 CUDA autocast, eager uncached
attention, batch one and sequence length 2048. Use all 500 MiniPile validation
documents: 338 complete blocks, 692224 input tokens, and the excluded 1444-token
tail. Verify the existing cache hashes before use. Pool integer counts before
dividing. The clipping grid and every evaluated point remain available even
when dominated or beyond a figure's visible loss range.

Retain calibrated thresholds, per-layer/site activation exact-zero and
near-zero counts, RMS/L2 statistics, all six logical-product count families,
paired validation loss, source/checkpoint/cache/code identities, timings,
progress and environment. Source checkpoints and weight diagnostics remain
retained; gradient interaction belongs to the original training records.
Save per-checkpoint and pooled nondominated memberships separately from the
complete trajectories. R_model is logical opportunity, not measured speedup.
For normalization, use the union of trained gate sites and clipping sites.

Support for an extended frontier requires evaluated clipped points to improve
the pooled quality/sparsity trade-off. Unchanged or dominated clipping outcomes
are equally valid results. This one-seed grid does not establish optimal
intermediate settings, causal attribution to a site, or a scaling law.

## Execution and retention

The initial RTX 5090 (32 GB) calibration passed all six complete evaluations:
about 12 seconds per 14M/70M point and 64 seconds per 410M point. The local
bootstrap and focused tests passed 247/247; the remote focused tests passed 5/5.
To shorten elapsed time, the 250 missing 14M/70M points and two complete 410M
sweeps use that RTX 5090. The remaining eight 410M sweeps use one RTX PRO 6000
Blackwell worker (96 GB). A second RTX PRO 6000 request returned unavailable;
no third Pod was created. The smaller same-region GPU types
have no remaining advertised availability. Every worker uses the existing
100 GB network volume 9luykg5yc3 in EUR-IS-1; no new network volume.
Live Secure quotes are USD0.99/hour and USD2.09/hour respectively. Use a
two-hour compute backstop per production worker, within the USD15 total cap
including setup and storage. Expected remaining wall time is 75-90 minutes,
subject to each worker's measured throughput. Use the previously exercised
image digest runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35.
All new files live under /workspace/run030, with detached execution, separate
worker output directories and durable per-point progress. The static assignment
is in worker-checkpoints.json (270/80 points). Monitor every 120 seconds, reporting completed evaluations,
latest loss, throughput and refreshed ETC. Inspect nonfinite values, identity
or p=0 failures, stale progress, CUDA errors, and projected budget overruns.

Copy completed output incrementally; verify every final artifact's SHA-256
locally before deleting the Pod. Preserve the existing network volume and
source checkpoints. A deadline guard stops compute while preserving outputs
if normal completion/teardown is interrupted. No data, weights or credentials
are committed. Publication PDFs and compact structured measurement artifacts
are retained beside their generating run or analysis.

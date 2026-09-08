# Research Index

> Compact state only. Detail belongs in numbered runs, analyses, observations,
> and findings.

## Current status

Run 030 completed all 540 post-hoc clipping evaluations across the 54-checkpoint
14M/70M/410M manuscript cohort (350 new, 190 reused). CSV, raw/normalized JSON,
per-scale full-range PDFs and hash verification are retained. Analysis 018
Figure 01 now selects the OL1 recipes and A0/A1-H controls (16 trajectories). Both Pods are deleted;
zero Pods/endpoints remain, with the existing network volume preserved. See
`runs/030-2026-09-08-all-models-posthoc-clipping/README.md`.

Analysis 018 packages the 8 September draft's results materials: eight revised
paper-width PDFs, 54 trained conditions, OL1-focused 14M overview, matched effects,
ceiling-versus-size and scale/clipping views, activation mass, and a consistent
30-checkpoint runtime subset. Other clipping panels retain their earlier subsets; finer
activation bins await diagnostic design confirmation. Historical h-only A4 pressure
is excluded. No manuscript edit or finding promotion. See `analyses/018-2026-09-08-results-materials/README.md`.

Run029 completed the matched retrospective:1173 process outcomes,42 eligible14M
proposals, one RTX5090 and a common native SDPA CUDA-graph denominator. Qualified
c30 progress reaches1.783x; K050 passes35/35 checkpoints, geomean1.250x,
canonical R_model regression R_squared0.781. Sparse-path benefit is conditional;
attention skipping adds overhead. The Sakana-derived P0 qualifies on5/35, not an
unchanged upstream benchmark. Evidence and publication PDFs are verified
locally; the Pod is deleted and the existing volume retained. See Run029's
observations. The 8 September implementation audit supports the user-approved
scoped case study now in `manuscript/draft/kernel-autoresearch.tex`; it records
a non-cohort threshold bug and the agent-identity/causal limits. No finding is
promoted.

Run 028's frozen K050 final evaluation and diagnostics are complete. Qualified
full-model graph speedups across35 checkpoints range0.979--1.739x; sparse-path
and fusion ablations are retained. Figure07 and Analysis017 require their
stated baseline boundaries; mixed historical phases are not one matched curve.
See Run028's final observations; Run029 now supplies a common-denominator history.

Run 027 completed the 35-checkpoint Pythia-14M characterization of K019+K018.
Six variants pass all full-validation gates; their speedups are 1.490-1.808x.
The best qualified A7+OL1 endpoint has a 1.2365x skip-toggle benefit, about
29% of its net latency saving; R_model alone is not proportional to speedup.
Both PDFs and all 105 processes are verified locally. The 29 logit-gate
failures remain visible; zero Pods/endpoints remain, estimated new expense
is $0.93/$5 (posted billing incomplete), and the existing volume is unchanged.
No finding or manuscript change is promoted; see Run 027's observations.

Run 026 completed the user-authorized Pythia-14M continuation: frozen K019+K018
achieves 1.8149x over qualified stock eager SDPA, with bitwise-identical full
validation logits in three processes. Fusion and sparsity both contribute;
this is not a >1.6x claim over custom fused-dense ablations. Its README links
all variant latencies, canonical R_model and progress. Evidence is hash-verified
locally; zero Pods/endpoints remain, estimated new study spend about $0.44/$25
(posted billing pending), and the existing volume is unchanged.

Run 025 completed its Sakana-derived search through K016, the frozen
36-checkpoint Blackwell matrix, three-process Blackwell replication, fixed-policy
H100 endpoint transfer, component probes, and direct fixed-`R_model`
confirmations. Analysis 015 owns the detailed reductions; Analysis 016 owns the
focused `R_model`--speedup and autoresearch-progress figures. Fresh-process
qualified `R2` is 0.347/0.0027/0.636 on RTX and 0.321/0.051/0.700 on H100 for
14M/70M/410M, but all six H100 410M sentinels remain slower than native. The
common high-sparsity endpoint score finishes at 1.0461x/1.0213x/1.0147x on RTX.
QK/PV stayed dense and the strongest compiled-dense comparator remains absent.
Evidence is local and hash-verified; zero Pods/endpoints remain and the existing
100 GB volume is retained.

Run 024 completed and independently verified the six Pythia-410M sparse-kernel
sentinels on the exact physical H100 NVL used by Run 023. The official
SparseLM0.5B control reached 1.2739x, but none of 432 linear primitives, 48 A7
attention compositions, or 12 full-model condition × batch timings broke
even. Best full-model speedup was 0.3275x at batch one and 0.4989x at batch 32.
All six batch-one endpoints were worse than their matched 70M results; only
high-threshold A7 improved slightly at batch 32 and remained a 2.32x slowdown.
Analysis 014 owns the same-GPU reduction. This refutes the narrow hypothesis
that increasing Pythia scale to 410M is enough to amortize the exact-ELL path;
it does not turn `R_model` into a runtime estimand. Artifacts were hash-verified
locally, the Pod was deleted, zero Pods/endpoints remain, and the pre-existing
volume is unchanged.

Run 023 completed and verified all six Pythia-70M sentinels. Its official
positive control reached 1.2876x, while no linear, attention, or full-model
Pythia timing broke even. The derived path is technically compatible through
A7 attention, but the result is a systems limitation rather than a runtime
validation of `R_model`.

Run 022 reproduced the official SparseLM0.5B H100 control at 1.3001x, then
stopped correctly when the raw Pythia-14M `M=256,K=512,N=128` ELL operation
had relative-L2 error 0.69055. Its bounded post-mortem found correct behavior
at N=256/512/2048 and isolated divergent full-warp shuffle participation below
N=256. Evidence was retrieved and the Pod was terminated.

Run 019 completed and terminally verified the approved one-seed Pythia-410M
promotion of A0, A1-H, A4-OL1, and A7-OL1. All 12 conditions completed 712
optimizer boundaries, full 338-block validation, retained diagnostics and
checkpoints, and the two complete ten-target post-hoc TEAL frontiers. Analysis
011 owns the paired-loss reduction, complete Markdown tables, and separate
410M-only and 14M/70M/410M publication PDFs. At `kappa=0.5`, A7-OL1 reaches
80.6155% `R_model` at loss 5.120692 versus A4-OL1's 71.5914% at 5.190966.
The A7-over-A4 high-dose ordering persists across all three sizes, while the
zero-threshold ordering reverses at 410M. This remains one-seed descriptive
evidence, not a scaling law or measured-speedup result; no finding or
manuscript claim was promoted. All Run 019 Pods and endpoints are deleted.

Run 018 completed and verified the selected one-seed Pythia-70M promotion of
A0, A1-H, A4-OL1, and A7-OL1: 12/12 conditions completed 712 optimizer
boundaries, full 338-block validation, retained diagnostics/checkpoints, and
the two complete ten-target post-hoc TEAL frontiers. Analysis 010 owns the
original two-scale reduction, complete Markdown tables, and publication PDF.
The 14M crossover persists descriptively at 70M: A4-OL1 has higher `R_model`
at `kappa=0`, while A7-OL1 is higher at `kappa=0.5`. Status Report Number 2
reports that result and its Pod-ID-reconciled billing audit.
Run 016 was superseded after GPU-dependent initialization hashes, and Run 017
stopped before science when its proposed portable initialization failed the
remote identity check.

Analysis 009 directly compares historical Run 012 A4-Z + OL1@h with corrected
Run 015 four-site A4-OL1 at all five matched thresholds. Status Report Number 1
now uses Run 015 only in its main A4-OL1 result and corrected Analysis 008
frontier, retaining Run 012 as an A4-OL1[`h`] appendix. Four-site pressure
raises validation loss by 0.242--0.320 at every `kappa`; its `R_model` is lower
at `kappa=0`, effectively tied at `0.01`, and 1.42--2.49 percentage points
higher at `0.05`--`0.5`. Its PDF also includes matched Run 011 A4 without OL1
as the common baseline. The analysis owns the all-eight-site count-pooled
exact-zero table and machine-readable reduction. The result is
descriptive; discarded F001 is not restored and no paper finding is promoted.

Run 015 completed and verified the corrective paper-scale Pythia-14M A4-OL1
cohort. Every one of 3,560 boundaries realized all 24
`{a,m,h,z}.layer_{0..5}` pressure tensors, and all five conditions completed
full validation and local artifact reconciliation. Against matched Run 011,
corrected OL1 improves both validation loss and `R_model` at `kappa=0` and
`0.01`; at larger thresholds it adds 2.25--2.50 percentage points of logical
opportunity with increasing validation-loss cost. All Pods are deleted and the
pre-existing volume is unchanged. Analysis 009 now compares the corrected
result directly with Run 012's historical `h`-only realization; Analysis 007
remains historical A4-Z + OL1@h frontier evidence.

Run 014 completed the five-condition paper-scale A7-OL1 cohort with valid
evidence. At matched `kappa=0.1`, OL1 adds 1.3717 percentage points of
`R_model` for `+0.000816` validation loss; at `kappa=0.5`, it adds 12.0959
points for `+0.126466` loss. The zero-threshold row regresses, so the effect is
threshold-dependent rather than uniform. All attempts, diagnostics, and final
checkpoints are local and verified; all Pods are deleted. Analysis 008 now
includes the endpoints in its count-reconciled full-pass frontier and
interleaved all-site table. Status Report Number 1 now includes that table and
the corrected single-panel Analysis 008 frontier; A7-OL1 is marked completed at
the report cutoff. The comparison remains descriptive and has not been promoted
to a finding.
Finding F002 tentatively consolidates the one-seed Pythia-14M A4/A7
comparison. A7 is a near-null topology expansion at `kappa=0`; at
`kappa=0.01`, it improves loss by 0.007678 and `R_model` by 0.2039 percentage
points. Larger thresholds add 0.9210--5.1713 percentage points of logical
opportunity with increasing validation-loss cost. Analysis 008 owns the
count-reconciled A7/A7-OL1 zero-mass table, expanded full-pass frontier figure,
observation, and machine-readable evidence. Finding F002 remains scoped to the
matched Runs 011/013 A4/A7 endpoints; Run 014's A7-OL1 extension has no promoted
finding. All Run 013 artifacts are local and verified, all Pods are deleted,
and the pre-existing volume remains intentionally retained.
Run 012 completed five paper-scale A4-Z + OL1@h conditions. Its 3,560 optimizer
steps, validation passes, checkpoints, and diagnostics remain locally
reconciled for that realized intervention, but its declared four-site A4-OL1
identity and former verifier label are invalid.
Run 011 completed its five-condition, paper-scale A4-Z threshold cohort with
valid evidence. `R_model` rose monotonically from 7.212% at `kappa=0` to
10.216% at `kappa=0.5`; the lowest final validation loss was 5.419642 at
`kappa=0.1`, while `kappa=0.5` degraded to 5.659680. All requested diagnostics
and final recovery checkpoints are local and verified. All Pods are deleted,
the pre-existing 100 GB volume remains intentionally retained, and the matched
A4 endpoints remain the Run 015 comparator and support tentative Finding F002.
Run 010 completed its mixed A7-Z-POST plus all-site OL1 cohort in 62m00s with
valid evidence. `R_model` rose from 7.774% at `kappa=0` to the 29.952% analytic
ceiling at `kappa=0.5`; the lowest validation loss was 5.645979 at `kappa=0.01`.
Run 009 completed its separate full-pass, `h`-only OL1 grid with valid evidence
and reused Run 004 controls. Matched validation loss was lower than naive L1 at
lambda `0.05`, `0.1`, and `0.5`, and higher at `1.0`. Analysis 003 now records
the complete per-site, validation-loss, and quality--logical-opportunity
comparison, together with boundary-level gradient interference, OL1 geometry,
and per-boundary conflict and projection PDFs; no finding was promoted.
Run 008 completed its five-condition mixed A7-Z-POST threshold cohort in 46m49s
with valid evidence. `R_model` rose from 7.882% at `kappa=0` to 27.572% at
`kappa=0.5`; the lowest final validation loss was 5.648046 at `kappa=0.1`.
Run 004 is closed with valid evidence: six
full-pass conditions, locally verified checkpoint/diagnostic inventories, two
post-hoc PDF figures, and a near-zero/`R_model` table. Its observations have not
been promoted to a finding or manuscript claim. A live RunPod closeout found
zero Pods and one intentionally retained 100 GB volume at `$7/month`.

Next run number: `031`. Next analysis number: `019`. Next finding number: `F003`.

## Where we stopped

- 2026-09-06: Run 025's full achieved systems package completed. Analysis 015
  owns the detailed checkpoint, hardware, component, and fixed-`R_model`
  reductions; Analysis 016 distills the `R_model`--speedup relation and
  correctness-constrained candidate progress into two verified PDFs. The
  evidence is one adaptive systems case study, not a universal speed law or an
  optimizer comparison; no finding or manuscript claim was promoted. No GPU
  resource remains.
- 2026-09-04: Run 024 completed the matched Pythia-410M sparse-kernel sentinel
  on Run 023's physical H100 NVL. All six conditions and full validation pass;
  no primitive, attention composition, or full model breaks even. Analysis 014
  records the negative 70M→410M systems scaling result. The result archive and
  internal hashes were independently verified; the Pod and guard are deleted.
- 2026-09-04: Run 023 completed and verified the six Pythia-70M sparse-kernel
  sentinels. The official control accelerates, but every derived Pythia path is
  slower than native dense. Artifacts are local and no GPU resource remains.
- 2026-09-04: Run 022 stopped at its declared Pythia correctness gate after the
  official SparseLM positive control passed. The N=128 failure and width audit
  are retained; all GPU compute was terminated.
- 2026-09-03: Analysis 011 completed the selected-ladder synthesis through
  Pythia-410M. It records 30 trained endpoints and all 60 A0/A1-H TEAL points,
  pairs trained loss and `R_model` within the same eager logical pass, and owns
  separate 410M-only and three-scale frontier PDFs, an A0 cross-scale training
  loss, gradient-norm, and learning-rate PDF, and complete sitewise tables. The
  result is descriptive;
  no finding or manuscript claim was promoted.
- 2026-09-03: Run 019 completed and terminally verified all 12 Pythia-410M
  conditions and both ten-target TEAL frontiers. All result archives and
  checkpoints are local and hash-reconciled. The final control-plane audit
  found zero GPU Pods and zero endpoints; the pre-existing unattached network
  volume remains intentionally retained.
- 2026-09-01: Run 019 was implemented and locally verified for the 410M
  selected-ladder promotion. The canonical seed-1234 initialization and RNG
  artifacts are pinned, and the exact A0/A4-OL1/A7-OL1 calibration will compare
  at least two GPU SKUs before any scientific launch. No Run 019 cloud resource
  or scientific attempt has been created.
- 2026-09-01: Status Report Number 2 was compiled and adversarially reviewed.
  It retains Report 1's architecture/intervention diagram and 14M evidence,
  updates the scope and current RunPod costs, adds cross-scale A4-OL1/A7-OL1
  exact-zero tables, shows all ten 70M TEAL targets, and embeds Analysis 010's
  single loss--`R_model` frontier. No finding or scaling-law claim was promoted.
- 2026-09-01: Analysis 010 completed the 14M/70M selected-ladder synthesis with
  all 20 trained endpoints and all 40 post-hoc TEAL points. Its PDF retains
  off-scale trajectories above validation loss 6 and its Markdown tables carry
  all requested sitewise exact-zero masses.
- 2026-09-01: Run 018 completed and verified all 12 canonical Pythia-70M
  conditions, including A0/A1-H TEAL, diagnostics, final checkpoints, archive
  hashes, and complete validation. All Pods were deleted; the one pre-existing
  100 GB volume remains intentionally retained.
- 2026-09-01: Runs 016 and 017 stopped before scientific execution after two
  distinct initialization-portability failures. Their preflight and
  compatibility records are retained; Run 018 replaced initialization with a
  canonical, hash-verified model artifact.

- 2026-08-31: Status Report Number 1 now uses corrected Run 015 as its only
  main-text A4-OL1 evidence, updates Analysis 008's full-pass frontier to the
  same series, and retains historical Run 012 as an A4-OL1[`h`] appendix with
  its endpoint table, direct comparison figure, and compute accounting. F001
  remains discarded; the corrected A4/A4-OL1 comparison remains descriptive.
- 2026-08-31: Analysis 009 completed the matched Run 012/015 pressure-target
  comparison. Its PDF plots matched A4 without OL1, historical `h`-only OL1,
  and corrected four-site OL1; its observation contains
  the count-pooled exact-zero table for all eight recorded diagnostics. The
  result remains descriptive and does not restore F001.
- 2026-08-31: Run 015 completed and verified the corrected five-condition
  four-site A4-OL1 full pass. All 3,560 boundaries proved the 24-tensor pressure
  identity; complete validation, diagnostics, checkpoints, archive hashes, and
  local verification reconcile. All RunPod Pods are deleted. The candidate
  observation is not a promoted finding; Analysis 009 is complete and F001
  remains discarded.
- 2026-08-31: Status Report Number 1 was updated through Run 014: A7-OL1 is
  complete at the cutoff, the report contains the interleaved all-site
  A7/A7-OL1 table, Run 014 compute/cost, and Analysis 008's corrected
  single-panel frontier with plain OL1 labels. No A7/A7-OL1 finding was
  promoted.
- 2026-08-31: Analysis 008 added Run 014 A7-OL1 to its full-pass trained and
  post-hoc frontier, widened the plotted logical-opportunity range, and updated
  its machine-readable reduction, interleaved A7/A7-OL1 table, and observation.
  The A7/A7-OL1 result remains descriptive; Finding F002 is unchanged.
- 2026-08-31: Run 014 completed and verified the five paper-scale A7-OL1
  conditions against matched Run 013 A7 endpoints. It adds nondominated
  `kappa=0.1` and `0.5` points, while `kappa=0` regresses. All eight Run 014
  infrastructure/science Pods were deleted; the pre-existing volume is
  unchanged. No finding or manuscript claim was promoted.
- 2026-08-31: Finding F002 tentatively consolidated Analysis 008's matched
  full-pass A4/A7 result and linked its dose-dependent
  quality--logical-opportunity statement into the manuscript and
  experiment-control crosswalks.
- 2026-08-31: Analysis 008 added Run 013 A7 to Analysis 007's full-pass
  trained/post-hoc frontier and produced one count-pooled table containing
  `kappa`, validation loss, `R_model`, and all eight recorded exact-zero site
  masses. It now supports tentative Finding F002.
- 2026-08-31: Run 013 completed and verified valid: five condition-parallel
  A100 attempts, 3,560 optimizer steps, 20 complete validation passes, common
  initialization/schedule/code identities, retained recovery checkpoints, and
  all approved diagnostics reconcile. All Run 013 Pods were deleted, and the
  run-local candidate observation records the matched A7/A4 result.
- 2026-08-30: Finding F001 tentatively consolidated what was then interpreted
  as matched A4/A4-OL1 evidence. The 2026-08-31 implementation audit supersedes
  this entry and discards the finding.
- 2026-08-30: Analysis 007 added what was then labeled Run 012 A4-OL1 to the
  frontier. It is now classified as historical A4-Z + OL1@h data.
- 2026-08-30: Run 012 completed and verified valid: five condition-parallel A100
  attempts, 3,560 optimizer steps, 20 complete validation passes, common
  initialization/schedule/code identities, five checkpoints, and all approved
  diagnostics reconcile. All Run 012 Pods were deleted.
- 2026-08-30: Analysis 006 completed uniform TEAL-style clipping of all 13
  trained full-pass A1-H/A4-Z checkpoints and combined them with the two frozen
  control sweeps. All 150 points are verified; the one-panel PDF and table are
  complete, with no finding promoted.
- 2026-08-30: Analysis 005 evaluated uniform TEAL-style post-hoc clipping of
  Run 004's verified full-pass GeLU and ReLU controls over 20 complete-validation
  points. The count-reconciled PDFs and observations are complete, and the user
  approved the result for Status Report Number 1; no centralized finding was
  promoted.
- 2026-08-30: Analysis 004 combined the verified full-pass Run 004/009 A1-H
  naive-L1/OL1 endpoints with Run 011's A4-Z threshold cohort in a
  count-reconciled quality--logical-opportunity table and paper-ready PDF. The
  topology/intervention difference is explicit; no finding was promoted.
- 2026-08-30: Run 011 completed and verified valid: five condition-parallel
  A100 attempts, 3,560 optimizer steps, 20 complete validation passes, common
  initialization/schedule/code identities, exact checkpoint inventories, and
  all approved diagnostics reconcile. All Run 011 Pods were deleted. Its later
  consolidation with Run 012 in F001 was discarded after the capture audit.
- 2026-08-30: Analysis 003 completed the verified Run 004 naive-L1 versus Run
  009 OL1 endpoint and gradient-interaction comparison, with count-reconciled
  exact/near-zero tables, endpoint, per-boundary conflict, and OL1-projection
  PDFs, and explicit limits on the OL1 guarantee. No finding was promoted.
- 2026-08-30: Run 010 completed and verified valid: five conditions, 2,905
  updates, 15 complete validation passes, five checkpoints, mixed-gate and
  seven-site OL1 identities, gradient diagnostics, and logical counters all
  reconcile.
- 2026-08-30: Run 009 completed and verified valid: four distributed OL1
  conditions, 2,848 boundaries, complete validation/diagnostics/checkpoints,
  verified retrievals, zero remaining Pods, and one pre-existing retained
  volume. No finding was promoted.
- 2026-08-30: Run 004 closed. Posted Pod charges are `$21.5213452158`; all Pods
  are deleted; volume `9luykg5yc3` remains intentionally billable. The optional
  checkpoint validation-loss trajectory is deferred and requires no retraining.
- 2026-08-30: Run 008 completed and verified valid. All five conditions, 2,905
  updates, 15 complete validation passes, five checkpoints, mixed-gate
  identities, diagnostics, and logical counters reconcile.
- 2026-08-29: Run 007 remains paused after repeated OOM in its final condition;
  Runs 002, 003, 005, and 006 are complete at the evidence statuses below.

## Available baseline

- Core Pythia site/gate, L1/OL1, diagnostic, and artifact primitives.
- Known MiniPile/Pythia-14M revisions and cache identities in `DATA.md`.
- Local token caches and retained run-owned checkpoints/diagnostics are present;
  use each run's verification and inventory rather than assuming availability.

## Paper direction

- Test and delimit sparsity spillover: targeted near-zero concentration versus
  responses in untargeted attention sites.
- Connect sitewise activation behavior to actual zero-operand counts across the
  complete Pythia block and dense LM-head denominator.
- Validate topology/architecture/sequence-dependent `R_model_max` ceilings.
- Evaluate whether architecture-wide interventions improve the
  quality--logical-compute frontier without claiming unmeasured speedup.

These are manuscript-led goals, not accepted findings or approved runs.

## Findings

| # | Statement | Status | Source |
| --- | --- | --- | --- |
| F001 | Former four-site A4-OL1 claim; discarded because Run 012 actually applied OL1 only at `h`. | discarded | `research/findings/F001-a4-ol1-improves-moderate-threshold-frontier.md` |
| F002 | At Pythia-14M, A7 is near-null at `kappa=0`, improves matched A4 loss and `R_model` at `kappa=0.01`, and adds logical opportunity with increasing quality cost at larger thresholds. | tentative | `research/findings/F002-a7-extends-a4-logical-opportunity.md` |

## Runs

| # | Question | State | Folder |
| --- | --- | --- | --- |
| 001 | Can the local pipeline complete a matched four-LR Pythia-14M shakedown? | completed (provisional) | `runs/001-2026-08-28-local-lr-pipeline-calibration/` |
| 002 | Does naive L1 pressure at `h` induce opposing near-zero movement in untargeted attention sites? | completed (valid) | `runs/002-2026-08-29-l1n-spillover-local/` |
| 003 | Does disabling global gradient clipping change the matched GeLU/ReLU `lambda=5` L1N outcome? | completed (valid with provenance limitation) | `runs/003-2026-08-29-l1n-lambda5-no-gradient-clipping/` |
| 004 | Does a full MiniPile pass show ReLU L1N spillover under the mapped Pythia recipe? | closed (valid; observations complete, no finding promoted) | `runs/004-2026-08-29-pythia14m-full-pass-l1n/` |
| 005 | How does ReLU `h`-only OL1 strength change sparsity, quality, conflict, and logical opportunity locally? | completed (valid with provenance limitation) | `runs/005-2026-08-29-pythia14m-relu-ol1-local/` |
| 006 | How does joint one-sided thresholding at `a,m,h,z` change quality and logical opportunity locally? | completed (valid) | `runs/006-2026-08-29-pythia14m-a4z-threshold-local/` |
| 007 | Does all-site OL1 improve the joint A4-Z threshold quality--logical-opportunity frontier? | four conditions complete; repeated local OOM, paused | `runs/007-2026-08-29-pythia14m-a4z-threshold-ol1-local/` |
| 008 | Do symmetric post-RoPE Q/K/V gates extend Run 006's joint threshold quality--logical-opportunity frontier? | completed (valid) | `runs/008-2026-08-29-pythia14m-a7-z-post-mixed-threshold-local/` |
| 009 | Does `h`-only OL1 improve the full-pass ReLU naive-L1 frontier? | completed (valid; Analysis 003 complete) | `runs/009-2026-08-30-pythia14m-full-pass-ol1/` |
| 010 | Does all-site OL1 improve Run 008's mixed A7-Z-POST threshold frontier? | completed (valid) | `runs/010-2026-08-30-pythia14m-a7-z-post-mixed-threshold-ol1-local/` |
| 011 | How does paper-scale A4-Z threshold strength change quality and logical opportunity? | completed (valid; Run 015 comparator and F002 source) | `runs/011-2026-08-30-pythia14m-full-pass-a4z/` |
| 012 | What happened under A4-Z gates plus the realized `h`-only OL1 pressure? | completed (valid only as A4-Z + OL1@h; declared four-site identity invalid) | `runs/012-2026-08-30-pythia14m-full-pass-a4-ol1/` |
| 013 | Do symmetric post-RoPE Q/K/V gates improve the paper-scale A4 quality--logical-opportunity frontier? | completed (valid; Analysis 008 and F002 complete) | `runs/013-2026-08-30-pythia14m-full-pass-a7/` |
| 014 | Does seven-site OL1 improve the paper-scale A7 quality--logical-opportunity frontier? | completed (valid; run-local observation, no finding promoted) | `runs/014-2026-08-31-pythia14m-full-pass-a7-ol1/` |
| 015 | Does correctly realized four-site OL1 improve the matched paper-scale A4 frontier? | completed (valid; Analysis 009 complete) | `runs/015-2026-08-31-pythia14m-corrected-a4-ol1/` |
| 016 | Does the selected ladder persist at 70M under a GPU-realized initialization identity? | superseded before science; GPU-specific initialization hash | `runs/016-2026-08-31-pythia70m-selected-ladder/` |
| 017 | Can CPU-before-CUDA initialization make the selected 70M promotion portable? | stopped before science; remote identity mismatch | `runs/017-2026-09-01-pythia70m-selected-ladder-portable-init/` |
| 018 | Does the selected A0/A1-H/A4-OL1/A7-OL1 ladder persist at Pythia-70M with a canonical initialization artifact? | completed (valid; Analysis 010 and Status Report 2 complete) | `runs/018-2026-09-01-pythia70m-selected-ladder-canonical-init/` |
| 019 | Does the selected ladder and its A0/A1-H TEAL frontier persist at Pythia-410M? | completed (valid; Analysis 011 complete) | `runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/` |
| 020 | Can a two-arm higher-LR screen repair the weak one-pass Pythia-410M A0 endpoint? | terminally failed before science; replaced by Run 021 | `runs/020-2026-09-03-pythia410m-a0-learning-rate-screen/` |
| 021 | Does the corrected higher-LR A0 screen outperform the Run-019 3e-4 baseline? | completed (valid; baseline retained) | `runs/021-2026-09-03-pythia410m-a0-learning-rate-screen-resolved-lr/` |
| 022 | Can the official sparse kernel execute an exact Pythia-14M A0 W2 shape? | stopped at correctness gate; bounded N=128 defect isolated | `runs/022-2026-09-04-pythia14m-sparse-kernel-a0-baseline/` |
| 023 | Do the Sakana-derived kernels accelerate selected Pythia-70M endpoints? | completed (valid negative systems result) | `runs/023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels/` |
| 024 | Does scaling the same sparse-kernel sentinels to Pythia-410M reach break-even? | completed (valid negative systems result; Analysis 014 complete) | `runs/024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels/` |
| 025 | Can a bounded Sakana-derived agentic search convert Pythia logical opportunity into full-model speedup? | Blackwell matrix complete; partial against paper minimum; Analysis 015 complete | `runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/` |

## Analyses

| # | Question | State | Folder |
| --- | --- | --- | --- |
| 001 | How does seven-site OL1 change Run 008 at each matched kappa? | completed; descriptive, no finding promoted | `analyses/001-2026-08-30-run008-vs-run010-all-site-ol1/` |
| 002 | How does four-site OL1 change Run 006 at each completed matched kappa? | completed for four available pairs; descriptive, no finding promoted | `analyses/002-2026-08-30-run006-vs-run007-partial-a4z-ol1/` |
| 003 | How does `h`-only OL1 change Run 004's matched endpoints and gradient interactions? | completed; descriptive, no finding promoted | `analyses/003-2026-08-30-run004-vs-run009-full-pass-l1-ol1/` |
| 004 | Where do the verified full-pass A1-H pressure and A4-Z threshold endpoints lie on the quality--logical-opportunity plane? | completed; descriptive, no finding promoted | `analyses/004-2026-08-30-full-pass-quality-logical-frontier/` |
| 005 | How does uniform TEAL-style post-hoc clipping change the Run 004 GeLU/ReLU control frontiers? | completed; descriptive; reported in Status Report Number 1; no finding promoted | `analyses/005-2026-08-30-run004-controls-teal-posthoc/` |
| 006 | How does uniform TEAL-style post-hoc clipping change every full-pass A1-H and A4-Z checkpoint frontier? | completed; descriptive, no finding promoted | `analyses/006-2026-08-30-full-pass-all-variants-teal-posthoc/` |
| 007 | Where do the historical Run 012 A4-Z + OL1@h endpoints lie relative to the trained and post-hoc frontiers? | completed as historical reduction; does not support four-site A4-OL1 or F001 | `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/` |
| 008 | Where do the full-pass A7 and A7-OL1 endpoints lie relative to the trained and post-hoc frontiers? | completed; supports tentative F002 for A4/A7, A7/A7-OL1 descriptive | `analyses/008-2026-08-31-full-pass-frontier-with-a7/` |
| 009 | How does corrected four-site A4-OL1 compare with Run 012's historical `h`-only realization? | completed; descriptive, no finding promoted | `analyses/009-2026-08-31-run012-vs-run015-a4-ol1-pressure-sites/` |
| 010 | Does the selected A0/A1-H/A4-OL1/A7-OL1 loss--`R_model` structure persist from Pythia-14M to 70M? | completed; descriptive two-size synthesis, no finding promoted | `analyses/010-2026-09-01-pythia14m-vs-70m-selected-ladder/` |
| 011 | How do the selected trained and post-hoc loss--`R_model` frontiers and A0 optimization trajectories compare through 410M? | completed; descriptive three-size synthesis and A0 loss/gradient/learning-rate diagnostic, no finding promoted | `analyses/011-2026-09-03-pythia14m-70m-410m-selected-ladder/` |
| 012 | What is the evidence-preserving paper synthesis of the selected ladder through 410M? | completed; descriptive paper-facing reduction, no finding promoted | `analyses/012-2026-09-04-paper-synthesis/` |
| 013 | What matched intervention evidence should support the manuscript rewrite? | completed; manuscript evidence refactor and verified draft assets | `analyses/013-2026-09-04-matched-intervention-manuscript/` |
| 014 | Does moving the sparse-kernel sentinels from 70M to 410M improve realized speedup? | completed; same-GPU negative systems calibration, no finding promoted | `analyses/014-2026-09-04-pythia70m-vs-410m-sparse-kernel-sentinels/` |
| 015 | After specialization, how do `R_model`, full-model speed, and fixed-`R_model` implementation changes relate through 410M? | completed Blackwell reduction; partial systems evidence, no finding promoted | `analyses/015-2026-09-06-pythia-agentic-kernel-search/` |

## Key documents

`DEFINITIONS.md` · `DATA.md` · `METHODS.md` · `METRICS.md` · `MANUSCRIPT.md` ·
`WORKFLOW.md` · `COMPUTE.md` · `RUNPOD.md` · `PLOTTING.md`

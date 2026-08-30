# Research Index

> Compact state only. Detail belongs in numbered runs, analyses, observations,
> and findings.

## Current status

Run 011's implementation passes all 150 repository tests. Its approved A100 SXM
endpoint preflight passed at both A4-Z threshold extremes with 57.242 GiB peak
reserved memory, about 430k input tokens/s, and no skipped boundary. The packet
is locally hash-verified and the Pod is stopped; scientific launch awaits a
separate user decision.
Run 010 completed its mixed A7-Z-POST plus all-site OL1 cohort in 62m00s with
valid evidence. `R_model` rose from 7.774% at `kappa=0` to the 29.952% analytic
ceiling at `kappa=0.5`; the lowest validation loss was 5.645979 at `kappa=0.01`.
Run 009 completed its separate full-pass, `h`-only OL1 grid with valid evidence
and reused Run 004 controls. Matched validation loss was lower than naive L1 at
lambda `0.05`, `0.1`, and `0.5`, and higher at `1.0`. Analysis 003 now records
the complete per-site, validation-loss, and quality--logical-opportunity
comparison with a PDF figure, together with the boundary-level gradient
interference and OL1 geometry; no finding was promoted.
Run 008 completed its five-condition mixed A7-Z-POST threshold cohort in 46m49s
with valid evidence. `R_model` rose from 7.882% at `kappa=0` to 27.572% at
`kappa=0.5`; the lowest final validation loss was 5.648046 at `kappa=0.1`.
Run 004 is closed with valid evidence: six
full-pass conditions, locally verified checkpoint/diagnostic inventories, two
post-hoc PDF figures, and a near-zero/`R_model` table. Its observations have not
been promoted to a finding or manuscript claim. A live RunPod closeout found
zero Pods and one intentionally retained 100 GB volume at `$7/month`.

Next run number: `012`. Next analysis number: `004`. Next finding number: `F001`.

## Where we stopped

- 2026-08-30: Run 011's approved A100 endpoint preflight passed, was retrieved
  and hash-verified locally, and the Pod was stopped after 1,770 seconds. The
  five-worker ETC is now 1.676h median/1.790h p90; scientific launch remains
  separately prohibited.
- 2026-08-30: Analysis 003 completed the verified Run 004 naive-L1 versus Run
  009 OL1 endpoint and gradient-interaction comparison, with count-reconciled
  exact/near-zero tables, an `R_model` versus final-validation-loss PDF, and
  explicit limits on the OL1 guarantee. No finding was promoted.
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
| 011 | How does paper-scale A4-Z threshold strength change quality and logical opportunity? | preflight passed; awaiting scientific-launch approval | `runs/011-2026-08-30-pythia14m-full-pass-a4z/` |

## Analyses

| # | Question | State | Folder |
| --- | --- | --- | --- |
| 001 | How does seven-site OL1 change Run 008 at each matched kappa? | completed; descriptive, no finding promoted | `analyses/001-2026-08-30-run008-vs-run010-all-site-ol1/` |
| 002 | How does four-site OL1 change Run 006 at each completed matched kappa? | completed for four available pairs; descriptive, no finding promoted | `analyses/002-2026-08-30-run006-vs-run007-partial-a4z-ol1/` |
| 003 | How does `h`-only OL1 change Run 004's matched endpoints and gradient interactions? | completed; descriptive, no finding promoted | `analyses/003-2026-08-30-run004-vs-run009-full-pass-l1-ol1/` |

## Key documents

`DEFINITIONS.md` · `DATA.md` · `METHODS.md` · `METRICS.md` · `MANUSCRIPT.md` ·
`WORKFLOW.md` · `COMPUTE.md` · `RUNPOD.md` · `PLOTTING.md`

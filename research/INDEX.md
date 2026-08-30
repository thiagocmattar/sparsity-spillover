# Research Index

> Compact state only. Detail belongs in numbered runs, analyses, observations,
> and findings.

## Current status

Run 008 is the active local scientific cohort; its authoritative progress lives
in its detached attempt artifacts. Run 004 is closed with valid evidence: six
full-pass conditions, locally verified checkpoint/diagnostic inventories, two
post-hoc PDF figures, and a near-zero/`R_model` table. Its observations have not
been promoted to a finding or manuscript claim. A live RunPod closeout found
zero Pods and one intentionally retained 100 GB volume at `$7/month`.

Next run number: `009`. Next analysis number: `001`. Next finding number: `F001`.

## Where we stopped

- 2026-08-30: Run 004 closed. Posted Pod charges are `$21.5213452158`; all Pods
  are deleted; volume `9luykg5yc3` remains intentionally billable. The optional
  checkpoint validation-loss trajectory is deferred and requires no retraining.
- 2026-08-30: Run 008 is active locally under its approved locked design.
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
| 008 | Do symmetric post-RoPE Q/K/V gates extend Run 006's joint threshold quality--logical-opportunity frontier? | local scientific cohort active; first condition healthy | `runs/008-2026-08-29-pythia14m-a7-z-post-mixed-threshold-local/` |

## Analyses

| # | Question | State | Folder |
| --- | --- | --- | --- |

## Key documents

`DEFINITIONS.md` · `DATA.md` · `METHODS.md` · `METRICS.md` · `MANUSCRIPT.md` ·
`WORKFLOW.md` · `COMPUTE.md` · `RUNPOD.md` · `PLOTTING.md`

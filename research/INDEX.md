# Research Index

Compact current state; full history belongs to numbered runs and analyses.
[Historical navigation](ARCHIVE-INDEX-2026-10-04.md) preserves the earlier index.

## Current status

[Analysis043](../analyses/043-2026-10-09-global-tile-head-frontier/README.md)
consolidates 138 retained execution points, common latency decomposition and
Ph versus Ph+head global OL1 geometry. No head point extends the observed global
envelope; T2 tile kappa=.1 does. Six verified PDFs and tables distinguish trained
body/head gates, precision and timing coverage. Local descriptive analysis;
no new GPU run, manuscript change or finding promotion.

[Run068](../runs/068-2026-10-08-pythia14m-aligned-vocabulary-tiles/results/REPORT.md)
completed fresh 14M Ph+head models: head16x16 kappa=.1/.5, h/z8x16 kappa=0. Loss is
5.560725/5.882600; empty head tiles 39.878%/70.763%. Sparse full-forward changes
+1.03%/-3.10% versus native dense. All six processes match native logits/loss
exactly; artifacts are hash-verified locally and both owned Pods are deleted.

[Run066 training](../runs/066-2026-10-08-pythia14m-vocabulary-tile-gates/results/REPORT.md)
and [Run067 timing](../runs/067-2026-10-08-pythia14m-sparse-vocabulary-kernel/results/DIAGNOSTICS.md)
retain the four head8 Ph/Ph+head models. Sparse full forward is 7.19-8.86% slower
than dense despite bypass savings; h/z arithmetic remains native dense.

[Analysis042](../analyses/042-2026-10-08-model-latency-decomposition/README.md)
retains full profiles from Runs064/065: tiled h/z at 14M, scalar at 31M/70M,
with common categories/colors. [Analysis041](../analyses/041-2026-10-04-corner-tile-versus-elementwise/README.md)
retains matched T2/T7 tile/scalar quality and latency comparisons.

## Working contracts and approved findings

- [Definitions](DEFINITIONS.md), [data](DATA.md), [methods](METHODS.md), [metrics](METRICS.md).
- [Workflow](WORKFLOW.md), [compute](COMPUTE.md), [RunPod](RUNPOD.md), [plotting](PLOTTING.md).
- [Manuscript crosswalk](MANUSCRIPT.md), [paper source](../manuscript/README.md), [backlog](BACKLOG.md).
- [F003: 70M h benefit and z limitation](findings/F003-70m-sparse-h-gain-does-not-establish-broad-hz-exploitation.md).
- [F004: closed P2 versus Ph comparison](findings/F004-14m-p2-ph-quality-and-latency.md).

The human selects direction and approves designs/launches. Verified artifacts
provide evidence. Existing foreign resources remain outside experiment ownership.

# Research Index

Compact current state; full history belongs to numbered runs and analyses.
[Historical navigation](ARCHIVE-INDEX-2026-10-04.md) preserves the previous index.

## Current status

[Run062](../runs/062-2026-10-04-pythia14m-t7-corner-tile-ph/README.md) completed14M
T7/Ph tiles at kappa0/.1. Loss rises.501244/.597707; matched sparse latency falls
11.133%/7.187%. All graphs pass full RTX5090 qualification; all artifacts are
local and owned Pods deleted. [Analysis041](../analyses/041-2026-10-04-corner-tile-versus-elementwise/README.md)
places both points outside the previous orange frontier, with three new PDFs.

[Run060](../runs/060-2026-10-04-pythia14m-temporal-corner-tile/README.md) completed
14M T2/Ph corner gates at kappa 0/.1: two verified 712-update conditions, full
validation/diagnostics, all checkpoints local, owned Pod deleted. Both models
learn with surviving h/z activity, but loss rises .426737/.444400 and perplexity
53.225%/55.955% versus matched scalar controls. Empty 8x16 tiles increase while
h/z element sparsity decreases. [Analysis041](../analyses/041-2026-10-04-corner-tile-versus-elementwise/README.md)
contains the full table and three verified PDF figures. One seed; redistribution
is untested, with no manuscript edit or finding promotion. The latency follow-up
is completed:
[Run061](../runs/061-2026-10-04-pythia14m-corner-tile-latency/README.md) compares
the four retained Ph checkpoints on one RTX5090. Sparse tile/scalar speedups
are1.123x/1.112x (latency-10.990%/-10.067%); every graph passes full own-eager
qualification. Original tile quality penalties remain. Artifacts/PDF are local;
the owned Pod is terminated (estimated GPU USD.303). Tile pressure
remains deferred; [backlog](BACKLOG.md) retains the contract.

The user closed P2 versus Ph: P2 raises loss, with only small threshold-dependent
K050 latency changes. All artifacts are local and owned Pods deleted.
[F004](findings/F004-14m-p2-ph-quality-and-latency.md) is the approved finding.

[Run057](../runs/057-2026-10-03-pythia14m-t4-gate-sign/README.md) completed the
14M T4 one-/double-sided a/m comparison: bilateral improves loss under Ph and
worsens it under Pall. This one-seed result is pressure-dependent.

[F003](findings/F003-70m-sparse-h-gain-does-not-establish-broad-hz-exploitation.md)
records the 70M kernel limitation: measured sparse benefit covers h in five
layers; z remains dense. Run052/053 searches did not promote a broader kernel.

## Working contracts

- [Definitions](DEFINITIONS.md), [data](DATA.md), [methods](METHODS.md), [metrics](METRICS.md).
- [Workflow](WORKFLOW.md), [compute](COMPUTE.md), [RunPod](RUNPOD.md), [plotting](PLOTTING.md).
- [Manuscript crosswalk](MANUSCRIPT.md), [paper source](../manuscript/README.md).
- [Backlog](BACKLOG.md), [prior runs/analyses/findings](ARCHIVE-INDEX-2026-10-04.md).

The human selects direction and approves designs/launches. Plans and manuscript
text are guides; verified artifacts provide evidence. Existing foreign resources
are outside experiment ownership; current inventories are in each run closeout.

# Research Index

Compact current state; full history belongs to numbered runs and analyses.
[Historical navigation](ARCHIVE-INDEX-2026-10-04.md) preserves the previous index.

## Current status

[Analysis042](../analyses/042-2026-10-08-model-latency-decomposition/README.md)
collects per-model latency tables and PDFs with explicit tiled (14M) and scalar
(31M/70M) threshold labels, using completed Runs064/065. Common categories/colors
and [exact definitions](../analyses/042-2026-10-08-model-latency-decomposition/LABELS.md)
separate cross-model budgets from measured implementation detail.

[Run065](../runs/065-2026-10-07-larger-model-forward-decomposition/README.md)
profiles retained scalar-gated 31M/70M T2/Ph models on one RTX 5090. Sparse
full-forward reductions are 4.63%/15.61% (31M) and 28.22%/31.03% (70M), at
kappa 0/.1. 31M's joint consumer remains a bottleneck; 70M's vocabulary head
leads. All 12 processes qualify; all 48 maps and artifacts are verified, owned Pod
deleted. [Full decomposition and PDF](../runs/065-2026-10-07-larger-model-forward-decomposition/observations/01-larger-forward-decomposition.md).
Different gates/consumers limit comparison with Run064's 14M corner models.

[Run064](../runs/064-2026-10-07-pythia14m-forward-latency-decomposition/README.md)
decomposes the complete forward on one RTX5090. The repeated specialization
gain is 10.337%/9.002%, localized to joint gate/down-projection/residual kernels.
Vocabulary projection and attention now take 38-40% and 26-27% of specialized
GPU-active time. Full qualification is exact; artifacts are local and the Pod
is terminated. [Detailed map, timings and PDF](../runs/064-2026-10-07-pythia14m-forward-latency-decomposition/observations/01-forward-decomposition.md)
separate wall time, CUPTI activities, event overhead and internal SM cycles.

[Run063](../runs/063-2026-10-07-pythia14m-tile-specialized-kernel/README.md)
completed the same-gate RTX5090 consumer comparison for retained T2/Ph models.
Specialized latency falls 10.526%/8.896% at kappa 0/.1; all five graphs match eager
logits/loss exactly over 338 validation blocks in each of six processes. With
skipping disabled, specialization is 3.409%/4.775% slower. Weights/gates stay
unchanged; the trained tile models' quality penalty remains. Artifacts are local
and the owned Pod is deleted. [Observation and PDF](../runs/063-2026-10-07-pythia14m-tile-specialized-kernel/observations/01-generic-versus-specialized.md).

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

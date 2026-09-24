# Base, T2/Ph and T7/Ph across model scales

Completed 24 September 2026. The comparison includes 14M, 31M (the
30,494,720-parameter Pythia-31M architecture) and 70M: Base under kernel and
PyTorch CUDA-graph execution, and T2/Ph/T7/Ph kernel execution at kappa
0, 0.01, 0.05, 0.1 and 0.5. These are 36 execution points from 33 checkpoints.
Historical endpoints come unchanged from Analysis037; five new 31M T7/Ph
endpoints passed Run055 training verification and all 15 final timing
qualifications. No preflight endpoint enters the figure.

Quality uses complete canonical validation; full-model timing uses RTX5090,
BF16,B1/T2048,full logits and CUDA graphs. Source hashes and all process values
are retained. Kernels are specialized by architecture; sessions differ, and
training uses one seed/fixed token budget. The empirical frontier is descriptive,
not a fitted scaling law. The user-approved manuscript integration is recorded
in [O003](observations/003-manuscript-integration.md).

The new T7/Ph points at kappa 0.05 and 0.1 are nondominated in the measured
cross-scale comparison: (loss 4.893721, 0.895587 ms) and
(loss 4.942717, 0.800644 ms), respectively. Compared at the same kappa, 31M
T7/Ph is 3.15-11.86% faster than T2/Ph and has 0.173-0.301 higher loss.
The remaining new points are dominated by previously measured endpoints.
Small differences, especially the 1.41% latency gap between T7/Ph 0.05 and
T2/Ph 0.1, require caution because the sessions differ.

- [Publication figure](figures/01-base-t2-t7-scale-frontier.pdf)
- [Observation, method and caveats](observations/001-scale-frontier.md)
- [Exact plotted coordinates and Pareto set](data/scale-figure.json)
- [New endpoints, process values and source hashes](data/31m-t7-results.json)

Reproduce from the repository root with `python 01_collect.py` and
`python 02_plot.py`, using their paths under this analysis folder. The collector
checks all final checkpoint hashes, full validation, 448 samples per backend
per process, runtime diagnostics, and a single GPU/source identity across
all 15 processes. The plotter checks historical source/PDF hashes, complete
five-kappa families, 36 points and 33 checkpoints. The PDF was rendered and
visually inspected; source coordinates and historical PDFs remain unchanged.

The paper-ready revision follows the typography, colors, line weights and
compact layout of the original Analysis034 `02-14m-70m-latency-quality.pdf`.
Its approved title is "Quality–latency trade-offs across model scales".
Dashed blue T2/Ph and solid orange T7/Ph curves use circles,
triangles and squares for 14M, 31M and 70M. Three vertical gray guides mark
the respective Base losses, now labeled by model size. Kappa labels, the workload subtitle and the legend
footer are omitted. The user corrected the earlier 30M label to 31M; all
36 measured coordinates and the Pareto set are unchanged.

The [two-panel companion](figures/02-absolute-relative-scale-frontier.pdf),
retained as an analysis artifact but excluded from the paper by the user,
retains the absolute comparison in panel (a). Panel (b) uses only T2/Ph,
with two aligned mini-panels sharing model size: purple loss-change boxes
above (`L - L_Base`) and teal latency-change boxes below
(`t - t_PyTorch_Base`, milliseconds). Each box summarizes the five kappa
settings, all overlaid as dots. Each mini-panel has its own linear Y axis
and zero reference. Boxes span the middle 50%; whiskers use 1.5 IQR.
The individual dots include every outlying value. Generate
it with `python analyses/038-2026-09-24-base-t2-t7-scale-frontier/03_absolute_relative.py`
from the repository root. See [O002](observations/002-absolute-relative.md)
for its caption and Base-reference provenance, and
`data/absolute-relative-figure.json` for exact values. The single-panel PDF
and its source coordinates were preserved at that revision. Its provenance
remains pinned to the earlier Figure 01 version, before the new guide labels.

## Manuscript integration

The [user's surgical change plan](manuscript_changes.md) is applied to the
draft: Figure 01 replaces its Figure 5 asset, with the 14M mechanistic core
preserved. Appendix tables add the 31M architecture, training settings,
analytic ceilings and all eleven measured conditions. The text distinguishes
fixed-pressure threshold comparisons, native/kernel baselines and timing
sessions. Figure 6 adds the actual 31M Base trajectory, generated as
[Figure 03](figures/03-base-model-training.pdf); see [O004](observations/004-base-training.md).

`04_manuscript_evidence.py` verifies and writes `data/manuscript-evidence.json`;
`05_base_training.py` verifies and plots all four Base histories. The manuscript
copies Figure 01 to `figures/02-14m-70m-latency-quality.pdf` and Figure 03 to
`figures/19-base-model-optimization.pdf`, retaining stable include paths.
Compile `manuscript/draft/main.tex` with latexmk, then run
`06_verify_manuscript.py` to check the embedded table, unchanged evidence and
compiled references. The verification record is
`data/manuscript-verification.json`; [O003](observations/003-manuscript-integration.md)
maps all 17 requested edits to evidence, preserved material and caveats.

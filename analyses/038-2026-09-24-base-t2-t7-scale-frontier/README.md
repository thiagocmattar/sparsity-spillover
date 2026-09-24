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
not a fitted scaling law. No manuscript edit or finding promotion.

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
compact layout of `manuscript/draft/figures/02-14m-70m-latency-quality.pdf`.
Its approved title is "Quality–latency trade-offs across model scales".
Dashed blue T2/Ph and solid orange T7/Ph curves use circles,
triangles and squares for 14M, 31M and 70M. Three vertical gray guides mark
the respective Base losses. Kappa labels, the workload subtitle and the legend
footer are omitted. The user corrected the earlier 30M label to 31M; all
36 measured coordinates and the Pareto set are unchanged.

The [two-panel companion](figures/02-absolute-relative-scale-frontier.pdf)
retains the absolute comparison in panel (a). Panel (b) uses only T2/Ph,
with paired boxplots of `L - L_Base` and `t - t_PyTorch_Base` for each model
size. Each box summarizes the five kappa settings. Purple loss-change boxes
use the left Y axis; teal latency-change boxes use the right Y axis, in
milliseconds. Both axes are linear and zero-aligned. Boxes span the middle
50%, whiskers use 1.5 IQR, and outliers remain visible. Generate
it with `python analyses/038-2026-09-24-base-t2-t7-scale-frontier/03_absolute_relative.py`
from the repository root. See [O002](observations/002-absolute-relative.md)
for its caption and Base-reference provenance, and
`data/absolute-relative-figure.json` for exact values. The single-panel PDF
and its source coordinates are preserved unchanged.

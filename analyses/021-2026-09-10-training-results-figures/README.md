# Analysis 021 - Training-results figures

Rework the manuscript's figures from retained evidence, starting with an
introduction-facing Pythia-14M quality-sparsity overview. The author approved
this new analysis and the first figure on 10 September 2026. No new training,
checkpoint evaluation, or measurement is required.

## Figure 01

[Quality-sparsity trade-offs](figures/01-14m-quality-sparsity.pdf) retains
26 trained checkpoints, omits naive L1, and shows both baseline and ReLU
post-hoc clipping. Direct annotations highlight the local-pressure regime and
the matched high-threshold pressure addition. Vertical guides expose the
12.83% and 29.95% architectural reach.

The compact revision is 5.5 by 3.35 inches (1.64:1), 34% shorter than the first
version. The one-row legend sits below the axes, preserving the preceding
version's data-region height. Inside the plot, labels identify baseline, ReLU,
ReLU plus pressure, the two reach guides and the 27.48% endpoint; the only
prose annotation reports the matched pressure effect. There is no internal
title, clipping text, comparison arrow or utilization callout. Only the local
pressure label retains a leader line.
Four-site thresholds act at a,m,h,z; seven-site thresholds add Q/K/V.
The caption defines these sites. Solid, dashed and dotted paths respectively
denote training without pressure, training with pressure and post-hoc clipping.
"Pressure" means orthogonal L1 throughout this selected view. Stronger markers
emphasize measured checkpoints over their connecting lines.

The orange annotation reports the matched seven-site pressure addition at
threshold 0.5: +12.1 percentage points of sparsity for +0.13 loss. The caption
identifies the gray/green post-hoc sweeps and reports that the 27.48% endpoint
uses 91.8% of seven-site reach.

Sources are the tracked Analysis 018 `figure_data.json` and Run 030
`results/clipping-points.json`. The compact
[figure data](data/14m-quality-sparsity.json) retain all 46 selected records,
pooled integer counters, coverage, recipe identifiers, and source hashes.
Eight clipping evaluations lie above the displayed loss range; their measured
coordinates remain in the figure paths and data. Earlier analysis artwork and
the manuscript's full recipe-level PDF are preserved.

The manuscript retains the first version from commit `56d4ec6`, shown as
Figure 2 on page 5 of its 26-page PDF. At the author's request, this compact
revision changes only Analysis 021; the draft copy, caption and `main.pdf`
have not been updated. The current analysis PDF therefore differs from the
manuscript copy until a later adoption.

## Reproduction and checks

From the repository root:

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/01_quality_sparsity.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_overview.py -q
```

The focused checks cover exact source selection and coordinates, complete
validation, integer pooling, both unclipped p=0 measurements, and independent
ceiling accounting. All three checks passed. All 46 measurements are unchanged
from the preceding version. The compact PDF was rendered and visually checked;
all fonts are embedded. The annotation values, page dimensions, text bounds
and placement of the legend outside the axes were also verified.
See [O001](observations/O001-14m-quality-sparsity.md) for the caption,
results and interpretation limits.

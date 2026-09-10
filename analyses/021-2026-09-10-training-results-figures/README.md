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
12.83% and 29.95% architectural ceilings.

Short labels use the introduction's thresholding, attention and pressure terms.
"Thresholds" includes both FFN and attention projections; "+ attention" adds
Q/K/V operands. This avoids calling the projection-site recipe FFN-only.
"Pressure" means orthogonal L1 throughout this selected view.

Sources are the tracked Analysis 018 `figure_data.json` and Run 030
`results/clipping-points.json`. The compact
[figure data](data/14m-quality-sparsity.json) retain all 46 selected records,
pooled integer counters, coverage, recipe identifiers, and source hashes.
Eight clipping evaluations lie above the displayed loss range; their measured
coordinates remain in the figure paths and data. Earlier analysis artwork and
the manuscript's full recipe-level PDF are preserved.

The draft adopts this figure in `training-results.tex`, with a matching caption
and a hash-identical copy in `draft/figures/`. The rebuilt 26-page `main.pdf`
shows it as Figure 2 on page 5.

## Reproduction and checks

From the repository root:

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/01_quality_sparsity.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_overview.py -q
```

The focused checks cover exact source selection and coordinates, complete
validation, integer pooling, both unclipped p=0 measurements, and independent
ceiling accounting. All three checks passed. The final PDF and draft pages
4-6 were rendered and visually checked; fonts are embedded, references resolve,
and the only LaTeX box warning is the existing underfull box on page 3.
See [O001](observations/O001-14m-quality-sparsity.md) for the caption,
results and interpretation limits.

# Pythia-70M sparsity, latency and quality

User-requested two-panel counterpart to Analysis033's 14M overview:
(a) model-wide sparsity versus latency; (b) validation loss versus latency.
No new training or timing is performed. The combined figure is now installed
in the manuscript at `fig:70m-quality-latency`; see the
[integration and checked claims](observations/003-manuscript-integration.md).

- [Publication PDF](figures/01-70m-sparsity-latency-quality.pdf).
- [14M and 70M on one loss-latency panel](figures/02-14m-70m-latency-quality.pdf).
- [Caption, method and interpretation](observations/001-two-panel.md).
- [Exact coordinates, backend identities and source hashes](data/figure-data.json).

The figure shows 26 checkpoints: Base and five threshold settings for each of
T2/Ph, T4/Ph, T7/Ph, T4/Pall and T7/Pall. It omits GeLU-to-ReLU and post-hoc
clipping, matching the focused 14M presentation. T2/Ph is blue, T7/Pall orange,
Base hollow charcoal, and the other recipes gray. There are no subtitles or
extra recipe labels, comparison arrows or point annotations.

Latency follows the existing 70M manuscript figure: Base uses native PyTorch
(1.611048 ms), explicitly identified in the legend and horizontal guides;
the trained recipes use opt073. This differs from Analysis033's custom-kernel
14M Base marker. The plot retains 25 Run045 points and the Run047 T2/Ph
kappa=0.5 endpoint, without session rescaling or an extra session marker.
The original Run045 Base remains the displayed reference.

The combined figure overlays the 40 focused 14M points and these 26 70M points
with logarithmic validation loss and latency axes. Each Base checkpoint is
shown under both kernel and native PyTorch execution: 68 execution points for
66 checkpoints. Circles denote 14M and squares 70M, with the same recipe colors
and gray context. Hollow Base markers denote kernels and smaller filled Base
markers denote PyTorch; all four have separate legend entries. The nearly
coincident 14M Base markers retain their actual coordinates. All existing
measurements are retained. See
[observation 002](observations/002-combined-scales.md) and
[combined data](data/combined-figure.json) for provenance and interpretation.

Quality is ordinary final-checkpoint FP16 validation over all 338 complete
2,048-token blocks from 500 MiniPile validation documents; 1,444 tail tokens
are excluded. Model-wide sparsity is the pooled integer zero-product count
divided by the full-model product count, including the dense output head.
Timings use RTX5090, BF16, batch one, 2,048 tokens and full output logits,
with 64 inputs x seven passes x three processes per checkpoint.

Reproduce from the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/034-2026-09-21-70m-latency-quality-frontier/01_plot.py
.venv/Scripts/python.exe -X utf8 analyses/034-2026-09-21-70m-latency-quality-frontier/02_combined.py
```

The script checks source hashes, all 37 upstream evidence hashes, cohort and
threshold coverage, backend qualification, displayed latency identities and
integer-pooled sparsity. It verifies that every point is visible in both panels.
The publication output is PDF only; raster layout checks live under `tmp/pdfs/`.
Analysis030 supplies the measurements; Analysis033 supplies the plot style.
The second script verifies all 66 checkpoint records, the 64 intervention points
and four Base execution references, backend qualification, axis coverage and
preservation of existing PDFs. Analysis024 supplies the paired 14M Base timings;
the 70M Base timings come from the existing Run045 record.

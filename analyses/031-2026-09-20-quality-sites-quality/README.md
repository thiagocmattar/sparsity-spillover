# Quality, conditional site savings, and scale: introduction figure

Approved three-panel figure:

- (a) Pythia-14M quality-sparsity trade-off.
- (b) Table 2's conditional saved time by sparsification site.
- (c) Pythia-70M quality-sparsity trade-off.

The side panels contain only the seven matching recipes: Base, GeLU-to-ReLU,
T2/Ph, T4/Ph, T7/Ph, T4/Pall and T7/Pall. Every pressure recipe has the same five
thresholds, giving 27 trained checkpoints per size, with a single shared legend.
The original Base/ReLU clipping curves are retained in both quality panels;
there is no rotated Post-hoc annotation. The different quality and sparsity axis
ranges preserve each size's established view. The approved revision removes
the vertical ceiling guides and the explanatory footer from the graphic.

The middle panel uses unrounded Table 2 data from Run037, checks its source
differences and process spans, and labels the six bars with the same one-decimal
values as the manuscript. Negative saved times remain visible. Q/K are one joint
QK skipping control; v is the PV skipping control. The effects are conditional
on all other paths remaining on, not additive allocations of total savings.
Whiskers show process-extrema differences, not confidence intervals.

The PDF is `figures/01-quality-sites-quality.pdf`. Figure data, source hashes and
verification are under `data/`; the caption and scientific limits are in
[observation 001](observations/001-three-panel-preview.md).
The approved PDF is installed as `manuscript/draft/figures/24-quality-sites-quality.pdf`
at `fig:quality-sparsity-overview`. The caption follows the revised introduction's
14M exploration, h/z execution diagnosis and 70M comparison, and defines the
conditional bars and whiskers. The author's surrounding prose is preserved.
The manuscript copy and supplementary data are hash-linked in their SOURCES.json
manifests. A clean 21-page compilation resolves all references with no overfull
boxes; the figure and caption on page 2 were visually inspected. See
`data/manuscript-verification.json` for the checked source and PDF hashes.

Reproduce:

```powershell
.venv/Scripts/python.exe analyses/031-2026-09-20-quality-sites-quality/01_figure.py
python analyses/031-2026-09-20-quality-sites-quality/02_verify.py
```

The figure uses the stable Analysis027 14M and Analysis029 70M canonical quality
records. The later Analysis030 optimized-latency integration leaves these
quality/sparsity coordinates unchanged and is not needed for this display.
No new training, kernel measurement, or cloud work occurs. Verification checks
all matching memberships, unchanged source coordinates, six Table 2 values,
hashes, PDF text bounds and fonts; PyMuPDF renders a temporary raster for visual
review. Only the PDF is retained as the publication artifact.

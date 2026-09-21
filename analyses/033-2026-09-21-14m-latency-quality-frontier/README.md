# Pythia-14M latency-quality frontier

User-requested figures from existing measurements, following the manuscript
figure style. No training or timing experiment is performed. The approved
three-panel figure is also installed in the manuscript at
`fig:quality-sparsity-overview`, with an updated introduction caption.

- [Focused overview: Base, T2/Ph and T7/Pall](figures/01-14m-latency-quality-frontier.pdf).
- [Sparsity versus latency, with the same highlights](figures/03-14m-sparsity-latency.pdf).
- [Three-panel figure: sparsity, conditional savings and quality](figures/04-14m-quality-savings-sparsity.pdf).
- [Version 2: latency on X, validation loss on Y](figures/01-14m-latency-quality-frontier-v2.pdf).
- [Close-up near Base validation loss](figures/02-14m-latency-quality-frontier-near-base.pdf).
- [Exact coordinates, excluded checkpoint identities and provenance](data/frontier.json).
- [Observations and captions](observations/INDEX.md).

The user requested a simpler version of the initial 45-point preview: remove
naive L1, remove the Pareto overlay, and make recipe connections more subtle.
The shared data contain the manuscript's 41-checkpoint, ten-recipe cohort.
The original overview now excludes GeLU-to-ReLU as well, showing 40 checkpoints
across nine recipes. Base, T2/Ph and T7/Pall are emphasized; the other recipes
are gray and unlabeled. The four excluded naive-L1 checkpoint identities and
the overview's omitted ReLU identity remain recorded in the data.
OL1 remains included. All points use the retained K050 latency grid, including
Base's specialized latency. Run048's separate execution controls are not
substituted into this cross-recipe grid. Post-hoc clipping is outside scope.

The focused overview uses the 70M manuscript figure's blue T2/Ph and orange
T7/Pall, with 1.6 pt lines, 5.8 pt markers and white marker edges. Base has a
9 pt hollow marker and a labeled vertical loss guide. Other recipes use
gray #9DA3AB, 0.85 pt lines at 65% opacity and smaller markers. Point callouts
are removed; the legend labels only Base and the two highlighted recipes.
No Pareto overlay is drawn. Both axes are minimized.

The sparsity-latency figure uses the same 40 points, highlights and gray context,
with model-wide sparsity (%) on X and full-model latency (ms) on Y. Sparsity is
recomputed from pooled integer zero-product counts and the full-model product
denominator, including the dense output head. Its guide marks Base sparsity.

The three-panel figure orders the focused views as (a) sparsity and latency,
(b) manuscript Figure 24's six conditional time-savings controls, and
(c) the latency-quality trade-off. A single dashed arrow connects T7/Pall
to the gray T7/Ph point at the same kappa=0.5: 27.48% versus 16.66% sparsity,
with retained latencies of 0.473366 and 0.473888 ms. It uses the manuscript's
panel proportions and typography, one shared legend, and no subtitles. The
middle panel measures T7/Pall at kappa=0.5 in a separate controlled session;
its savings are neither additive nor a decomposition of the historical grid
latencies. Its caption and interpretation are in observation 005, and exact
coordinates and provenance are in [data/three-panel.json](data/three-panel.json).

The close-up and transposed v2 are unchanged from the earlier simplified
version: 18 visible points and 41 points respectively, retaining all recipe
colors and selected dose callouts. They still include the ReLU control.

Quality uses ordinary final-checkpoint validation over all 338 complete blocks
from 500 MiniPile documents, excluding the 1,444-token tail. Latencies are
geometric means of synchronized host timings: RTX5090, BF16, batch one,
2,048 tokens, full vocabulary logits, 64 inputs x seven passes x three
independent processes. The shared data contain Run029:31, Run033:5, Run041:4
and Run044:1 checkpoints; the focused figures omit Run029's ReLU control.
No session normalization is applied.
This comparison does not establish significance for small differences
or generalize beyond the measured workload and single training seed.

Sources are [Analysis028's complete results](../028-2026-09-20-70m-optimized-grid/observations/001-matched-70m-grid.md)
and [Analysis027's manuscript figure](../027-2026-09-20-run044-manuscript/observations/001-run044-integration.md).
Analysis030's `02_plot.py` and `23-70m-quality-sparsity-native-latency.pdf`
supply the focused overview's highlight geometry; their hashes are recorded.

Reproduce from the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/033-2026-09-21-14m-latency-quality-frontier/01_plot.py
.venv/Scripts/python.exe -X utf8 analyses/033-2026-09-21-14m-latency-quality-frontier/02_three_panel.py
```

The script verifies pinned source hashes, the 45-checkpoint source inventory,
all 41 source coordinates, exclusion of the four naive-L1 points and the
overview's ReLU control, dose-group coverage and timing-session membership. Publication outputs
are PDF only; layout renders are temporary QA artifacts outside this analysis.
The second script verifies the shared 40-point cohort, pooled sparsity and all
six conditional savings and whisker spans against the original Run037 summary.
It preserves the four standalone PDFs.

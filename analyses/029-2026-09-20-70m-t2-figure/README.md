# 70M figure with the complete T2/Ph grid

The user requested that `figures/23-70m-quality-sparsity-native-latency.pdf`
include the completed 70M T2 results. This analysis adds Run043's four thresholds
and Run046's kappa=0.5 endpoint to the existing Analysis024 figure: 27 trained
checkpoints in both panels, plus the unchanged 20 Base/ReLU clipping settings
in the quality panel only. No new measurements or cloud work are performed.

`01_collect.py` verifies the original figure sources, joins final checkpoint
identities to all 15 new full-validation timing processes, recomputes pooled
logical sparsity from integer counts, and checks geometric-mean latency against
the retained raw samples. `02_plot.py` preserves the existing plot's dimensions,
axes, data and styling, adds the T2/Ph style used in the 14M figure and its exact
15.443084% analytic reach guide, and expands the legend to seven recipes.

The original 22 timings and native Base reference come from Run035. The added
T2 points use the same frozen `k050-70m-v2` in later Run043 and Run046 RTX5090
sessions. Cross-session absolute differences are descriptive; the historical
1.663883297ms Base line is not a matched denominator for the new T2 points.
Their same-checkpoint native latencies and paired ratios are retained separately
in the JSON. The separate Analysis028/Run045 optimized-grid work was in progress
at this update; its unpublished measurements are not inputs to this figure.

The publication PDF and its JSON are copied to `manuscript/draft/figures/` and
`manuscript/draft/supplementary-data/`. The figure caption alone is revised for
27 checkpoints and the session distinction; unrelated author edits are preserved.
The compiled `manuscript/draft/main.pdf` is not rebuilt by this figure-only task.
The author's uncommitted manuscript reorganization moves this figure from
`experimental-study.tex` into `training-results.tex`. The working caption is
updated at its new location; the scoped commit updates the same caption at its
committed location, preserving the author's reorganization as uncommitted work.

Reproduce with:

```powershell
.venv/Scripts/python.exe analyses/029-2026-09-20-70m-t2-figure/01_collect.py
.venv/Scripts/python.exe analyses/029-2026-09-20-70m-t2-figure/02_plot.py
python analyses/029-2026-09-20-70m-t2-figure/03_verify.py
```

The verifier checks figure/source hashes, all 27 point identities, unchanged
historical values and clipping settings, complete threshold coverage, PDF text
bounds and embedded fonts, and renders a temporary PNG for visual review.
The temporary raster is not a publication artifact. The available MiKTeX
`pdftoppm` required external setup, so the installed PyMuPDF renderer was used.

See [the observation and caption](observations/001-70m-t2-figure.md) and
`data/verification.json` for results and verification.

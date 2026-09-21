# Pythia-14M latency-quality frontier

User-requested figures from existing measurements, following the manuscript
figure style. No training, timing experiment or manuscript edit is performed.

- [All 41 manuscript checkpoints](figures/01-14m-latency-quality-frontier.pdf).
- [Version 2: latency on X, validation loss on Y](figures/01-14m-latency-quality-frontier-v2.pdf).
- [Close-up near Base validation loss](figures/02-14m-latency-quality-frontier-near-base.pdf).
- [Exact coordinates, excluded checkpoint identities and provenance](data/frontier.json).
- [Observations and captions](observations/INDEX.md).

The user requested a simpler version of the initial 45-point preview: remove
naive L1, remove the Pareto overlay, and make recipe connections more subtle.
Both figures now show the main manuscript's 41-checkpoint, ten-recipe cohort.
The four excluded naive-L1 checkpoint identities remain recorded in the data.
OL1 remains included. All points use the retained K050 latency grid, including
Base's specialized latency. Run048's separate execution controls are not
substituted into this cross-recipe grid. Post-hoc clipping is outside scope.

Validation loss and latency are both minimized. Thin recipe lines (0.85 pt,
40% opacity) connect separately trained dose settings; checkpoint markers
retain their full color. No Pareto line, rings or legend item is drawn.
The close-up displays 18 of the same points. Selected operating points retain
their recipe/dose labels, and the vertical guide marks Base validation loss.
Version 2 transposes the full overview's axes and uses a horizontal Base-loss
guide; the measurements and simplified styling are unchanged.

Quality uses ordinary final-checkpoint validation over all 338 complete blocks
from 500 MiniPile documents, excluding the 1,444-token tail. Latencies are
geometric means of synchronized host timings: RTX5090, BF16, batch one,
2,048 tokens, full vocabulary logits, 64 inputs x seven passes x three
independent processes. The four sessions contribute Run029:31, Run033:5,
Run041:4 and Run044:1 checkpoints. No session normalization is applied.
This comparison does not establish significance for small differences
or generalize beyond the measured workload and single training seed.

Sources are [Analysis028's complete results](../028-2026-09-20-70m-optimized-grid/observations/001-matched-70m-grid.md)
and [Analysis027's manuscript figure](../027-2026-09-20-run044-manuscript/observations/001-run044-integration.md).
Their inputs and historical palette, line patterns and font conventions are
retained, with lighter connections and a compact two-row legend.

Reproduce from the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/033-2026-09-21-14m-latency-quality-frontier/01_plot.py
```

The script verifies pinned source hashes, the 45-checkpoint source inventory,
all 41 displayed manuscript coordinates, exclusion of exactly the four naive-L1
points, dose-group coverage and timing-session membership. Publication outputs
are PDF only; layout renders are temporary QA artifacts outside this analysis.

# Analysis 011 - Pythia selected ladder through 410M

## Question

How do the verified Pythia-410M A0/A1-H post-hoc TEAL frontiers and trained
A4-OL1/A7-OL1 ladders compare on validation loss versus measured `R_model`, and
which parts of the 14M/70M ordering persist at 410M?

## Sources and method

The analysis reduces verified Runs 014, 015, 018, and 019 plus the verified
14M, 70M, and 410M A0/A1-H TEAL artifacts. Every point covers all 338 complete
2,048-token blocks from all 500 MiniPile validation documents (692,224 input
tokens), with the 1,444-token tail excluded and reported. Fractions are rebuilt
from integer counts where available.

For every trained point, the plotted validation loss is taken from the same
eager full-validation logical-product pass that produced `R_model`. This is a
stricter pairing than Analysis 010's use of the separately retained final-loss
pass; both losses remain in `figure_data.json`, and the differences are small.
TEAL points already pair loss and logical counts within one evaluation.

The figure style deliberately follows Analysis 010: the same colorblind-safe
family palette, marker families, dose-order lines, typography, legend language,
loss ceiling at 6, and explicit off-scale exits. Figure 1 isolates 410M. Figure
2 uses marker fill/edge to distinguish 14M, 70M, and 410M while preserving
family color and shape.

## Result

At 410M, the trained ladders reach `R_model=71.5914%` for A4-OL1 and
`80.6155%` for A7-OL1 at `kappa=0.5`; A7-OL1 has both lower paired loss and
higher `R_model` at that endpoint. Unlike 14M and 70M, A7-OL1 also dominates
A4-OL1 at `kappa=0` at 410M. Thus the high-dose A7-over-A4 ordering persists,
but the zero-threshold A4-over-A7 ordering does not. The complete numeric result
and sitewise exact-zero masses are in `tables.md`.

## Limits

This is one seed and one MiniPile pass per scale. Lines connect intervention
dose or clipping-target order and are not fitted curves. TEAL is post-hoc
evaluation-only clipping. `R_model` is logical zero-product opportunity, not
removed FLOPs or measured speedup. Mixed Run 019 GPU hardware affects execution
provenance and throughput, not the declared scientific factors. No finding or
manuscript claim is promoted by this analysis.

## Reproduction

```powershell
.\.venv\Scripts\python.exe analyses/011-2026-09-03-pythia14m-70m-410m-selected-ladder/01_build.py
.\.venv\Scripts\python.exe -m pytest analyses/011-2026-09-03-pythia14m-70m-410m-selected-ladder/test_build.py -q
```

# Analysis 011 - Pythia selected ladder and A0 optimization through 410M

## Question

How do the verified Pythia-410M A0/A1-H post-hoc TEAL frontiers and trained
A4-OL1/A7-OL1 ladders compare on validation loss versus measured `R_model`, and
which parts of the 14M/70M ordering persist at 410M?

For the matched A0 controls, how does the global full-model task-gradient L2
norm evolve with training tokens before and after the shared clip threshold?

## Sources and method

The frontier analysis reduces verified Runs 014, 015, 018, and 019 plus the
verified 14M, 70M, and 410M A0/A1-H TEAL artifacts. The A0 gradient diagnostic
additionally reduces the Run 004 A0 event stream. Every frontier point covers
all 338 complete 2,048-token blocks from all 500 MiniPile validation documents
(692,224 input tokens), with the 1,444-token tail excluded and reported.
Fractions are rebuilt from integer counts where available.

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

Figure 3 reads all 712 optimizer-boundary records from the A0 event stream at
each scale. Its 2-by-3 grid plots unsmoothed training task loss in the top row
and the global task-gradient L2 norm before and after clipping on shared
logarithmic axes in the bottom row. Every boundary covers 2,097,152 tokens; all
three streams end at 1,493,172,224 tokens. The clip threshold is 1.0
throughout, and no A0 boundary overflowed or skipped its optimizer update.

## Result

At 410M, the trained ladders reach `R_model=71.5914%` for A4-OL1 and
`80.6155%` for A7-OL1 at `kappa=0.5`; A7-OL1 has both lower paired loss and
higher `R_model` at that endpoint. Unlike 14M and 70M, A7-OL1 also dominates
A4-OL1 at `kappa=0` at 410M. Thus the high-dose A7-over-A4 ordering persists,
but the zero-threshold A4-over-A7 ordering does not. The complete numeric result
and sitewise exact-zero masses are in `tables.md`.

The A0 histories contain 5 clipped boundaries at 14M, 8 at 70M, and 56 at
410M. Their maximum pre-clip norms are 2.0019, 3.5132, and 26.9615,
respectively. Final boundary task losses are 5.2439, 3.9830, and 4.4483.

## Limits

This is one seed and one MiniPile pass per scale. Lines connect intervention
dose or clipping-target order and are not fitted curves. TEAL is post-hoc
evaluation-only clipping. `R_model` is logical zero-product opportunity, not
removed FLOPs or measured speedup. Mixed Run 019 GPU hardware affects execution
provenance and throughput, not the declared scientific factors. No finding or
manuscript claim is promoted by this analysis. The gradient measurements are
global norms rather than parameter-count-normalized quantities, so their
cross-scale difference documents the executed optimizer mechanics but does not
by itself identify an optimization defect.

## Reproduction

```powershell
.\.venv\Scripts\python.exe analyses/011-2026-09-03-pythia14m-70m-410m-selected-ladder/01_build.py
.\.venv\Scripts\python.exe -m pytest analyses/011-2026-09-03-pythia14m-70m-410m-selected-ladder/test_build.py -q
```

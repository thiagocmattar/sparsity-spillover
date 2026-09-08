# Figure 01 v2: all 14M training variants

## Question

How do all eight 14M training variants compare on the existing quality versus
model-wide sparsity axes, with post-hoc clipping applied only to A0?

## Method

Extend Figure 01 with A1-H-L1, A4 and A7, retaining the same colors, markers,
axis limits, title font and thin horizontal/vertical grids. Connect trained
settings in increasing lambda or kappa order, solid for unpressured recipes
and dashed for pressure variants. Keep A0's gray dashed clipping trajectory
and its actual measured p=0. The eight-entry training legend occupies two
rows below the plot; the A0 clipping key is beneath it.

## Coverage

All 30 trained 14M checkpoints: A0 (1), A1-H (1), A1-H-L1 (4), A1-H-OL1 (4),
A4 (5), A4-OL1 (5), A7 (5), A7-OL1 (5). All 30 lie within loss 5.04-6.15.
A0 contributes ten clipping evaluations, six visible within this loss window.
One initialization/data-order seed, 712 training updates, complete validation
over 500 MiniPile documents packed into 338 blocks of 2,048 tokens, excluding
the 1,444-token tail. No new evaluation or training is performed.

## Figure and caption

[Publication PDF](../figures/01-v2-14m-overview.pdf)

**Quality-sparsity trade-offs for all Pythia-14M training variants.** Filled
markers show all 30 trained checkpoints across A0, A1-H, A1-H-L1, A1-H-OL1,
A4, A4-OL1, A7 and A7-OL1. Local L1/OL1 vary lambda over .05, .1, .5 and 1;
A4/A7 variants vary kappa over 0, .01, .05, .1 and .5. Multisite OL1 fixes
lambda and trust budget at 1. Solid curves denote unpressured training
variants; dashed colored curves denote pressure variants. The gray dashed
open-marker path applies evaluation-only clipping at a,m,h,z to A0, ordered
by target p=0,.1,...,.9; six of ten points lie within the displayed range.
Both axes use complete-validation measurements. Connections order evaluated
settings, not fitted Pareto envelopes or attainable intermediate models.
Logical sparsity counts zero-operand products and does not measure speedup.

## Result

The added 14 checkpoints expose the naive-L1 versus OL1 local comparison
and the pressure-free A4/A7 curves alongside their OL1 counterparts.
All original coordinates and styles are preserved. The original Figure 01
and the manuscript's copy remain unchanged; this v2 is a separate alternative.

## Caveats

This is a one-seed descriptive view. Some nearby settings overlap; no jitter
is introduced. Clipping targets are repeated evaluations, not training
replicates. Complete clipping trajectories for every model, including the
high-loss points outside this window, remain in Run 030.

## Source and verification

Run `04_overview_v2.py`; it reads [figure_data.json](../figure_data.json) and
calls `plots.py:overview(all_variants=True)`. The existing default rendering
remains byte-identical. [Verification](../overview-v2-verification.json)
checks every plotted coordinate (30 trained plus ten clipped), the v2 legend
bounds, font embedding and unchanged original-PDF hash. The 5.5-by-4.6-inch
PDF was rendered and visually inspected; there are no overlapping legends.

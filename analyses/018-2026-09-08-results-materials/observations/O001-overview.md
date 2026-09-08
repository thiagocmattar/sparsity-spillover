# Quality-sparsity frontiers

## Question

How do the OL1 recipes and subsequent clipping compare with the A0/A1-H controls at 14M?

## Method

Show A0, A1-H, A1-H-OL1, A4-OL1 and A7-OL1. Connect every retained training setting in increasing lambda or kappa order. Draw each matching clipping trajectory with a fine, faint dashed line and small open markers, ordered by target p. Preserve actual measured p=0 coordinates; trained markers render above clipping paths. A1-H-L1, A4 and A7 are omitted from this view at the user's request. Their measurements remain in the complete data release.

## Coverage

Five families, 16 trained checkpoints and 160 clipping evaluations: A0 (1 checkpoint), A1-H (1), A1-H-OL1 (4), A4-OL1 (5), and A7-OL1 (5). Every source has p=0,.1,...,.9. The 5.04-6.15 loss window contains all 16 trained endpoints and 116 clipping evaluations; 44 clipping evaluations exceed it. Run 030 retains the full 30-checkpoint/300-point 14M cohort and its full-range PDF. One seed, 712 training updates, and complete validation over 500 documents packed into 338 blocks, with the 1444-token tail excluded. Clipping acts at a,m,h,z after trained gates, preserving A7 query/key/value gates.

## Figure and caption

[Publication PDF](../figures/01-14m-overview.pdf)

Quality-sparsity frontiers. The 14M comparison shows A0/A1-H controls and the A1-H-OL1, A4-OL1 and A7-OL1 recipes. Large filled markers identify 16 trained endpoints; bold curves connect every evaluated training setting within each OL1 recipe. Faint dashed paths with small open markers trace 160 post-hoc clipping evaluations from those same checkpoints, preserving measured p=0 and following p=0,.1,...,.9. Colors and symbols identify the source family. Local OL1 uses lambda=.05,.1,.5,1; A4/A7-OL1 uses kappa=0,.01,.05,.1,.5. The quality-focused loss window contains 116 clipping evaluations. The compact legend is below the plot. Connections order measured settings, not interpolated or fitted Pareto envelopes. The full cohort and loss range are retained in Run 030.

## Result

The selected families retain the low-loss local-OL1 regime and the higher-sparsity A7-OL1 regime. A7-OL1 at kappa=.5 has 27.4827% model-wide sparsity at loss 5.8294; clipping at p=.8 reaches 28.2258% at loss 6.0221. Removing displayed families does not change measurements or full-cohort frontier membership.

## Caveats

This is a selected-family, one-seed view. The full-cohort numerical frontier in tables/frontiers.md still includes the omitted families and is not restricted to this display. Repeated targets may coincide because of natural zero mass. No jitter or artificial p=0 anchors are introduced. Small p=0 numerical differences are not substantive improvement claims. The finite uniform grid establishes neither optimal clipping allocation, attainable intermediate models nor runtime gains.

## Source script and evidence

`plots.py:overview`, invoked by `01_build.py`; `evidence.py:load_evidence` declares the 21 displayed series. [Overview series](../tables/overview-series.md) records all 176 input coordinates. [figure_data.json](../figure_data.json) and [Run 030](../../../runs/030-2026-09-08-all-models-posthoc-clipping/results/README.md) retain the complete measurements, identities and counts.

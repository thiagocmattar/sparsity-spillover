# Quality vs. model-wide sparsity frontier for train-time and post-hoc interventions (Pythia-14M)

## Question

How do the OL1 training recipes compare with the A0/A1-H controls and post-hoc clipping of A0 at 14M?

## Method

Show A0, A1-H, A1-H-OL1, A4-OL1 and A7-OL1. Connect every retained training setting in increasing lambda or kappa order. Show post-hoc clipping only for A0, using a clearly visible grey dashed line and open markers ordered by target p. Preserve actual measured p=0 coordinates; trained markers render above clipping paths. A1-H-L1, A4 and A7 are omitted from this view at the user's request. Their measurements remain in the complete data release.

## Coverage

Five training families and 16 trained checkpoints: A0 (1), A1-H (1), A1-H-OL1 (4), A4-OL1 (5), and A7-OL1 (5). Only A0's ten clipping evaluations are drawn, at p=0,.1,...,.9. The 5.04-6.15 loss window contains every trained endpoint and six A0 clipping evaluations. All 300 clipping measurements and the full-range 14M PDF remain in Run 030. One seed, 712 updates, and full validation over 500 documents packed into 338 blocks, excluding the 1444-token tail. The A0 post-hoc operator clips at a,m,h,z.

## Figure and caption

[Publication PDF](../figures/01-14m-overview.pdf)

Quality vs. model-wide sparsity frontier for train-time and post-hoc interventions (Pythia-14M). Large filled markers identify the 16 trained endpoints from A0, A1-H and the three OL1 recipes. Bold curves connect every evaluated training setting. The grey dashed open-marker path shows post-hoc clipping of A0 only, preserving its measured p=0 and following p=0,.1,...,.9. Local OL1 uses lambda=.05,.1,.5,1; A4/A7-OL1 uses kappa=0,.01,.05,.1,.5. Six of the ten A0 clipping coordinates fall inside the displayed loss window. Connections order evaluated settings, not fitted Pareto envelopes or attainable intermediate models. Full clipping results for every checkpoint remain in Run 030.

## Result

The selected training families retain the low-loss local-OL1 regime and the higher-sparsity A7-OL1 regime. A7-OL1 at kappa=.5 reaches 27.4827% model-wide sparsity at loss 5.8294. The A0 clipping curve supplies the evaluation-only comparison. Display selection does not change measurements or full-cohort frontier membership.

## Caveats

This is a selected-family, one-seed view. The full-cohort numerical frontier in tables/frontiers.md still includes the omitted families and is not restricted to this display. Repeated targets may coincide because of natural zero mass. No jitter or artificial p=0 anchors are introduced. Small p=0 numerical differences are not substantive improvement claims. The finite uniform grid establishes neither optimal clipping allocation, attainable intermediate models nor runtime gains.

## Source script and evidence

`plots.py:overview`, invoked by `01_build.py`; `evidence.py:load_evidence` declares the six displayed series. [Overview series](../tables/overview-series.md) records all 26 input coordinates. [figure_data.json](../figure_data.json) and [Run 030](../../../runs/030-2026-09-08-all-models-posthoc-clipping/results/README.md) retain the complete measurements, identities and counts.

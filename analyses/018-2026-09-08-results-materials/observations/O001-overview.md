# 14M training and clipping trade-offs

## Question

How do train-time sparsification recipes and subsequent clipping shape the observed quality-sparsity trade-off?

## Method

Keep all 30 trained endpoints, connecting each family in increasing lambda or kappa order, including dominated points. Draw the 15 retained post-hoc trajectories behind them with thin, faint lines and small open markers, each ordered by clipping target p. Preserve the actual measured p=0 coordinates; never replace them with the separately measured training endpoints. Trained markers render above the clipping paths. Dashed training curves identify L1/OL1 pressure; solid curves identify unpressured recipes.

## Coverage

Eight trained families and 15 clipped source checkpoints, each evaluated at p=0,.1,...,.9: 30 training endpoints plus 150 clipping evaluations. Clipping sources are A0 (1), A1-H (1), local L1 (4), local OL1 (4), and A4 (5). Corrected A4-OL1, A7 and A7-OL1 have no retained clipping sweeps. The 5.04–6.15 loss window contains every trained endpoint and 100 clipping evaluations; all 150 clipping evaluations remain visible over the complete range in Figure 06. One matched seed and 712 training updates. Validation uses all 500 documents packed into 338 complete 2048-token blocks, excluding the 1,444-token tail.

## Figure and caption

[Publication PDF](../figures/01-14m-overview.pdf)

14M quality-sparsity trade-offs. Large filled markers show the 30 final trained endpoints. Bold curves connect every evaluated training threshold or pressure value within its recipe: solid for unpressured recipes and dashed for L1/OL1. Thin faint curves and small open markers trace all retained uniform post-hoc clipping sweeps for 15 source checkpoints from A0, A1-H, A1-H-L1, A1-H-OL1 and A4. Clipping colors follow the source family; each trajectory preserves its measured p=0 endpoint and follows p=0,.1,...,.9. A4/A7 training thresholds are kappa=0,.01,.05,.1,.5; local pressure weights are lambda=.05,.1,.5,1. The single panel uses a quality-focused loss window; Figure 06 shows the complete clipping loss range. The compact legend is below the plot. Connections order measured settings, not an interpolated or fitted Pareto frontier. Corrected A4-OL1 and A7 variants have no retained clipping sweeps.

## Result

Local L1 at lambda=1 has the lowest trained loss (5.1023) at 3.9493% model-wide sparsity. Post-hoc clipping of this checkpoint gives 4.7982% at loss 5.1070 (p=.1) and 5.6441% at loss 5.1441 (p=.2). A7-OL1 at kappa=.5 reaches 27.4827% at loss 5.8294. Corrected A4-OL1 has no globally nondominated trained endpoint within this 30-condition pool.

## Caveats

This is a one-seed descriptive comparison. Missing clipping sweeps prevent identifying an optimal joint recipe across every trained family. Some points overlap; no jitter or coordinate adjustment is introduced. A0/A1-H each have one trained endpoint. Separately measured clipping p=0 and canonical training endpoints have small numerical differences. Curves do not establish attainable intermediate models or runtime gains. Strict evaluated nondomination is tabulated separately in tables/frontiers.md.

## Source script and evidence

`plots.py:overview`, invoked by `01_build.py`. `evidence.py:load_evidence` declares all 23 series. Supporting evidence: tables/all-trained-endpoints.md, tables/all-clipping-points.md, tables/frontiers.md and tables/overview-series.md. Integer counts, source identities and exact values remain in [figure_data.json](../figure_data.json).

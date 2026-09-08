# 14M recipe overview

## Question

Which quality–sparsity regimes do the 30 included trained recipes occupy?

## Method

Select every included 14M trained endpoint. Use paired eager loss and count-derived model-wide sparsity. Draw one scatter with consistent recipe color/marker identity; preserve exact frontier membership only in the table.

## Coverage

30 trained conditions, eight recipe families, one matched seed and 712 updates. Unless the section specifies runtime timing inputs, validation covers all 500 documents packed into 338 complete 2048-token blocks, excluding the 1,444-token tail.

## Figure and caption

[Publication PDF](../figures/01-14m-overview.pdf)

Quality–sparsity trade-offs in the 14M study. Each marker is one final trained condition; symbols identify recipe families. All 30 included endpoints are shown, with no connecting curves or interpolated frontier. A4/A7 families use κ=0,.01,.05,.1,.5; local L1/OL1 use λ=.05,.1,.5,1. See the endpoint table for individual doses.

## Result

Local L1 at λ=1 has the lowest observed loss (5.1023) at 3.9493% sparsity. A7-OL1 at κ=.5 reaches 27.4827% at loss 5.8294. Corrected A4-OL1 has no globally nondominated endpoint within this 30-condition pool.

## Caveats

This is a one-seed descriptive comparison. The low-threshold cluster overlaps; no data-dependent jitter is introduced. Frontier selection reuses validation data. Clipping is reported separately in Figures 03/06.

## Source script and evidence

`plots.py:overview`, invoked by `01_build.py`. Supporting evidence: tables/all-trained-endpoints.md and tables/frontiers.md.
All source identities, integer counts and exact values are retained in [figure_data.json](../figure_data.json).

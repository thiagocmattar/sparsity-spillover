# 14M recipe overview

## Question

Which quality–sparsity regimes do the 30 included trained recipes occupy?

## Method

Select every included 14M trained endpoint. Use paired eager loss and count-derived model-wide sparsity. Keep all trained markers and connect each recipe's own nondominated set. Add separate A0/A1-H post-clipping frontier curves. Do not pool trained recipes or clipping controls into one envelope.

## Coverage

30 trained conditions, eight recipe families, and two clipping controls with ten targets each. The 5.04–6.15 loss window excludes eight high-loss clipping evaluations from view; their values remain in Figure 03 and the tables. One matched seed and 712 training updates. Unless the section specifies runtime timing inputs, validation covers all 500 documents packed into 338 complete 2048-token blocks, excluding the 1,444-token tail.

## Figure and caption

[Publication PDF](../figures/01-14m-overview.pdf)

Quality–sparsity trade-offs in the 14M study. Each marker is one final trained condition; symbols identify recipe families. All 30 trained endpoints are shown. Each legend entry has an independent evaluated frontier; solid curves connect nondominated points within a trained recipe and dotted open-marker curves connect the A0/A1-H clipping frontiers. There is no global envelope. The clipped-control loss range above 6.15 is reported in Figure 03. A4/A7 families use κ=0,.01,.05,.1,.5; local L1/OL1 use λ=.05,.1,.5,1. See the endpoint table for individual doses.

## Result

Local L1 at λ=1 has the lowest observed loss (5.1023) at 3.9493% sparsity. A7-OL1 at κ=.5 reaches 27.4827% at loss 5.8294. Corrected A4-OL1 has no globally nondominated endpoint within this 30-condition pool.

## Caveats

This is a one-seed descriptive comparison. The low-threshold cluster overlaps; no data-dependent jitter is introduced. Frontier selection reuses validation data. A series' frontier may contain only one point. Connections do not establish attainable intermediate operating points. The full clipping range is reported in Figures 03/06.

## Source script and evidence

`plots.py:overview`, invoked by `01_build.py`. Supporting evidence: tables/all-trained-endpoints.md, tables/frontiers.md and tables/overview-series-frontiers.md.
All source identities, integer counts and exact values are retained in [figure_data.json](../figure_data.json).

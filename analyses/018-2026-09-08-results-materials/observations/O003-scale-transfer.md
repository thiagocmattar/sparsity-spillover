# Scale transfer with clipping controls

## Question

Which recipe relationships survive larger models, and what does ceiling normalization explain?

## Method

Plot five A4-OL1/A7-OL1 doses and ten targets per clipped A0/A1-H control at each scale. Top uses raw loss and raw sparsity. Bottom uses loss minus unmodified same-scale A0 and U_arch. Clipped controls use the declared evaluation-site ceiling {a,m,h,z}, including p=0.

## Coverage

90 plotted evaluations: 30 trained recipe endpoints plus 60 clipped-control evaluations across 14M/70M/410M. Each p=0 is the actual sweep measurement. Unless the section specifies runtime timing inputs, validation covers all 500 documents packed into 338 complete 2048-token blocks, excluding the 1,444-token tail.

## Figure and caption

[Publication PDF](../figures/03-scale-transfer-and-ceilings.pdf)

Selected recipes and uniform clipping across model sizes. Columns denote size. Top: absolute validation loss versus raw model-wide sparsity. Bottom: loss relative to unmodified same-size A0 versus 100 R_model/R_model_max. Solid curves connect trained κ=0,.01,.05,.1,.5 endpoints; dotted open-marker curves connect p=0,.1,…,.9 clipping evaluations. Curves only order evaluated points. Clipped A0/A1-H use the reach of their clipping sites a,m,h,z, not the source checkpoint topology. Both rows retain every value. Dashed vertical lines in the first row mark the A4/clipping-site and A7 theoretical ceilings, identified in the shared legend.

## Result

At κ=.5 A7-OL1 has lower loss and greater raw sparsity than A4-OL1 at every scale. At κ=0 that ordering holds only at 410M. A4-OL1 has higher utilization of its own narrower ceiling in all 15 matched pairs.

## Caveats

Recipe transfer, not replication of an isolated pressure effect: larger scales lack no-pressure A4/A7. Equal tokens are unequal tokens/parameter; 410M uses a different peak LR. U_arch is not a size-invariant quality score or speedup and need not be bounded by one. Full clipping losses compress fine trained differences, which the tables retain.

## Source script and evidence

`plots.py:scaling`, invoked by `01_build.py`. Supporting evidence: tables/scale-paired-recipes.md, tables/all-clipping-points.md and tables/normalization-audit.md.
All source identities, integer counts and exact values are retained in [figure_data.json](../figure_data.json).

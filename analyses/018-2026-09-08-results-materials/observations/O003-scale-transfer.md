# Scale transfer with A0 clipping and an A7 reference

## Question

Which recipe relationships survive larger models, and what does ceiling normalization explain?

## Method

Plot five A4-OL1/A7-OL1 thresholds and ten A0 clipping targets at each scale. Top uses raw loss and raw sparsity, from zero to the same-size A7 ceiling. Bottom uses loss minus unmodified same-scale A0 and a common A7-reference utilization for all three curves: `U_arch(A7) = R_model / R_model_max(A7)`. Equivalently, divide the pooled zero-product count by `338 * A7.reachable_product_count`. The A7 ceilings are 29.9524%, 49.4239% and 87.2452% at 14M, 70M and 410M. Source-specific `U_arch` values in the evidence bundle and tables are retained separately; Figure 03 computes its A7-reference coordinate without changing them.

## Coverage

60 distinct evaluations: 30 trained recipe endpoints plus 30 A0 clipping evaluations across 14M/70M/410M, each shown in both rows (120 plotted coordinates). Each p=0 is the actual sweep measurement. Validation covers all 500 documents packed into 338 complete 2048-token blocks, excluding the 1,444-token tail.

## Figure and caption

[Publication PDF](../figures/03-scale-transfer-and-ceilings.pdf)

Quality vs. model-wide sparsity frontier across model sizes (Pythia-family). Columns denote size. Top: absolute validation loss versus raw model-wide sparsity, with each x axis ending at its A7 ceiling. Bottom: loss relative to unmodified same-size A0 versus 100 R_model/R_model_max(A7), using the same denominator for every curve within a column. Solid curves connect trained A4-OL1/A7-OL1 endpoints at κ=0,.01,.05,.1,.5; dotted open-marker curves connect A0 clipping evaluations at p=0,.1,…,.9. Curves only order evaluated points; all selected values are shown. A4/clipping-site and A7 theoretical ceilings remain as vertical guides in the top row, identified in the compact legend below the grid. The A4 guide is not the bottom-row denominator.

## Result

At κ=.5 A7-OL1 has lower loss and greater raw sparsity than A4-OL1 at every scale. At κ=0 that ordering holds only at 410M. The common positive A7 denominator preserves the within-size sparsity ordering in the bottom row. The separate topology-specific tables show higher utilization of A4's own narrower ceiling in all 15 matched pairs; that is a different normalization from this figure.

## Caveats

Recipe transfer, not replication of an isolated pressure effect: larger scales lack no-pressure A4/A7. Equal tokens are unequal tokens/parameter; 410M uses a different peak LR. A7 reaches every counted block operation, so this reference gives the block-only zero-product fraction, bounded by one. Its operation weights still differ across architectures; it is not a size-invariant quality score or speedup. Full clipping losses compress fine trained differences, which the tables retain.

## Source script and evidence

`plots.py:scaling`, invoked by `01_build.py`. Supporting evidence: tables/scale-paired-recipes.md, tables/all-clipping-points.md and tables/normalization-audit.md.
All source identities, integer counts and exact values are retained in [figure_data.json](../figure_data.json).

# Operation contributions

## Question

Which counted operations supply the high-threshold raw sparsity?

## Method

For each operation, divide pooled integer zero products by the same model-product denominator. Stack the six contributions for κ=.5 A4-OL1/A7-OL1 at each scale. Move per-operation zero rates and all other doses to the table.

## Coverage

Six trained endpoints; all six counted block operations, with the dense LM head retained only in the denominator. Unless the section specifies runtime timing inputs, validation covers all 500 documents packed into 338 complete 2048-token blocks, excluding the 1,444-token tail.

## Figure and caption

[Publication PDF](../figures/04-operation-accounting.pdf)

Operation contributions at κ=.5. In each size group, A4 and A7 denote A4-OL1 and A7-OL1. Stacked segments show zero products divided by all counted model products, in percentage points; the sum is measured model-wide sparsity. Solid colors identify QKV projections, FFN up/down, attention output projection, QK scores and PV. Analytic ceilings are shown separately in Figure 08.

## Result

The A4-OL1/A7-OL1 raw totals are 12.7134/27.4827%, 35.5962/40.6019% and 71.5914/80.6155% at 14M/70M/410M. Broader attention reach contributes alongside different zero rates in the linear operations.

## Caveats

Stack height combines learned zero rate with architecture/workload weighting. It is not runtime speedup. Some narrow segments are smaller than print resolution; exact counts/rates are retained in the table. The plot selects the high-threshold comparison, not the full dose response.

## Source script and evidence

`plots.py:operations`, invoked by `01_build.py`. Supporting evidence: tables/operation-counts.md and tables/architecture-counts.md.
All source identities, integer counts and exact values are retained in [figure_data.json](../figure_data.json).

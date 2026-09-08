# Paired intervention effects

## Question

What changes when a specified intervention is added to a matched reference?

## Method

Compute treatment minus reference for seven comparison blocks, retaining the same x positions across loss and sparsity rows. Numeric doses and per-group lambda/kappa labels sit below the plots. Two categorical annotation rows follow, with horizontal row labels at the left: Intervention: describes the action/sites, and Paired: names treatment minus reference. Pair expressions wrap onto two lines with the minus sign before the reference. The metric y-axes show only delta loss and delta model sparsity (pp). Dose increases left to right within each block, with ordinal rather than numerical spacing. All points use the same marker. No averaging across doses or seed inference.

## Coverage

29 contrasts from the included 14M cohort; matching initialization, order, validation cache, training tokens and seed checked in evidence.py. Unless the section specifies runtime timing inputs, validation covers all 500 documents packed into 338 complete 2048-token blocks, excluding the 1,444-token tail.

## Figure and caption

[Publication PDF](../figures/02-blocked-intervention-effects.pdf)

Paired 14M intervention effects. The upper panel shows validation-loss changes (negative is better); the lower panel shows model-sparsity changes in percentage points (positive is more logical zero-product opportunity). The Intervention: row describes each action. The Paired: row explicitly gives treatment minus reference: A1-H − A0, A1-H-L1 − A1-H, A1-H-OL1 − A1-H-L1, A4 − A1-H, A4-OL1 − A4, A7 − A4, and A7-OL1 − A7. A0 is the original GELU control, and A1-H is the ReLU control. Numeric labels give λ in the two local-pressure groups and κ in the four gated-recipe groups; the dash marks the dose-free ReLU replacement. L1-to-OL1 pairs share λ. The four-site and seven-site pressure additions and the query/key/value gate addition share κ between treatment and reference. The other references are fixed controls. Positions order doses without encoding numerical distances. The one-sided-gate action includes h; query/key gates are post-RoPE. Four-site and seven-site OL1 use fixed pressure weight λ=1 and trust budget 1. These 29 one-seed comparisons use separately trained conditions and are not cumulative updates or runtime measurements.

## Reference map

The displayed deltas are not uniformly relative to the immediately preceding
group or to A0. Each pair uses its declared reference:

| Displayed action | Reference | Treatment | Matching dose |
| --- | --- | --- | --- |
| Use ReLU at h | A0 (GELU) | A1-H | None |
| Add L1 at h | A1-H | A1-H-L1 | Reference is fixed; treatment varies λ |
| L1 → OL1 at h | A1-H-L1 | A1-H-OL1 | Same λ |
| Apply G+ at a,m,h,z | A1-H | A4 | Reference is fixed; treatment varies κ |
| Add OL1 at a,m,h,z | A4 | A4-OL1 | Same κ |
| Apply Gpm at q,k,v | A4 | A7 | Same κ |
| Add OL1 at all seven sites | A7 | A7-OL1 | Same κ |

## Result

At κ=.5, OL1 adds 2.4979 sparsity points to A4 for +.3783 loss, versus 12.0959 points to A7 for +.1265 loss. At κ=0/.01 the A4 comparison improves both metrics; pressure is not uniformly beneficial.

## Caveats

One seed; thresholds are treatments rather than replicates. A4→A7 at κ=0 has a small numerical residual despite identity Q/K/V gates. A1-H→A4 also changes the h boundary derivative; at nonzero κ it additionally thresholds h. This block reports the complete A4 intervention at each threshold. Comparing pressure responses across topologies changes pressure sites and normalization.

## Source script and evidence

`plots.py:effects`, invoked by `01_build.py`. Supporting evidence: tables/blocked-effects.md and tables/pressure-and-gate-effects.md.
All source identities, integer counts and exact values are retained in [figure_data.json](../figure_data.json).

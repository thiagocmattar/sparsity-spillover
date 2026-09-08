# Matched intervention effects

## Question

What changes when a specified intervention is added to a matched reference?

## Method

Compute treatment minus reference for seven comparison blocks, retaining the same x positions across loss and sparsity rows. The categorical x-axis is Intervention; numeric dose labels sit directly under each point, above a per-block lambda or kappa label and an action/site description. A separate Ref: row identifies each group's comparator. Both y-axes explicitly say "vs group reference." Dose increases left to right within each block, with ordinal rather than numerical spacing. All points use the same marker. No averaging across doses or seed inference.

## Coverage

29 contrasts from the included 14M cohort; matching initialization, order, validation cache, training tokens and seed checked in evidence.py. Unless the section specifies runtime timing inputs, validation covers all 500 documents packed into 338 complete 2048-token blocks, excluding the 1,444-token tail.

## Figure and caption

[Publication PDF](../figures/02-blocked-intervention-effects.pdf)

Matched 14M intervention effects. Both panels show treatment minus the group-specific reference printed in the Ref: row: validation loss above (negative is better) and model-sparsity change in percentage points below (positive is more logical zero-product opportunity). The seven actions are using ReLU at h, adding L1 at h, replacing L1 with OL1 at h, applying one-sided gates at a,m,h,z, adding OL1 at those four sites, applying symmetric gates at q,k,v, and adding OL1 at all seven sites. Numeric labels give λ in the two local-pressure groups and κ in the four gated-recipe groups; the dash marks the dose-free ReLU replacement. Ref: A0 is the original GELU control; A1-H is the ReLU control; L1(λ) is the h-only L1 recipe at the same λ; A4(κ) and A7(κ) are the corresponding unpressured gate recipes at the same κ. Positions order the doses without encoding numerical distances. The one-sided-gate action includes the change at h. Query/key gates are post-RoPE. In the four-site and seven-site OL1 treatments, pressure weight λ and trust budget are both fixed at 1. The references vary by group; the actions are not a sequence of cumulative updates. These are separately trained, one-seed comparisons, not runtime measurements.

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

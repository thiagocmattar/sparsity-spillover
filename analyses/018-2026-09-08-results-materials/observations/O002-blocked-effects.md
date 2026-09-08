# Matched intervention effects

## Question

What changes when a specified intervention is added to a matched reference?

## Method

Compute treatment minus reference for seven comparison blocks, retaining the same x positions across loss and sparsity rows. The categorical x-axis is Intervention; numeric dose labels sit directly under each point, above a per-block lambda or kappa label and the intervention name. Dose increases left to right within each block, with ordinal rather than numerical spacing. All points use the same marker. No averaging across doses or seed inference.

## Coverage

29 contrasts from the included 14M cohort; matching initialization, order, validation cache, training tokens and seed checked in evidence.py. Unless the section specifies runtime timing inputs, validation covers all 500 documents packed into 338 complete 2048-token blocks, excluding the 1,444-token tail.

## Figure and caption

[Publication PDF](../figures/02-blocked-intervention-effects.pdf)

Matched 14M intervention effects. Top: treatment-minus-reference validation loss in nats/token, where negative is better. Bottom: model-sparsity change in percentage points, where positive is more logical zero-product opportunity. The x-axis groups GELU→ReLU, adding L1 at h, replacing L1 with OL1 at h, A1-H→A4, adding OL1 to A4, A4→A7, and adding OL1 to A7. Numeric labels directly below the points give λ in the two local-pressure blocks and κ in the four A4/A7 blocks; the dash marks the dose-free GELU→ReLU comparison. Positions order the doses without encoding numerical distances. The A1-H→A4 block compares fixed A1-H with the complete A4 recipe at each κ=0,.01,.05,.1,.5, including its change at h. A4→A7 adds symmetric post-RoPE q/k and v gates at matched κ. In A4-OL1 and A7-OL1, pressure weight λ and trust budget are both fixed at 1. Each pair has an explicit reference/treatment ID in the table. These are separately trained, one-seed comparisons, not an additive decomposition or runtime measurements.

## Result

At κ=.5, OL1 adds 2.4979 sparsity points to A4 for +.3783 loss, versus 12.0959 points to A7 for +.1265 loss. At κ=0/.01 the A4 comparison improves both metrics; pressure is not uniformly beneficial.

## Caveats

One seed; thresholds are treatments rather than replicates. A4→A7 at κ=0 has a small numerical residual despite identity Q/K/V gates. A1-H→A4 also changes the h boundary derivative; at nonzero κ it additionally thresholds h. This block reports the complete A4 intervention at each threshold. Comparing pressure responses across topologies changes pressure sites and normalization.

## Source script and evidence

`plots.py:effects`, invoked by `01_build.py`. Supporting evidence: tables/blocked-effects.md and tables/pressure-and-gate-effects.md.
All source identities, integer counts and exact values are retained in [figure_data.json](../figure_data.json).

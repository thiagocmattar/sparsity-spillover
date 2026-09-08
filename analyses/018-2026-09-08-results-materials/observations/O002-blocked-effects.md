# Matched intervention effects

## Question

What changes when a specified intervention is added to a matched reference?

## Method

Compute treatment minus reference for seven comparison blocks, retaining the same x positions across loss and sparsity rows. Dose increases left to right within each block. No averaging across doses or seed inference.

## Coverage

25 contrasts from the included 14M cohort; matching initialization, order, validation cache, training tokens and seed checked in evidence.py. Unless the section specifies runtime timing inputs, validation covers all 500 documents packed into 338 complete 2048-token blocks, excluding the 1,444-token tail.

## Figure and caption

[Publication PDF](../figures/02-blocked-intervention-effects.pdf)

Matched 14M intervention effects. Top: loss change, where negative is better. Bottom: model-sparsity change in percentage points, where positive is more. Actions proceed from GELU→ReLU, adding L1 at h, replacing L1 with OL1, applying G+(x) at a,m,z, adding OL1 to A4, applying Gpm(x) at q,k,v, and adding OL1 to A7. The shared key identifies λ in the two local-pressure blocks and κ in the A4/A7 blocks; GELU→ReLU has no dose. The a,m,z addition uses κ=0. Each pair has an explicit reference/treatment ID in the table; the sequence is not an additive decomposition.

## Result

At κ=.5, OL1 adds 2.4979 sparsity points to A4 for +.3783 loss, versus 12.0959 points to A7 for +.1265 loss. At κ=0/.01 the A4 comparison improves both metrics; pressure is not uniformly beneficial.

## Caveats

One seed; thresholds are treatments rather than replicates. A4→A7 at κ=0 has a small numerical residual despite identity Q/K/V gates. A1-H→A4 also changes the h boundary derivative. Comparing pressure responses across topologies changes pressure sites and normalization.

## Source script and evidence

`plots.py:effects`, invoked by `01_build.py`. Supporting evidence: tables/blocked-effects.md and tables/pressure-and-gate-effects.md.
All source identities, integer counts and exact values are retained in [figure_data.json](../figure_data.json).

# Matched intervention effects: alternative orientation

## Question

What changes when a specified intervention is added to a matched reference?
This is the user-requested alternative layout of O002, using the same evidence.

## Method and coverage

Plot treatment minus reference for all 29 retained Pythia-14M comparisons.
Intervention is the shared categorical y-axis. The left column shows validation
loss changes; the right shows model-sparsity changes. Horizontal stems start
at zero, and every comparison has the same y position in both columns.
Dose increases downward within a group with ordinal spacing. A numeric column
beside the plots distinguishes each value; separate lambda and kappa labels
identify the parameter for each intervention group. Circles carry no dose code.

Initialization, realized training order, validation cache, training tokens and
seed are matched. Validation uses all 500 MiniPile documents packed into 338
complete 2,048-token blocks (692,224 input tokens), excluding the 1,444-token
tail. There is one seed. No model evaluation or new statistical analysis was run.

## Figure and caption

[Figure 02-v2](../figures/02-v2-blocked-intervention-effects.pdf)

Matched Pythia-14M intervention effects. Each row is one treatment-minus-reference
comparison: (a) validation loss in nats/token, where negative is better, and
(b) model-sparsity change in percentage points, where positive is more logical
zero-product opportunity. Interventions are grouped on the shared y-axis.
Numeric labels give lambda in the two local-pressure groups and kappa in the
four A4/A7 groups; the dash marks the dose-free GELU-to-ReLU comparison.
Doses increase downward without encoding numerical distance. A1-H to A4
compares fixed A1-H with the complete A4 recipe, including the change at h.
A4 to A7 adds symmetric post-RoPE query/key and value gates at matched kappa.
A4/A7 OL1 uses pressure weight lambda=1 and trust budget 1. These comparisons
use separately trained conditions and one seed; they are not an additive
decomposition or runtime measurements.

## Result

The numerical result is unchanged from O002. At kappa=0.5, adding OL1 to A4
adds 2.4979 sparsity points at a validation-loss cost of 0.378304, while adding
OL1 to A7 adds 12.0959 points at a cost of 0.126512. All doses and all seven
comparison groups remain visible. The alternative uses 5.5 by 5.5 inches.

## Caveats

Thresholds are treatment levels, not independent replicates. The small
A4-to-A7 residual at kappa=0 is not a benefit of identity gates. The A1-H-to-A4
comparison changes the h boundary derivative and, at nonzero kappa, thresholds
h. Pressure responses across topologies change pressure sites and objective
normalization. The shared horizontal scales compress the smallest L1-to-OL1
effects; their signs and precise values remain available in the numerical table.

## Source and verification

`plots.py:effects_v2`, also invoked by `01_build.py`, reads the existing
`contrasts` in [figure_data.json](../figure_data.json). The complete values and
reference/treatment IDs are in [blocked-effects.md](../tables/blocked-effects.md).
[O002](O002-blocked-effects.md) records the original orientation and scientific
interpretation. [VERIFICATION.md](../VERIFICATION.md) records coordinate,
label, rendering and preservation checks for this alternative.

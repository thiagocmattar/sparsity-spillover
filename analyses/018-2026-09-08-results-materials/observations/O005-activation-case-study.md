# Activation magnitude-band case study

## Question

How do comparable FFN zeros coexist with different attention opportunities in A4-OL1 and A7-OL1?

## Method

Select 14M A0 and the two OL1 recipes at kappa=0/.5. Reconstruct four disjoint magnitude bands from integer counts: exact zero; (0,.001]; (.001,.01]; and >.01. Pool counts and square sums across six layers before dividing or taking RMS. Use only six sites common to all five source diagnostics.

## Coverage

All 338 validation blocks and all six layers. Five unique checkpoints; the identical A0 diagnostic repeats in both rows. Each width-d site pools 531,628,032 values; h pools 2,126,512,128. Diagnostics are the retained FP16 eager activation passes; loss and logical counts come from their separately paired logical passes.

## Figure and caption

[Publication PDF](../figures/05-activation-mass-grid.pdf)

**Activation mass and scale at 14M.** Rows compare zero and high threshold; columns show FFN input m, hidden h, post-RoPE query/key, value, and attention output after W_o. Each series shows probability mass in four categorical |x| bands; connecting lines guide comparison and are not continuous density estimates. The shared y scale is logarithmic above .01% and linear near zero to display empty bands. Text reports count-pooled activation RMS. A0 is the same checkpoint in both rows. The final column is post-W_o attention_output, not the pre-W_o gated context z.

## Result

At kappa=.5, h is >99.8% zero under both recipes. A4-OL1 Q/K have 47.64/56.12% mass within .01 but <.001% exact zeros. A7-OL1 Q/K have 93.54/94.54% exact zeros and v has 98.71%. A4-OL1 query RMS is smaller (.146 versus .483), illustrating that small magnitude and exact-zero opportunity differ. At zero threshold, neither recipe creates substantial Q/K exact zeros.

## Caveats

No full histograms or signed samples are retained, so density shape, quantiles and within-band structure are unknown. A0 lacks a/z three-threshold statistics in this pass; those ports are not fabricated from another precision or pass. Separate activation and logical passes have slight numerical differences. This selected-recipe comparison does not isolate pressure placement or establish a causal spillover mechanism; RMS changes are not automatically distributional broadening.

## Source script and evidence

`plots.py:activations`; `evidence.py:mass_bands/load_evidence`; `01_build.py`; `tables/activation-statistics.md`. Run 004 A0 and Runs 014/015 OL1 activation_statistics.json.

Paths above are relative to the analysis root or repository root as named.
Complete count-derived values and source hashes are in [figure_data.json](../figure_data.json).

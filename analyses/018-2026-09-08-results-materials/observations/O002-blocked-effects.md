# Blocked matched intervention effects

## Question

Does the additional effect of pressure depend on the trained threshold and gate placement?

## Method

Compute child-minus-parent differences in paired final loss and count-derived R_model. Draw 25 contrasts in seven sequential blocks: GELU to ReLU; add local L1; replace local L1 with OL1; expand A1-H to A4 at kappa=0; add OL1 to A4; add Q/K/V gates to A4; add OL1 to A7. Five further h-only-to-four-site pressure contrasts are in the complete table.

## Coverage

Same-size initialization, ordered-schedule and validation-cache hashes agree across contrasts. One seed; complete 338-block validation; 712 boundaries and 1,493,172,224 training tokens. Lambda levels .05/.1/.5/1 and kappa levels 0/.01/.05/.1/.5 are treatment levels.

## Figure and caption

[Publication PDF](../figures/02-blocked-intervention-effects.pdf)

**Matched intervention effects at 14M.** Each x position identifies a treatment level within the named parent-to-child block, aligned vertically across loss (top) and model-wide sparsity (bottom). Negative loss differences indicate better quality; positive sparsity differences indicate more zero-product opportunity. Stems originate at zero. Alternating backgrounds separate comparisons with distinct comparators. Lambda labels apply to local-pressure blocks and kappa labels to gate blocks. These are separately trained conditions; the display is not a cumulative training sequence. The final table retains exact small differences not resolvable visually.

## Result

At A4 kappa=0/.01, OL1 improves both axes. At .5 it adds 2.4979 sparsity points at +.378304 loss. For A7, the zero-threshold OL1 effect worsens both axes; at .1 it adds 1.3717 points at +.000819 loss, and at .5 it adds 12.0959 points at +.126512 loss. Local OL1 is not uniformly better than naive L1: the lambda=1 local comparison favors naive L1 on both axes.

## Caveats

Fixed-topology pressure contrasts isolate addition of the named objective. Comparing pressure responses between A4 and A7 changes pressure sites and their equal-tensor weighting. Missing A7+OL1@four-sites prevents a fixed-objective placement interaction claim. The A4/A7 identity-gate kappa=0 residual is numerical/implementation discrepancy, not a gate benefit. One seed precludes significance or equivalence claims.

## Source script and evidence

`plots.py:effects`; `evidence.py:load_evidence`; `01_build.py`; `tables/blocked-effects.md` and `tables/pressure-and-gate-effects.tex`. Runs 004/009/011-015.

Paths above are relative to the analysis root or repository root as named.
Complete count-derived values and source hashes are in [figure_data.json](../figure_data.json).

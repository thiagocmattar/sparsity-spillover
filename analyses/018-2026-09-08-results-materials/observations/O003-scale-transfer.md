# Scale transfer and ceiling normalization

## Question

Which 14M recipe relationships persist at 70M/410M, and does dividing by the reach ceiling remove architecture effects?

## Method

Plot all five doses for A4-OL1 and A7-OL1 at three sizes. Subtract the same-size A0 paired loss for the vertical coordinate. Top row uses raw R_model; bottom uses observed zero products divided by selected-site reachable products, U_arch. Recompute all analytic counts from L,d,FFN width,T,V and compare to retained diagnostics. Record U_reach as a numerator-scope sensitivity check.

## Coverage

Thirty trained OL1 endpoints and six A0/A1-H controls. All endpoints use complete validation (338 blocks, 1,444-token tail excluded), one seed per size, and 1,493,172,224 training tokens. Controls are shown on raw panels; A0 normalization is undefined.

## Figure and caption

[Publication PDF](../figures/03-scale-transfer-and-ceilings.pdf)

**Selected-recipe transfer in raw and reach-normalized coordinates.** Columns show 14M, 70M and 410M. The y axis is final loss relative to that size's A0. Top panels display raw model-wide sparsity; colored vertical dotted lines mark each topology's analytic ceiling. Bottom panels divide by that recipe's own ceiling; the dotted line is a ratio of one. Points connect in kappa order 0, .01, .05, .1, .5; endpoint labels identify 0 and .5. A0 is omitted from normalized panels because its selected reach is zero. Ratio values need not be bounded by 100%, and different recipes use different denominators.

## Result

At kappa=.5, A7-OL1 improves both raw axes over A4-OL1 at every size. At kappa=0 the A4-over-A7 ordering at 14M/70M reverses at 410M. The A7 high-dose normalized values are 91.75%, 82.15%, 92.40%, despite raw values 27.48%, 40.60%, 80.62%. A4-OL1 has a higher fraction of its own narrower ceiling at all 15 matched recipe pairs.

## Caveats

Normalization changes the question and does not create a size-invariant quality score. Natural zeros outside selected reach remain in U_arch. Even block-only rates retain architecture-dependent operation weights. Larger cohorts lack no-pressure A4/A7 controls, so only recipe transfer is identified. Tokens/parameter and peak LR differ across sizes; no scaling law, seed uncertainty, or causal explanation of the 410M reversal is inferred.

## Source script and evidence

`plots.py:scaling`; `evidence.py`; `01_build.py`; `tables/scale-endpoints.tex`, `tables/scale-paired-recipes.md`, `tables/normalization-audit.md`. Runs 014/015/018/019 and Run 004 controls.

Paths above are relative to the analysis root or repository root as named.
Complete count-derived values and source hashes are in [figure_data.json](../figure_data.json).

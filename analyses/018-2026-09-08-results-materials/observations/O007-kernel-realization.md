# Qualified kernel realization on the included cohort

## Question

Can a specialized implementation realize measured inference gains on the included checkpoints?

## Method

Read Run 029 matched-retrospective-001.json. Keep final c01–c30 and recompute qualification counts, speedup summaries, ablation ratios and unweighted OLS with an intercept. Preserve qualified search progress on fixed c30; the search comparison set c01/c11/c25/c30 is already inside the included cohort.

## Coverage

One physical RTX5090; BF16 batch 1, uncached T=2048, full-vocabulary logits. Same-checkpoint native SDPA CUDA-graph denominator. Timings: 64 fixed inputs, seven paired passes, three fresh processes. Numerical qualification: all 338 validation blocks. Final cohort: 30 checkpoints; progress: 42 eligible proposals. Unless the section specifies runtime timing inputs, validation covers all 500 documents packed into 338 complete 2048-token blocks, excluding the 1,444-token tail.

## Figure and caption

[Publication PDF](../figures/07-kernel-realization.pdf)

Sparse kernel auto-research for full-model acceleration. Panel (a), “Search progress,” shows the best fully qualified incumbent on fixed A7-OL1 κ=.5 (c30), initialized at native 1×, across 42 eligible proposal ordinals. Panel (b), “Best sparse kernel vs. sparsity,” shows final K050 on the 30 included checkpoints. The shared recipe legend is below the plots and uses the same symbols/colors as Figure 01. The y-axis reports full-model speedup (×). Dashed line is an unweighted descriptive OLS fit with an intercept (R²=.8167); dotted horizontal lines mark native 1×. Model-wide sparsity uses the same canonical field as the training figures. Failed/unqualified timings are not plotted as usable speedups.

## Result

Qualified incumbent: 1.7830×. K050 qualifies on 30/30, geometric mean 1.2340×, range 1.0059–1.7832×. Sparse paths versus fused no-skip: geometric ratio 1.0432×, faster on 14/30. Disabling attention skipping improves every included checkpoint (geomean 1.2506×). P0 qualifies on four checkpoints only.

## Caveats

This revised 30-checkpoint summary supersedes the original 35-checkpoint statistics only within this analysis; the source run and manuscript remain untouched. Ablations compare separately measured native-normalized timings. Canonical sparsity is FP16 while execution is BF16. Quality, topology and weights vary, preventing a causal or equal-quality interpretation. No cached-decoding or larger-model transfer follows.

## Source script and evidence

`plots.py:kernels`, invoked by `01_build.py`. Supporting evidence: evidence.py:runtime_subset; tables/runtime-summary.md; Run 029 results/matched-retrospective-001.json.
All source identities, integer counts and exact values are retained in [figure_data.json](../figure_data.json).

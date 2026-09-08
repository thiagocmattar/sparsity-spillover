# Complete retained post-hoc comparison

## Question

Does the 14M narrative omit favorable trained-plus-clipped alternatives?

## Method

Overlay all 150 raw Analysis 006 clipping evaluations on all 35 trained endpoints and recompute the pooled nondominated set. Each of 15 source checkpoints has targets p=0,.1,...,.9. Use actual raw p=0 loss and counts rather than historical display anchors. Preserve the entire loss range in a second panel.

## Coverage

185 evaluated 14M points: 35 trained and 150 post-hoc. One training seed; all 338 validation blocks for every evaluation, 1,444-token tail excluded. Each clip calibrates per-site/layer magnitude thresholds on the first ten complete source-order training blocks; sites are a,m,h,z.

## Figure and caption

[Publication PDF](../figures/06-complete-posthoc-comparison.pdf)

**Complete available training-plus-clipping comparison.** Gray trajectories show 15 retained checkpoints evaluated under ten uniform clipping targets each. Colored trajectories retain the trained-recipe encoding of Figure 01. The dotted black envelope and open circles mark nondominated evaluated points across the full pool. The main panel focuses on quality; the right panel includes every high-loss point. Targets below natural zero mass can be no-ops. Lines do not imply a fitted frontier or attainable intermediate losses.

## Result

Clipping the local lambda=1 naive-L1 checkpoint adds low-loss operating points at p=.1/.2: 4.7982%/5.6441% sparsity with loss 5.106970/5.144074. Including historical A4-OL1[h] supplies strong intermediate points. A7/A7-OL1 still supply the high-opportunity end of the pooled observed frontier.

## Caveats

Clipping covers A0, A1-H, four L1 and four OL1 local-pressure checkpoints, and five unpressured A4 checkpoints. It does not cover A7 or corrected four-site A4-OL1, so this is the complete available pool rather than every possible composition. Uniform TEAL-style clipping is not the original greedy allocation method. Tiny p=0 versus trained-pass differences are numerical rather than substantive evidence.

## Source script and evidence

`plots.py:all_clipping`; `evidence.py:frontier/load_evidence`; `01_build.py`; `tables/all-clipping-points.md`, `tables/frontiers.md`. Analysis 006 teal_all_variants.json plus the 35 raw trained diagnostics.

Paths above are relative to the analysis root or repository root as named.
Complete count-derived values and source hashes are in [figure_data.json](../figure_data.json).

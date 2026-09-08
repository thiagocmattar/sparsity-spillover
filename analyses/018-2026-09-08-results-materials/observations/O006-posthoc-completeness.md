# Complete retained 14M clipping comparison

## Question

Which post-hoc quality–sparsity regimes exist in the available clipping evaluations?

## Method

Use all raw uniform-clipping evaluations across 15 source checkpoints and ten target sparsities. Group markers by the five source recipe families, retaining the full loss range. Do not connect source trajectories or overlay the trained overview.

## Coverage

150 actual 14M evaluations: A0/A1-H controls, four L1 doses, four local OL1 doses, and five A4 thresholds; p=0,.1,…,.9. Unless the section specifies runtime timing inputs, validation covers all 500 documents packed into 338 complete 2048-token blocks, excluding the 1,444-token tail.

## Figure and caption

[Publication PDF](../figures/06-complete-posthoc-comparison.pdf)

14M post-hoc clipping: 15 pretrained checkpoints, ten targets each. Source family counts are A0 (1), A1-H (1), A1-H-L1 (4), A1-H-OL1 (4), and A4 (5). Each open marker is one evaluated source-checkpoint/target pair; symbols and colors identify the source family. The plot includes the full loss range for all 150 evaluations. Every source checkpoint is evaluated at p=0,.1,…,.9 with fixed weights; these are 150 evaluations, not 150 trained models. Source λ/κ, clipping p and exact values are in the table. No curve or interpolated frontier is asserted; Figure 01 supplies the separate trained overview.

## Result

Clipping the λ=1 L1 model to p=.1/.2 gives 4.7982%/5.6441% sparsity at losses 5.1070/5.1441. Aggressive clipping frequently increases loss sharply; the retained alternatives matter to an overall comparison.

## Caveats

No retained post-hoc study of A7 or corrected A4-OL1 is available. Uniform TEAL-style clipping does not reproduce greedy TEAL allocation. Target p is a calibration target, not achieved model-wide sparsity. Overlapping scatter points remain individual table rows.

## Source script and evidence

`plots.py:all_clipping`, invoked by `01_build.py`. Supporting evidence: tables/all-clipping-points.md and tables/frontiers.md.
All source identities, integer counts and exact values are retained in [figure_data.json](../figure_data.json).

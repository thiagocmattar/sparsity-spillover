# 14M training and post-hoc clipping

## Question

How does uniform evaluation-only clipping extend each trained 14M checkpoint's quality versus logical-sparsity trade-off?

## Method

Keep the original weights and trained gates. Calibrate separate absolute-value thresholds at every layer's a,m,h,z sites using the first ten source-order training blocks. Apply targets p=0,.1,...,.9 after trained gates; preserve A7 query/key/value gates. Evaluate with FP32 parameters, FP16 autocast, eager attention, batch one and sequence length 2048. Pool integer logical-product counts before dividing. Retain actual measured p=0 coordinates and every target, including ties and dominated points. The normalization ceiling uses the union of retained trained gates and clipping sites.

## Coverage

All 30 manuscript-cohort 14M checkpoints and 300 clipping evaluations. One initialization/data-order seed (1234), step 712. Every point uses all 500 MiniPile validation documents: 338 complete blocks, 692224 input tokens, 691886 prediction tokens, and a 1444-token excluded tail. No new training. Source checkpoint hashes and training parameters are explicit in the input manifest and measurement release.

## Figure and caption

[Publication PDF](../figures/14m-posthoc-frontiers.pdf)

14M training and post-hoc clipping. Large filled markers identify the 30 canonical trained endpoints. Bold curves connect training settings within a recipe: solid for unpressured recipes, dashed for L1/OL1 pressure. Each thin, faint open-marker path follows one fixed checkpoint through all ten clipping targets in increasing p order, including its own measured p=0 evaluation. Colors and symbols identify the source recipe. All 300 clipping measurements and their complete loss range are included. Connections order evaluated settings; they do not assert attainable intermediate models or fitted Pareto envelopes. The numerical evaluated frontier is retained separately in frontiers.json.

## Result

The joint training-plus-clipping frontier has 29 records at 18 distinct (loss, sparsity) coordinates. Exact target ties remain separate records. Among clipped evaluations, the lowest loss is 5.102270 at 3.949330% model-wide sparsity (14M:A1-H-L1:1.0, p=0). The largest measured sparsity is 28.969358% at loss 6.584488 (14M:A7-OL1:0.5, p=0.9); this high-sparsity extreme is not a quality-preservation claim.

Clipping A7-OL1 kappa=.5 at p=.7 adds 0.15665 sparsity percentage points for 0.01619 loss versus its own p=0. Corrected A4-OL1 contributes no globally nondominated joint point. The only previously nondominated training endpoint displaced is A7-OL1 kappa=.1, through a tiny p=0 numerical difference.

## Caveats

One-seed descriptive comparisons, not independent replications across clipping targets. Calibration p is not achieved model-wide sparsity. Natural zero mass can make several targets identical. Small p=0 numerical differences can change strict frontier membership without constituting substantive improvement. The uniform grid does not establish optimal clipping allocation or reproduce greedy TEAL. Logical-product opportunity is not measured runtime speedup. Analysis 018 Figure 01 focuses on loss 5.04-6.15 and includes 222 of the 300 clipping coordinates in view; this full-range PDF includes all of them.

## Source script and evidence

`02_evaluate.py` evaluates the missing sweeps; `03_consolidate.py` reconciles all 540 measurements and computes strict frontier membership; `04_plot.py` generates this PDF. See [input manifest](../input-manifest.json), [complete measurement index](../results/README.md), [verification](../results/verification.json), and [frontier membership](../results/frontiers.json).

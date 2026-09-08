# 14M overview

## Question

Which complete recipes occupy different parts of the quality-sparsity frontier?

## Method

Read paired eager loss and integer logical-product counts for all 35 trained 14M conditions. Distinguish the main 30-condition ladder from five historical A4-OL1[h] conditions. Overlay the 20 raw A0/A1-H uniform-clipping evaluations. Compute nondominance by minimizing loss and maximizing measured R_model within each explicitly named trained pool.

## Coverage

All 500 validation documents; 338 complete 2048-token blocks; 692,224 inputs; 691,886 prediction targets; 1,444-token tail excluded. One seed, 712 optimizer boundaries, 1,493,172,224 training tokens per trained condition.

## Figure and caption

[Publication PDF](../figures/01-14m-overview.pdf)

**14M quality-sparsity overview.** Colors and markers identify trained recipes; connected points order tested pressure weights or thresholds. Gray dotted curves are evaluation-only uniform clipping of A0 and A1-H, with ten targets each. Black rings and thin envelopes identify nondominated trained conditions in the main ladder (solid) and after admitting historical h-only A4 pressure (dotted). Panel (a) focuses on loss 5.04-6.15; panel (b) retains the full loss range, including eight clipping points above panel (a). R_model is displayed as the draft's model-wide sparsity, S_model. Curves and envelope segments are visual guides through measured endpoints.

## Result

Local L1 at lambda=1 gives loss 5.102275 and 3.9493% S_model. A7-OL1 at kappa=.5 reaches 27.4827% at loss 5.829407. Corrected A4-OL1 has no globally nondominated endpoint among the main 30 trained conditions. Historical A4-OL1[h] supplies additional favorable intermediate points, including loss 5.195583 at 9.0356%.

## Caveats

Frontier membership depends on the candidate pool. It is descriptive and reuses validation for selection. An envelope segment does not establish an attainable interpolated operating point. OL1[h] is the realized Run 012 intervention, not corrected four-site A4-OL1. Figure 06 and the tables retain the additional trained-plus-clipped evaluations.

## Source script and evidence

`plots.py:overview`; `evidence.py:load_evidence/frontier`; generating command `01_build.py`. Runs 004/009/011-015; Analysis 006 raw clipping. See `figure_data.json` source hashes and `tables/frontiers.md`.

Paths above are relative to the analysis root or repository root as named.
Complete count-derived values and source hashes are in [figure_data.json](../figure_data.json).

# O004 - Complete fixed-budget 410M extension

## Question and method

How do selected pressured recipes behave at 410M under the same token budget?
Retain all 12 trained checkpoints and 20 uniform clipping evaluations from
Analysis 011. Each evaluation covers all 338 complete validation sequences
and excludes the 1,444-token tail. Use same-pass loss and pooled logical counts.

## Figure and caption

[Figure PDF](../figures/04-410m-complete-frontier.pdf).
All pressured A4/A7 thresholds and both full uniform-clipping paths at 410M.
The direction of the trained threshold response and its gap to A0 are visible.

## Result

Higher thresholds can reduce loss within the pressured families. At kappa=.5,
A7+OL1@7 reaches loss 5.1207 and opportunity 80.6155%, while the unmodified
A0 control has loss 4.5475. The shared 1.493-billion-token budget corresponds
to 3.7 tokens per parameter at 410M, versus 21.2 at 70M and 106.1 at 14M.
The selected A0 peak LR 3e-4 also has the lowest late training loss in the
executed three-rate screen (3e-4, 6e-4, 1e-3).

## Caveats and provenance

These observations do not identify the cause of the reversal. Additional
data-budget and optimization controls are needed before assigning it to
undertraining or interpreting it as a scale effect. Keeping the entire cohort
in the appendix preserves evidence that differs from the main 14M/70M pattern.

Source scripts: [evidence.py](../evidence.py), [plots.py](../plots.py),
[01_build.py](../01_build.py). Source reduction: Analysis 011, with exposure
context from Analysis 012. Tables: trained-410M.tex and clipping-410M.tex.
The learning-rate table is generated from Run 021's
[selection artifact](../../../runs/021-2026-09-03-pythia410m-a0-learning-rate-screen-resolved-lr/artifacts/selection.json),
which reuses the pinned Run 019 baseline and compares two new higher-rate arms.

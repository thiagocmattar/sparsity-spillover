# O002 - Training and clipping under explicit quality allowances

## Question and method

Which evaluated conditions maximize opportunity within a common loss
allowance? Use the complete 35-checkpoint 14M study and selected 12-checkpoint
70M study, plus both 10-target clipping paths per size. Targets are 0 to .9
in increments of .1; calibration uses ten training-cache blocks and fixed
per-layer/site thresholds at a,m,h,z. Validation uses all 338 complete blocks
and excludes the 1,444-token tail. Same-pass losses and pooled counts are used.

For each size and delta in {.05,.1,.25,.5,1}, select the maximum evaluated R_model
among points with loss <= that size's zero-clipped A0 loss + delta. Selection
uses the discrete grid. It is a descriptive validation-set summary.

## Figure and caption

[Figure PDF](../figures/02-training-and-clipping.pdf).
Complete uniform-clipping paths and trained conditions at 14M and 70M.
The A0 and ReLU controls coincide with each path's zero-target endpoint.
All trained families are drawn at 14M; only the two pressured A4/A7 families
are available at 70M. Lines give dose order, including dominated points.

## Result

| Added loss | 14M trained | 14M clipping | 70M trained | 70M clipping |
|---:|---:|---:|---:|---:|
| .05 | 9.34% | 2.55% | 0.00% | 7.41% |
| .10 | 9.34% | 4.42% | 0.00% | 7.41% |
| .25 | 11.80% | 5.27% | 10.07% | 19.99% |
| .50 | 15.39% | 6.13% | 10.07% | 22.52% |
| 1.00 | 27.48% | 6.99% | 30.57% | 25.11% |

Training extends the evaluated uniform-clipping frontier at both sizes.
The 70M clipping path is stronger under the tighter reported allowances;
the selected training grid is stronger at the larger allowance.
All selected condition identities are retained in figure_data.json.

## Caveats and provenance

Uniform quantile allocation is a TEAL-inspired comparator. Published TEAL's
blockwise optimized allocation is an untested stronger baseline. The 70M
training grid omits the local and unpressured branches available at 14M.
One validation set supplies the frontier and operating-point selection;
independent seeds and downstream evaluation are absent. Added cross-entropy
of 1 nat is a large quality allowance, so a frontier extension should be
read together with its quality cost.

Source scripts: [evidence.py](../evidence.py), [plots.py](../plots.py),
[01_build.py](../01_build.py). Source reductions: Analyses 008, 009, 011, 012.
Tables: loss-budgets.tex, trained-14M/70M.tex, clipping-14M/70M.tex.

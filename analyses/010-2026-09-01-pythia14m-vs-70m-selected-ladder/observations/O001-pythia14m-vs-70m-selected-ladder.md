# Pythia-14M versus 70M selected ladder

## Question

Does the selected validation-loss versus measured-`R_model` tradeoff persist
when the Pythia-14M endpoints are promoted to randomly initialized
Pythia-70M?

## Method and coverage

The analysis reduces 20 trained endpoints: A4-OL1 and A7-OL1 at kappa 0,
0.01, 0.05, 0.1, and 0.5 for each model scale. It also reduces 40 post-hoc
TEAL points: targets 0.0 through 0.9 for A0 and A1-H at both scales. Every
point uses all 338 complete 2,048-token validation blocks (692,224 input
tokens) from all 500 validation documents and records the excluded 1,444-token
tail. The trained endpoints use one matched seed, 712 optimizer boundaries,
and 1,493,172,224 training tokens per condition.

All plotted fractions are recomputed from pooled integer logical counts. The
reduction also reconciles per-site exact-zero counts and retains OL1
gradient-conflict and projection-step counts. Source-file hashes are embedded
in `figure_data.json`.

## Result

The qualitative frontier persists at 70M. At kappa 0, A4 dominates A7 at both
scales: it has higher `R_model` and lower validation loss. At kappa 0.5, A7
dominates A4 at both scales. The A7 kappa-0.1 point remains a favorable local
knee: relative to A7 kappa 0, it gains 4.743 percentage points of `R_model`
while lowering 14M loss by 0.05069, and gains 5.131 points while increasing
70M loss by only 0.00890.

Absolute opportunity changes with scale. Across kappa 0 to 0.5, A4 gains
4.903 points of `R_model` at 14M and 9.924 points at 70M; A7 gains 20.428 and
17.039 points, respectively. The endpoint loss increases are similar for A4
(0.5798 at 14M and 0.5843 at 70M) and smaller at 70M for A7 (0.3492 versus
0.2748). Post-hoc A0 and A1-H TEAL frontiers likewise retain the pattern of
initial opportunity gains at low loss cost followed by a steeper loss rise.

## Figure caption and legend

`figures/01-pythia14m-vs-70m-selected-ladder.pdf` is one absolute-coordinate
frontier: complete-validation loss against measured `R_model`. It overlays the
trained A4 and A7 ladders with the complete A0 and A1-H post-hoc TEAL
trajectories for both scales. The enlarged black-edged target-0 control points
are the final un-clipped checkpoints; their exact `R_model` and validation loss
appear in the figure callout. Color identifies the path, dashed circles identify
14M, and solid squares identify 70M. Lines guide dose order and are not fitted
response curves.

## Caveats

This is descriptive one-seed persistence across only two scales, not a scaling
law. Absolute loss is not directly comparable across model sizes as a treatment
effect. `R_model` is a logical-product opportunity and does not measure runtime
speedup. TEAL is post-hoc clipping, not training under the corresponding gate.
The 410M promotion remains unobserved.

## Sources

- Runs 014, 015, and 018 verification artifacts and canonical attempts.
- Analysis 005 `teal_frontier.json` and Run 018 `teal_frontiers.json`.
- Reduction and figure source: `01_build.py`.
- Complete numeric reduction: `figure_data.json` and `tables.md`.

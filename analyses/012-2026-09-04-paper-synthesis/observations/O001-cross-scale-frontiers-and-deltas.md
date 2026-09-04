# O001 - Absolute frontiers and within-family changes across scale

## Question

Which validation-loss--`R_model` relationships persist from Pythia-14M to 70M,
and which change in the one-pass Pythia-410M promotion?

## Method and coverage

Analysis 012 reuses Analysis 011's verified 30 trained and 60 post-hoc
endpoints. Each endpoint covers all 338 complete 2,048-token blocks from all
500 MiniPile validation documents (692,224 input tokens), with the 1,444-token
tail excluded and reported. Loss and `R_model` are paired within the same eager
logical-product pass. The three models share 712 optimizer boundaries and
1,493,172,224 training input tokens, but this corresponds to 106.142, 21.202,
and 3.684 tokens per parameter at 14M, 70M, and 410M.

The absolute figure uses common axes. The delta figure subtracts each trained
family's own `kappa=0` endpoint and each post-hoc family's own `p=0` endpoint.
Lines indicate tested dose order only.

## Result

The 14M and 70M panels share a qualitative trained pattern: moving from
`kappa=0` to `0.5` raises `R_model` but also raises loss in both A4-OL1 and
A7-OL1. At both sizes A4-OL1 dominates A7-OL1 at zero threshold, while A7-OL1
dominates A4-OL1 at `kappa=0.5`.

The 410M panel is not a vertically shifted copy. Its A0 loss (4.547456) is
worse than 70M (4.099766), its zero-threshold ordering reverses, and increasing
`kappa` from 0 to 0.5 lowers loss while raising `R_model` for both trained
families: A4-OL1 changes by -0.501109 loss and +33.2528 percentage points;
A7-OL1 changes by -0.305604 and +39.4940 points. In contrast, the post-hoc A0
and A1-H curves at all sizes retain the ordinary pattern of increasing loss as
more activations are clipped. The reversal is therefore specific to the
trained recipes under this run, not a generic consequence of the `R_model`
accounting.

Run 021's predeclared training-only selector retained the `3e-4` 410M baseline
over `6e-4` and `1e-3`. This rules out a simple upward peak-learning-rate repair
within that grid. It does not establish why the 410M trained dose response
reverses.

## Figure captions

**Figure 1.** Absolute validation loss versus measured `R_model` for Pythia-14M,
70M, and 410M. A0/A1-H lines are evaluation-only uniform clipping; A4-OL1 and
A7-OL1 lines are separately trained threshold doses. Upward boundary markers
show where post-hoc curves continue above the display range. Panel subtitles
report the unequal tokens-per-parameter exposure induced by the common token
budget.

**Figure 2.** Within-family changes in validation loss and `R_model`. Trained
rows are relative to each family's `kappa=0` endpoint; post-hoc rows are
relative to each checkpoint's `p=0` evaluation. The separation makes the 410M
trained reversal visible without concealing its weaker absolute A0 baseline.

## Caveats

There is one seed per scale, and training tokens are matched rather than
capacity-scaled. A4-OL1 versus A7-OL1 changes gate topology and pressure sites,
so it compares complete recipes. The result is not a scaling law, an identified
undertraining mechanism, or evidence of runtime speedup.

## Provenance

- Source script: `../01_build.py`
- Reduction: `../figure_data.json`
- Figures: `../figures/01-absolute-frontiers-by-scale.pdf` and
  `../figures/02-within-scale-deltas.pdf`
- Upstream verified reduction:
  `../../011-2026-09-03-pythia14m-70m-410m-selected-ladder/figure_data.json`

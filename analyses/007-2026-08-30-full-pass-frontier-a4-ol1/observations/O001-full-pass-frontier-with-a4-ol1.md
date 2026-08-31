# O001 - Full-pass frontier with A4-OL1

Correction (2026-08-31): Run 012's realized pressure capture was `h` only, not
`a,m,h,z`. The counts and losses below remain the historical Run 012 outputs,
but all mentions of A4-OL1 in the original observation denote the invalid
declared label. The scientifically correct series name is A4-Z + OL1@h; no
four-site conclusion follows.

## Question

Where do the five historical Run 012 A4-Z + OL1@h endpoints lie relative to the
trained full-pass frontier and TEAL-style post-hoc control clipping curves in
Analysis 005?

## Method and reduction

The figure reproduces the 15 trained endpoints and the visible post-hoc GeLU
and ReLU control curves from Analysis 005's immutable `figure_data_02.json`,
then adds the five verified historical Run 012 conditions. For Run 012, measured
`R_model` is recomputed as the summed integer zero-product count divided by the
integer model-product count. Exact-zero activation mass at `a`, `m`, `h`, and
`z` is recomputed from pooled integer counts over the complete validation pass;
layer or batch percentages are not averaged.

## Coverage

All plotted trained endpoints use seed 1 and all 338 complete 2,048-token
MiniPile validation blocks (692,224 input tokens from 500 documents). The
1,444-token incomplete tail is excluded. Each Run 012 condition trained for
712 optimizer steps. Post-hoc points retain Analysis 005's explicit loss-6
display cap.

## Caption and legend

Final validation loss is plotted against measured `R_model` in percent. Purple
hexagons are historical Run 012 A4-Z + OL1@h endpoints with one-sided
thresholds at the annotated `kappa` values and `h`-only orthogonal L1 pressure
at `lambda=1`. Green
diamonds are matched A4 thresholds without OL1. The other trained and post-hoc
series are unchanged from Analysis 005. Lines connect dose or target order
only.

## Result

Historical A4-Z + OL1@h improves both validation loss and measured `R_model`
relative to matched A4 at `kappa <= 0.1`. Its lowest loss is 5.195590 at `kappa=0.05`, with
`R_model=9.036%`. Raising `kappa` to 0.1 increases `R_model` to 9.343% while
loss rises slightly to 5.228687. At `kappa=0.5`, historical Run 012 reaches 10.227%
`R_model`, but loss worsens to 5.722666 and is 0.062986 above matched A4. The
matched OL1 advantage therefore contracts as threshold strength increases and
reverses in validation loss at the strongest threshold.

Across historical Run 012, exact-zero mass rises monotonically at all four gated sites.
At `kappa=0.05`, the count-pooled exact-zero masses are 51.791% (`a`), 52.391%
(`m`), 97.915% (`h`), and 88.307% (`z`). The near-saturation of `h` and `z` at
high thresholds is consistent with the additional `R_model` gain becoming small
even as validation loss degrades.

## Caveats

This is one seed, one Pythia-14M scale, and one MiniPile pass. Joint gates and
`h`-only OL1 pressure cannot be attributed site-by-site from this comparison.
The plot's connected lines are visual guides, not fitted response curves.
`R_model` is a logical exact-zero product opportunity and must not be read as
measured runtime speedup. Post-hoc points above loss 6 remain omitted exactly as
in the source figure.

## Provenance

- Source script: `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/01_build.py`
- Figure data: `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/figure_data.json`
- Source frontier: `analyses/005-2026-08-30-run004-controls-teal-posthoc/figure_data_02.json`
- Historical A4-Z + OL1@h verification: `runs/012-2026-08-30-pythia14m-full-pass-a4-ol1/artifacts/verification.json`
- Figure: `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/figures/01-full-pass-frontier-with-a4-ol1.pdf`
- Table: `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/tables.md`
- Consolidated finding:
  `research/findings/F001-a4-ol1-improves-moderate-threshold-frontier.md`

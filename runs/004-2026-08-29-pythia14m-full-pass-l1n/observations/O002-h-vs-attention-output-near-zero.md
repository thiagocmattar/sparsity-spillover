# O002 - Near-zero mass at h versus attention output

## Question

Does increasing ReLU L1N pressure at `h` change near-zero mass at the complete
attention output after `W_o` and before residual addition?

## Method and coverage

This post-hoc reduction uses the terminal activation diagnostics from all six
verified Run 004 conditions. It sums integer `|x| <= 1e-3` hit counts and
denominators over all six layers, checks those sums against the pooled integer
records, and divides once for both `h` and `attention_output`.

Every point covers seed 1234, final update 712, all 500 MiniPile validation
documents, all 338 complete 2,048-token blocks (692,224 input tokens), six
layers, and the declared 1,444-token excluded tail. `attention_output` is the
output of `attention.dense` (`W_o`) before it enters the residual addition; it
is not the attention head combination before `W_o`.

## Values

All values are pooled near-zero percentages at `epsilon = 1e-3`.

| Condition | `h` | Attention output after `W_o` |
| --- | ---: | ---: |
| GeLU control | 0.237736 | 0.701354 |
| ReLU control | 63.508542 | 0.480568 |
| ReLU L1N, lambda 0.05 | 73.501792 | 0.697766 |
| ReLU L1N, lambda 0.1 | 78.047723 | 0.801612 |
| ReLU L1N, lambda 0.5 | 88.579031 | 0.855134 |
| ReLU L1N, lambda 1.0 | 92.351206 | 0.724827 |

## Figure caption

Pooled near-zero mass after the attention output projection `W_o`, before
residual addition, against pooled near-zero mass at the directly pressured `h`
site, with `|x| <= 1e-3`. The blue open circle is the standalone GeLU control.
Orange open squares and the dashed guide connect the ReLU control through
increasing L1N pressure `lambda in {0.05, 0.1, 0.5, 1.0}`. The line indicates
dose order only.

## Result

Within the ReLU cohort, attention-output near-zero mass rises from 0.480568% at
control to 0.697766%, 0.801612%, and 0.855134% through lambda 0.5, then falls to
0.724827% at lambda 1.0 while `h` continues rising. Thus the output-site response
is substantial but non-monotonic at the largest tested pressure. The GeLU
control is 0.701354% at only 0.237736% `h` near-zero mass, reinforcing that the
two activation topologies cannot be read as one common dose curve.

## Caveats

This is one seed, one 14M scale, and one full MiniPile pass. Near-zero output
mass does not establish exact zero products, `R_model`, measured runtime gains,
or a causal mechanism. The connected ReLU points are a dose-order guide and no
uncertainty estimate is available from a single seed.

## Provenance

- Source script: `../07_plot_spillover_figures.py`
- Run verification: `../artifacts/verification.json`
- Source diagnostics: `../artifacts/attempts/*/diagnostics/activation_statistics.json`
- Count-first reduced data: `../artifacts/figure_data_attention_output.json`
- Figure: `../figures/02-h-vs-attention-output-near-zero.pdf`


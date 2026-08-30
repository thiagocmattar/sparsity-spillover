# O001 - Near-zero mass at h versus unpressured sites

## Question

As ReLU L1N pressure increases near-zero mass at the directly pressured `h`
site, does near-zero mass also move at the unpressured `q_post`, `k_post`, `v`,
and `m` sites?

## Method and coverage

This post-hoc reduction uses the terminal activation diagnostics from all six
verified Run 004 conditions. For each site and condition, it sums the integer
`|x| <= 1e-3` hit counts and denominators across the six layer rows, verifies
that they reproduce the stored pooled integer counts, and divides once. It does
not average percentages across layers or batches.

Coverage is identical for every point: seed 1234, final update 712, all 500
MiniPile validation documents, all 338 complete 2,048-token blocks (692,224
input tokens), six layers, and the declared 1,444-token excluded tail. The GeLU
cohort contains only its control; the ReLU cohort contains its control and four
L1N doses.

## Values

All values are pooled near-zero percentages at `epsilon = 1e-3`.

| Condition | `h` | `q_post` | `k_post` | `v` | `m` |
| --- | ---: | ---: | ---: | ---: | ---: |
| GeLU control | 0.237736 | 0.065518 | 0.059550 | 0.129597 | 0.070630 |
| ReLU control | 63.508542 | 0.050349 | 0.045815 | 0.132739 | 0.067001 |
| ReLU L1N, lambda 0.05 | 73.501792 | 0.056967 | 0.045499 | 0.149564 | 0.070568 |
| ReLU L1N, lambda 0.1 | 78.047723 | 0.055263 | 0.055462 | 0.155159 | 0.067589 |
| ReLU L1N, lambda 0.5 | 88.579031 | 0.057830 | 0.049814 | 0.180704 | 0.071537 |
| ReLU L1N, lambda 1.0 | 92.351206 | 0.051757 | 0.050614 | 0.185325 | 0.063650 |

## Figure caption

Pooled near-zero mass at four unpressured sites against pooled near-zero mass
at the directly pressured MLP hidden site `h`, with `|x| <= 1e-3`. The blue
open circle is the standalone GeLU control. Orange open squares and the dashed
guide connect the ReLU control through increasing L1N pressure
`lambda in {0.05, 0.1, 0.5, 1.0}`. Labels identify the control or pressure
weight; the connecting line is a dose-order guide, not an interpolation or
time trajectory.

## Result

Within the ReLU cohort, `h` near-zero mass rises from 63.508542% at control to
92.351206% at lambda 1.0. The clearest aligned movement is at `v`, which rises
from 0.132739% to 0.185325%. `q_post`, `k_post`, and `m` remain much smaller and
move non-monotonically. This is descriptive evidence of site-selective
spillover rather than a uniform attention-wide increase. The GeLU control is a
separate baseline point and does not define a GeLU dose response.

## Caveats

This is one seed, one 14M scale, and one full MiniPile pass. Near-zero mass is
not exact-zero mass, a logical-product opportunity, measured speedup, or proof
of a causal route. The visual line only orders the tested ReLU doses. No
uncertainty estimate is available from a single seed.

## Provenance

- Source script: `../07_plot_spillover_figures.py`
- Run verification: `../artifacts/verification.json`
- Source diagnostics: `../artifacts/attempts/*/diagnostics/activation_statistics.json`
- Count-first reduced data: `../artifacts/figure_data_sitewise.json`
- Figure: `../figures/01-h-vs-site-near-zero-grid.pdf`


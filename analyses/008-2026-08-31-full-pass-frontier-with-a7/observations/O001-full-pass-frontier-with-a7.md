# O001 - Full-pass frontier with A7 and A7-OL1

## Question

Where do corrected four-site Run 015 A4-OL1, the five verified Run 013 A7
endpoints, and the five matched Run 014 A7-OL1 endpoints lie on the full-pass
trained and post-hoc Pythia-14M frontier?

## Method and reduction

The figure retains Analysis 007's 15 unaffected trained endpoints and visible
post-hoc GeLU/ReLU control curves, removes its five historical Run 012 rows,
and imports the five corrected Run 015 four-site A4-OL1 rows verified by
Analysis 009. It then adds the five Run 013 `A7-Z-POST` conditions and the five
Run 014 conditions with the same mixed gates plus seven-site OL1. Measured
`R_model` and site exact-zero fractions are checked against their pooled integer
counts over all six layers and the complete validation pass.

## Coverage

All plotted trained endpoints use one matched seed and all 338 complete
2,048-token MiniPile validation blocks: 692,224 input tokens from 500 documents,
with the 1,444-token incomplete tail excluded. Each Run 013 and Run 014
condition trained for 712 optimizer steps and 1,493,172,224 input tokens.
All trained endpoints are shown through loss 6.1. Post-hoc points retain
Analysis 007's inherited loss-6 source filter.

## Caption and legend

Final validation loss is plotted against measured `R_model` in percent. Amber
stars are Run 013 A7 endpoints: one-sided thresholds at `a,m,h,z`, symmetric
thresholds at post-RoPE `q_post,k_post` and `v`, one common annotated `kappa`,
and no pressure objective. Sky-blue pentagons are matched Run 014 A7-OL1
endpoints, adding orthogonal L1 at all seven active sites with `lambda=1` and
trust budget `1`. Purple hexagons are corrected Run 015 A4-OL1, green diamonds
are A4, and the remaining trained and post-hoc series are inherited unchanged
from Analysis 007. Historical Run 012 A4-OL1[`h`] is excluded. The single panel
retains the complete range through the A7-OL1
`kappa=0.5` endpoint; the interleaved table supplies every matched `kappa` and
site value. Lines connect dose or clipping-target order only.

## Result

Corrected four-site A4-OL1 improves both matched axes over A4 at `kappa=0` and
`0.01`. At `kappa=0.05`, `0.1`, and `0.5`, it adds 2.2506, 2.4413, and 2.4979
percentage points of `R_model` for validation-loss costs of 0.055693, 0.128681,
and 0.378307. The historical Run 012 result is not used to characterize this
objective.

A7 increases measured `R_model` from 7.2177% at `kappa=0` to 15.3868% at
`kappa=0.5`. Its lowest final validation loss is 5.428681 at `kappa=0.1`, with
`R_model=10.4250%`. The strongest threshold extends substantially farther
right than the prior trained frontier but raises loss to 5.702923.

A7-OL1 increases measured `R_model` from 7.0542% at `kappa=0` to 27.4827% at
`kappa=0.5`. Its lowest final validation loss is 5.429497 at `kappa=0.1`, with
`R_model=11.7968%`. Relative to matched A7, the `kappa=0.1` row adds 1.3717
percentage points for `+0.000816` loss; the `kappa=0.5` row adds 12.0959 points
for a larger `+0.126466` loss. The zero-threshold A7-OL1 row instead loses
0.1634 points and raises loss by 0.011783, so the OL1 effect is not uniform.

Against matched A4, A7 at `kappa=0.01` improves validation loss by 0.007678 and
`R_model` by 0.2039 percentage points. At `kappa=0.05`, `0.1`, and `0.5`, A7
adds 0.9210, 1.4720, and 5.1713 percentage points of `R_model`, respectively,
with validation-loss costs of 0.003778, 0.009038, and 0.043244. The near-null
`kappa=0` pair differs by -0.002096 loss and +0.0056 percentage points of
`R_model`, consistent with the added symmetric gates being identities at zero
threshold within the numerical limits of independently scheduled workers.

The single table in `tables.md` interleaves every A7 and A7-OL1 endpoint and all
eight measured exact-zero site masses. At `kappa=0.5`, A7-OL1 reaches 93.5450%
(`q_post`), 94.5413% (`k_post`), and 98.7133% (`v`), compared with A7's
17.4347%, 18.1338%, and 31.2552%. The ungated post-`W_o`
`attention_output` remains effectively dense in exact-zero terms at every dose.

## Caveats

- One seed, one Pythia-14M scale, and one MiniPile pass do not establish
  replication or scaling.
- A7 adds Q/K/V gates jointly, so the comparison does not identify individual
  site effects.
- A7-OL1 pressures all seven sites jointly, so its matched effect cannot be
  attributed to one pressure site.
- Thresholded training changes forward values and gradient support.
- Connected lines are visual dose-order guides, not fitted response curves.
- `R_model` is logical exact-zero product opportunity, not removed FLOPs or
  measured sparse-kernel speedup.
- Post-hoc points above loss 6 remain omitted by Analysis 007's source filter;
  all trained endpoints are displayed.

## Provenance

- Source script: `analyses/008-2026-08-31-full-pass-frontier-with-a7/01_build.py`
- Figure data: `analyses/008-2026-08-31-full-pass-frontier-with-a7/figure_data.json`
- Source frontier: `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/figure_data.json`
- Corrected A4-OL1 reduction: `analyses/009-2026-08-31-run012-vs-run015-a4-ol1-pressure-sites/figure_data.json`
- A7 verification: `runs/013-2026-08-30-pythia14m-full-pass-a7/artifacts/verification.json`
- A7-OL1 verification: `runs/014-2026-08-31-pythia14m-full-pass-a7-ol1/artifacts/verification.json`
- Figure: `analyses/008-2026-08-31-full-pass-frontier-with-a7/figures/01-full-pass-frontier-with-a7.pdf`
- Table: `analyses/008-2026-08-31-full-pass-frontier-with-a7/tables.md`
- Consolidated finding:
  `research/findings/F002-a7-extends-a4-logical-opportunity.md`
- Finding scope: F002 covers A4/A7 only; no A7/A7-OL1 finding is promoted.

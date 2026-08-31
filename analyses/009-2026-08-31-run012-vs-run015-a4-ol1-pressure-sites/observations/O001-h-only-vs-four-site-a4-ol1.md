# O001 - Historical h-only versus corrected four-site A4-OL1

## Question

At matched A4 gate threshold, how does corrected OL1 pressure on every active
site in `{a,m,h,z}` compare with Run 012's historical realization, which
applied OL1 pressure only at `h`?

## Method and reduction

The comparison contains the five verified Run 012 endpoints and the five
verified Run 015 endpoints at `kappa in {0, 0.01, 0.05, 0.1, 0.5}`. Both
cohorts use one-sided A4-Z gates at `a,m,h,z`, operational `orthogonal_l1` with
`lambda=1` and trust budget 1, the same random initialization, data order,
optimizer schedule, 712-boundary budget, and complete validation workload.
Their paired manifests match exactly for model, data, recipe, seeds, topology,
declared condition, and declared activation-pressure fields.

The declared pressure metadata is not sufficient to identify Run 012. The
analysis audits its frozen training chain and requires the inherited
`ActivationCapture(model, ["h"], ...)` call. It identifies Run 012 as **A4
gates + OL1@h**. For Run 015, it requires all 24
`{a,m,h,z}.layer_{0..5}` capture names on every one of 712 boundaries per
condition and identifies it as **A4 gates + OL1@{a,m,h,z}**. Both objectives
use one aggregate activation loss multiplied once by `lambda=1`; the
four-site case is not four independent lambda-weighted losses.

Final validation loss is taken from each verified cohort summary. `R_model` is
recomputed from the integer zero-product and model-product counts. Every site
fraction below is recomputed as pooled integer exact-zero counts divided by
pooled activation counts. The logical-product diagnostic pass also records a
validation loss; it differs from the terminal cohort value by at most
0.000106 and is used only as a consistency check, not as the plotted loss.

## Coverage

Every endpoint covers all 500 MiniPile validation documents, all 338 complete
2,048-token blocks, and 692,224 input tokens; the 1,444-token incomplete tail
is excluded. Each condition trained for 712 optimizer boundaries and
1,493,172,224 input tokens. The comparison has one seed and one Pythia-14M
scale.

## Figure caption and legend

Final validation loss is plotted against measured `R_model` in percent. Blue
circles show historical Run 012 with A4 gates and realized `h`-only OL1
pressure. Vermilion squares show corrected Run 015 with the same A4 gates and
four-site OL1 pressure. Colored lines connect increasing `kappa` order only;
gray dotted segments join matched `kappa` pairs and are labeled at their
midpoints. Lower validation loss is better. `R_model` is logical exact-zero
product opportunity, not measured runtime speedup.

## Result

Correcting the pressure target from `h` only to all four A4 sites increases
validation loss at every matched threshold. The corrected-minus-h-only loss
deltas are +0.242422, +0.253774, +0.294213, +0.319637, and +0.315321 as
`kappa` increases.

The `R_model` effect changes with threshold. At `kappa=0`, four-site pressure
is 0.5994 percentage points lower than `h`-only pressure. At `kappa=0.01`, the
two are effectively tied (+0.0009 points for four-site pressure). At
`kappa=0.05`, `0.1`, and `0.5`, four-site pressure adds 1.4208, 2.0508, and
2.4861 percentage points, respectively, while paying the validation-loss
costs above. Thus, relative to the historical `h`-only realization, the
corrected intervention does not improve both plotted outcomes at any matched
threshold.

The site table helps locate the redistribution. At `kappa <= 0.1`, `h`-only
pressure produces more exact zeros at `h` than four-site pressure, while the
four-site objective produces more at `a` and `z`. At `m`, four-site pressure
starts below `h` only at `kappa=0` and `0.01`, then exceeds it from
`kappa=0.05` onward. At `kappa=0.5`, both variants nearly saturate `h` and
`z`; corrected four-site pressure additionally reaches 98.9060% at `a`,
96.7655% at `m`, and 0.6610% at the unpressured `v` diagnostic.

## Pooled exact-zero mass

| kappa | Realized pressure | a zero (%) | m zero (%) | h zero (%) | z zero (%) | q_post zero (%) | k_post zero (%) | v zero (%) | attention_output zero (%) |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | OL1@h | 49.413387 | 49.815554 | 93.542890 | 64.670981 | 0.000001 | 0.000002 | 0.000004 | 0.000031 |
| 0 | OL1@{a,m,h,z} | 68.890643 | 39.296132 | 74.204574 | 69.624807 | 0.000001 | 0.000004 | 0.000064 | 0.000329 |
| 0.01 | OL1@h | 49.941895 | 50.389685 | 94.870877 | 71.971289 | 0.000001 | 0.000002 | 0.000006 | 0.000031 |
| 0.01 | OL1@{a,m,h,z} | 73.490866 | 43.463745 | 80.833319 | 85.259458 | 0.000003 | 0.000005 | 0.000065 | 0.000299 |
| 0.05 | OL1@h | 51.790940 | 52.390750 | 97.914634 | 88.307198 | 0.000002 | 0.000002 | 0.000004 | 0.000046 |
| 0.05 | OL1@{a,m,h,z} | 88.504806 | 60.439015 | 93.274620 | 97.391954 | 0.000003 | 0.000011 | 0.000059 | 0.000120 |
| 0.1 | OL1@h | 53.808790 | 54.747458 | 99.124711 | 96.773788 | 0.000005 | 0.000002 | 0.000002 | 0.000126 |
| 0.1 | OL1@{a,m,h,z} | 91.309548 | 75.911749 | 97.149012 | 99.283269 | 0.000003 | 0.000007 | 0.000029 | 0.000024 |
| 0.5 | OL1@h | 63.621824 | 66.441056 | 99.944399 | 99.933706 | 0.000001 | 0.000001 | 0.000001 | 0.000000 |
| 0.5 | OL1@{a,m,h,z} | 98.905956 | 96.765529 | 99.940377 | 99.975715 | 0.000017 | 0.000001 | 0.660960 | 0.000000 |

Percentages are computed from pooled integer exact-zero counts over all six layers and the complete 338-block validation pass; layer or batch fractions are not averaged. All sites except `h` contain 531,628,032 activation elements per row; `h` contains 2,126,512,128 because of the four-times intermediate width. `attention_output` is the post-`W_o` diagnostic and is not an A4 gate or pressure site.

## Caveats

- One seed, one Pythia-14M scale, and one MiniPile pass do not establish
  replication or scale transfer.
- The comparison changes the aggregate pressure target jointly from one site
  to four. It cannot attribute effects separately to `a`, `m`, or `z`.
- Keeping `lambda=1` does not equalize the realized pressure-gradient direction
  or magnitude when the aggregate changes from six `h` tensors to 24
  site/layer tensors.
- Thresholded training changes forward support and pressure-gradient support;
  matching `kappa` does not match realized activation sparsity.
- The runs share their scientific identities but were independently scheduled
  and are not presumed bitwise replicas.
- Exact-zero activation mass, logical opportunity, removed FLOPs, and measured
  sparse-kernel speedup are distinct quantities.

## Provenance

- Source script:
  `analyses/009-2026-08-31-run012-vs-run015-a4-ol1-pressure-sites/01_build.py`
- Machine-readable reduction:
  `analyses/009-2026-08-31-run012-vs-run015-a4-ol1-pressure-sites/figure_data.json`
- Figure:
  `analyses/009-2026-08-31-run012-vs-run015-a4-ol1-pressure-sites/figures/01-rmodel-vs-validation-loss.pdf`
- Historical cohort:
  `runs/012-2026-08-30-pythia14m-full-pass-a4-ol1/artifacts/verification.json`
- Corrected cohort:
  `runs/015-2026-08-31-pythia14m-corrected-a4-ol1/artifacts/verification.json`
- Per-attempt count sources:
  `artifacts/attempts/*/diagnostics/logical_products.json` and
  `artifacts/attempts/*/diagnostics/activation_statistics.json` under each run.

This descriptive analysis does not restore discarded Finding F001, promote a
new finding, or change result-bearing manuscript text.

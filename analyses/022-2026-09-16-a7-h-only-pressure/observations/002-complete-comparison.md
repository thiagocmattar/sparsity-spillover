# Complete matched A7 pressure comparison

Question: does h-only OL1 reproduce the high-threshold logical-opportunity
increase produced by seven-site OL1 under A7 gates?

Method and coverage: `01_compare.py` reads verified Run 013 (no pressure),
Run 032 (h-only), and Run 014 (all seven sites). It checks all five thresholds,
initialization/order hashes, gating topology, pressure assignments, 712 updates
and 1,493,172,224 input tokens per condition. Final-checkpoint evaluation covers
all 500 validation documents, 338 complete 2,048-token blocks, and reports the
excluded 1,444-token tail. No training or evaluation was rerun for this reduction.

Legend/caption: each triple below is no pressure / h-only / all-seven pressure.
Loss is mean next-token cross-entropy in nats. R_model is the percentage of
full-model dense scalar products logically avoidable, not measured runtime.

| Kappa | Validation loss: none / h / all7 | R_model (%): none / h / all7 |
|---:|---:|---:|
| 0 | 5.4684 / 5.2198 / 5.4802 | 7.22 / 8.40 / 7.05 |
| 0.01 | 5.4588 / 5.1981 / 5.4758 | 7.62 / 8.86 / 7.72 |
| 0.05 | 5.4379 / 5.1950 / 5.4628 | 9.13 / 10.13 / 9.86 |
| 0.1 | 5.4287 / 5.2374 / 5.4295 | 10.43 / 10.90 / 11.80 |
| 0.5 | 5.7029 / 5.7320 / 5.8294 | 15.39 / 16.66 / 27.48 |

Result: h-only has lower validation loss than both comparison arms at the four
lower thresholds. Its R_model exceeds no pressure throughout, and exceeds
all-site pressure at kappa=0,0.01,0.05, but is lower at 0.1 and 0.5.

At kappa=0.5, h-only adds 1.276827 percentage points of R_model over no pressure,
versus 12.095871 points for all-site pressure: 10.555893% of the all-site
increment. H-only loss is 0.029126 nats worse than no pressure and 0.097340
nats better than all-site pressure. This endpoint does not reproduce most of
the all-site logical-opportunity increase.

The corresponding pooled exact-zero percentages expose the attention-site
difference despite nearly saturated h zeros:

| Site | No pressure | h-only | All seven |
|---|---:|---:|---:|
| h | 99.8588 | 99.9478 | 99.8764 |
| q_post | 17.4347 | 31.1480 | 93.5450 |
| k_post | 18.1338 | 31.6689 | 94.5413 |
| v | 31.2552 | 25.7569 | 98.7133 |

Caveats: one seed, one scale and one budget; no uncertainty or runtime claim.
The result does not isolate direct Q/K/V necessity, because all-site pressure
also adds a,m,z. Pressure uses the approved mean over six h tensors versus
42 all-site tensors, so target composition and normalization differ. No
manuscript update or promoted finding follows from this descriptive comparison.

Source script: `01_compare.py`. Exact metrics, matched differences, source
paths and SHA-256 values are retained in `artifacts/comparison.json`; machine
and human tables are `artifacts/comparison.csv` and `artifacts/comparison.md`.

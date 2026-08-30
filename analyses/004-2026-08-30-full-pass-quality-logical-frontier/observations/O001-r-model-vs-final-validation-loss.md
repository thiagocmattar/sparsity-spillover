# O001 - Full-pass R_model versus final validation loss

## Question

Where do verified full-pass `A1-H` naive-L1 and OL1 endpoints and `A4-Z`
one-sided-threshold endpoints lie in final validation loss versus measured
`R_model`?

## Sources and coverage

- Analysis 003 `comparison.json`, which count-reconciles Run 004 and Run 009
  endpoint artifacts and includes their source paths and hashes.
- Run 011 `artifacts/verification.json` and every selected attempt's
  `manifest.json`, `metrics.json`, `diagnostics/activation_statistics.json`, and
  `diagnostics/logical_products.json`.
- Fifteen unique endpoints: two reused Run 004 controls, four Run 004 naive-L1
  conditions, four Run 009 OL1 conditions, and five Run 011 A4-Z threshold
  conditions.
- One random initialization and data-order seed, Pythia-14M, 712 optimizer
  boundaries, and 1,493,172,224 training input tokens per endpoint.
- Final validation covers all 338 complete blocks and 692,224 input tokens;
  the 1,444-token tail is excluded.

The initial-parameter and training-schedule hashes match across all source
cohorts. `01_compare.py` rejects unverified evidence or mismatched identities,
steps, training tokens, or validation coverage. For Run 011, it recalculates
activation exact-zero fractions and `R_model` from stored integer counts before
writing the reduced artifact. `02_plot.py` reads only that artifact.

## Figure caption and legend

**Figure 1. Final validation quality versus measured model-level logical
opportunity after one full MiniPile pass.** The single panel shows all 15 unique
endpoints. Blue circles and a solid line show Run 004's A1-H naive-L1 grid;
orange squares and a dashed line show Run 009's matched A1-H OL1 grid; green
diamonds and a dash-dot line labeled A4 show Run 011's joint one-sided-threshold
grid at operational topology A4-Z. The gray plus and black triangle mark the
reused Run 004 GeLU and A1-H ReLU controls. Labels give lambda for pressure
endpoints and kappa for threshold endpoints. Lines connect increasing doses for
readability and do not imply interpolation. Horizontal position is exact-zero
`R_model` as a percentage, a logical-product opportunity rather than measured
speedup. Vertical position is final loss from the reloaded checkpoint on all
338 complete validation blocks; lower is better. The loss axis shows endpoint
detail and does not start at zero. No uncertainty bars are shown because only
one seed was evaluated.

## Observed pattern

Run 011's `R_model` increases monotonically from `7.212019%` at `kappa=0` to
`10.215537%` at `kappa=0.5`. Its validation loss improves from `5.470497` at
`kappa=0` to `5.419642` at `kappa=0.1`, a change of `-0.050855`, while
`R_model` increases by `+1.740986` percentage points. `kappa=0.5` reaches the
highest logical opportunity but worsens loss to `5.659680`, or `+0.189182`
relative to the within-A4-Z `kappa=0` reference. Thus `kappa=0.1` and
`kappa=0.5` form the within-Run-011 nondominated endpoint set; the lower doses
are dominated by `kappa=0.1` on these two recorded coordinates.

The A1-H pressure endpoints remain in the lower-left portion of the figure,
between roughly `3.14%` and `3.95%` `R_model` and `5.10` to `5.21` loss.
The A4-Z threshold endpoints occupy roughly `7.21%` to `10.22%` `R_model` and
`5.42` to `5.66` loss. This cross-series separation is descriptive: topology,
gate placement, and the presence or absence of a pressure objective differ.

## Caveats and nonclaims

This single-seed, single-scale endpoint analysis does not estimate realization
variance. Runs 004/009 and Run 011 are matched in initialization, schedule,
training budget, and validation workload, but not in topology or intervention.
The figure therefore does not establish that thresholding is better or worse
than L1/OL1, identify a causal site contribution, or measure runtime
acceleration. Independently scheduled GPUs are scientifically matched but not
bitwise identical. No observation is promoted to a finding or manuscript claim.

## Provenance

- Reduction: `../01_compare.py`
- Plot: `../02_plot.py`
- Reduced artifact: `../comparison.json`
- Generated tables: `../tables.md`
- Figure: `../figures/01-r-model-vs-final-validation-loss.pdf`

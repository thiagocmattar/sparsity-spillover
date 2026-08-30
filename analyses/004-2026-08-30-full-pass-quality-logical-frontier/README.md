# Analysis 004 - Full-pass quality versus logical opportunity

## Status

Completed as a descriptive comparison of verified Pythia-14M full-pass
endpoints from Runs 004, 009, and 011. No finding was promoted and no
manuscript text was changed.

## Question

Where do `A1-H` naive-L1 and OL1 pressure endpoints and `A4-Z` one-sided
threshold endpoints lie in final validation loss versus measured `R_model`
after one full MiniPile pass?

## Sources and matching

- Analysis 003 supplies ten count-reconciled unique endpoints: Run 004's GeLU
  and ReLU controls and naive-L1 grid, plus Run 009's matched OL1 grid.
- Run 011 supplies five verified `A4-Z` threshold conditions at `kappa` in
  `{0, 0.01, 0.05, 0.1, 0.5}`.
- All conditions share the same initial-parameter hash, training-schedule hash,
  712 optimizer boundaries, 1,493,172,224 training input tokens, and final
  validation over all 338 complete blocks and 692,224 input tokens. The
  1,444-token tail is excluded.

The executing interventions are not matched across all three series. Runs 004
and 009 use `A1-H`, ReLU at `h`, and pressure only at `h`; Run 011 jointly gates
`a,m,h,z` with one fixed one-sided threshold and no pressure objective. The
cross-series comparison is therefore a descriptive quality--logical-opportunity
view, not a method-isolated causal comparison.

`01_compare.py` requires verified source evidence, validates the shared
identities and complete coverage, and recalculates Run 011 activation and
`R_model` fractions from stored integer counts. It writes `comparison.json`
with source paths and SHA-256 values and generates `tables.md`. `02_plot.py`
reads only the reduced artifact.

## Results

Within Run 011, `R_model` rises monotonically from `7.212019%` at `kappa=0` to
`10.215537%` at `kappa=0.5`. Validation loss improves through `kappa=0.1`,
where it reaches `5.419642` with `8.953005%` `R_model`; relative to `kappa=0`,
this is `-0.050855` loss and `+1.740986` percentage points of `R_model`.
`kappa=0.5` increases logical opportunity by `+3.003518` percentage points but
degrades loss by `+0.189182`.

Selected-site exact-zero mass rises strongly with threshold. At `kappa=0.1`,
the pooled exact-zero fractions are `53.244%` at `a`, `55.200%` at `m`,
`91.853%` at `h`, and `89.234%` at `z`. Exact zeros remain negligible at the
untargeted `q_post`, `k_post`, `v`, and post-`W_o` attention-output sites.

The `A1-H` pressure series occupy lower-`R_model`, lower-loss endpoints than
the `A4-Z` series. That separation is consistent with their different topology
and intervention definitions and must not be read as an isolated method effect.
The single-panel figure shows all 15 endpoints and labels every lambda and
kappa value directly.

## Outputs

- `comparison.json` - count-reconciled combined endpoint artifact with source hashes.
- `tables.md` - all figure endpoints and Run 011 loss, `R_model`, and per-site zero-mass tables.
- `figures/01-r-model-vs-final-validation-loss.pdf` - paper-ready full-range and detail figure.
- `observations/O001-r-model-vs-final-validation-loss.md` - caption, provenance, result, and limits.

## Limits

This is one seed and one Pythia-14M scale, with no replicate uncertainty.
Lines connect dose-ordered endpoints for readability and do not imply
interpolation. `R_model` is exact-zero logical-product opportunity rather than
removed FLOPs or measured runtime speedup. Joint A4-Z gating cannot attribute
the response to one site. No finding or manuscript claim is promoted by this
analysis.

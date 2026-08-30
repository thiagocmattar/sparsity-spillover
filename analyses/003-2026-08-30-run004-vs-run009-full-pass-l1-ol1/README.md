# Analysis 003 - Run 004 naive L1 versus Run 009 OL1

## Status

Completed as a descriptive cross-run analysis of two verified full-pass cohorts.
No finding was promoted and no manuscript text was changed.

## Question

At matched activation, topology, pressure site, lambda, initialization, data
order, training budget, and complete-validation coverage, how do operational
OL1 endpoints from Run 009 compare with naive-L1 endpoints from Run 004 in
per-site exact/near-zero mass, measured `R_model`, and final validation loss?

## Source cohort and matching

- Run 004 supplies the GeLU and ReLU controls and four ReLU `A1-H` naive-L1
  conditions at lambda `{0.05, 0.1, 0.5, 1.0}`.
- Run 009 supplies four ReLU `A1-H` OL1 conditions on the same lambda grid and
  reuses, rather than reruns, Run 004's controls.
- Every condition uses the same random initialization seed and parameter hash,
  realized training schedule, 712 optimizer boundaries, 1,493,172,224 training
  input tokens, and final validation over all 338 complete blocks (692,224 input
  tokens). The 1,444-token tail is excluded and reported.
- The executing difference within each lambda pair is `l1_naive` versus
  `orthogonal_l1` with trust budget `1.0`. Independently scheduled Pods are
  scientifically matched but not bitwise identical.

`01_compare.py` validates those identities, recalculates every activation and
logical fraction from stored integer counts, and writes `comparison.json` plus
the generated [tables](tables.md). Source paths and SHA-256 values are retained
in the machine-readable comparison.

## Results

Naive L1 and OL1 produced closely matched `h` exact-zero mass and `R_model` at
each lambda. OL1 minus naive-L1 final validation loss was `-0.008087`,
`-0.006083`, `-0.002453`, and `+0.018753` at lambda `0.05`, `0.1`, `0.5`, and
`1.0`, respectively. The lowest naive-L1 loss was `5.102276` at lambda `1.0`;
the lowest OL1 loss was `5.110235` at lambda `0.5`.

Exact zeros outside `h` were negligible in every condition, so measured
`R_model` was driven almost entirely by `h` zero operands in MLP W2. At the
largest lambda, OL1 had slightly less `h` exact-zero mass and `R_model` than
naive L1 and a higher final loss. These are endpoint descriptions, not a claim
of a method effect under replicate uncertainty.

The figure plots all ten unique endpoints. The four lambda values are written
beside their respective naive-L1 and OL1 points; the two reused controls appear
once. The validation-loss axis uses conventional ascending numerical order, is
explicitly labeled as an endpoint detail, and does not start at zero.

## Outputs

- `comparison.json` - count-reconciled machine-readable endpoint and paired data.
- `tables.md` - exact-zero, near-zero, `R_model`, validation-loss, and matched-delta tables.
- `figures/01-r-model-vs-final-validation-loss.pdf` - quality/logical-opportunity scatter.
- `observations/O001-r-model-vs-final-validation-loss.md` - caption, provenance,
  observed pattern, and limits.

## Limits

This is one seed and one Pythia-14M scale with no replicate uncertainty.
`R_model` is exact-zero logical-product opportunity rather than removed FLOPs
or measured runtime speedup. `attention_output` is post-`W_o` before residual
addition and is not the pre-`W_o` site `z`. No finding or manuscript claim is
promoted by this analysis.

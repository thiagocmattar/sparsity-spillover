# 011: Matched 14M/70M validation loss versus model-wide sparsity

## Question

Provide a quality-sparsity companion to Figure10 using the same A4/A7
OL1(h)/OL1(all) recipes, plus the baseline and 1-site ReLU controls at both
model sizes. The user confirmed validation loss on Y and sparsity on X.

## Method and coverage

The cohort comprises 44 final checkpoints, with 22 matching conditions per
size: A0, A1-H ReLU, and A4/A7 with OL1(h) or OL1(all) at kappa
`{0, 0.01, 0.05, 0.1, 0.5}`. The 42-point cohort after removing the two ReLU
controls must equal Figure10's cohort exactly. Its eight family connections
are retained. No pressure-free A4/A7 or pressured single-site grids are added.

Every y value is ordinary reloaded final-checkpoint validation loss from
`metrics.validation.final.loss`, in nats. The 20 multisite 14M values are
reconciled with Analysis023's `ordinary_final_validation_loss`; the two 14M
controls are read from their Run004 metrics, with control activation and
pressure identity checked. All 22 70M values are reconciled with Analysis025's
`final_validation_loss`. Checkpoint weight paths identify the exact training
attempts used by the timing cohort. The retained weight hashes are recorded;
this plotting task does not reread model weights or reevaluate checkpoints.

This uniform loss convention is deliberate. Analysis023's archival approved
table mixes ordinary and logical-diagnostic pass losses; that table is
preserved. The current figure uses its already documented uniform ordinary
alternative, matching Analysis025. Logical-pass losses are retained separately
in this figure's JSON for audit and are never used as the y coordinate.

X is the unchanged canonical FP16 pooled integer zero-product numerator
divided by the full-model logical-product denominator, times 100. The dense
LM-head contribution remains in the denominator. The builder checks counts
against the original logical diagnostics and the timing-cohort reduction.
Every ordinary loss and logical diagnostic covers all 338 complete 2,048-token
blocks from 500 validation documents: 692,224 input tokens, with the
1,444-token tail excluded. All training endpoints use the retained one-seed
protocol; no new training, evaluation, timing or cloud work is performed.

## Legend and caption

**Validation loss versus model-wide logical sparsity for matched Pythia-14M
and Pythia-70M recipes.** Open markers denote 14M and filled markers denote
70M. Stars mark baseline A0, green circles 1-site A1-H ReLU, blue diamonds
4-sites A4*, and orange triangles 7-sites A7. Short-dashed lines identify
OL1(h), and long-dashed lines identify OL1(all). Each of eight separate curves
connects increasing kappa within a single model size, topology and pressure
recipe. Both axes are linear, and subtle labels identify the model-size groups.
All 44 points use ordinary final-checkpoint loss on complete validation.
Baseline and ReLU controls remain isolated. Lower loss is better.

## Result

The view places the same pressure-recipe families from Figure10 on a quality
axis while restoring the one-site controls. Observed loss ranges from
5.1949586 to 6.0379865 nats at 14M and 4.0997673 to 5.3895430 nats at 70M.
All sparsity coordinates agree with the retained matched latency cohort.
The figure shows the evaluated operating points without fitting a trend or
asserting a monotonic relation between sparsity and validation loss.

## Caveats

This is descriptive one-seed evidence. Shared recipe names do not imply
identical optimization regimes or equal checkpoint quality across model sizes.
Training hardware and execution sessions differ. Model-wide sparsity is a
logical opportunity, and its denominator depends on architecture; this plot
does not normalize by architectural reach or establish runtime acceleration.
It also does not establish a causal effect of sparsity or a scaling law.
Curves connect the fixed threshold grid, not repeated seeds, and no uncertainty
intervals are inferred from those five thresholds. Nearby markers may overlap.

The figure relates to the manuscript's quality-sparsity operating-regime and
pressure-placement discussion. No manuscript text or finding is updated.

## Sources and reproduction

Run `11_plot_matched_quality_sparsity.py` to regenerate
[Figure11](../figures/11-14m-70m-matched-quality-sparsity.pdf) and
[`data/matched-quality-sparsity.json`](../data/matched-quality-sparsity.json).
The JSON records all 44 points, exact checkpoint weight identities, ordinary
and audit-only logical losses, integer counts, 94 source hashes, coverage,
family connections, group labels, linear-axis limits and the PDF hash.
See also [Observation010](010-a0-normalized-speedup.md),
[Analysis023](../../023-2026-09-17-14m-pressure-targets-paper-table/README.md)
and [Analysis025](../../025-2026-09-17-70m-quality-sparsity/README.md).

Verification checks exact cross-size recipe/kappa matching, the Figure10 cohort
plus two controls, raw final losses, complete coverage, source hashes, checkpoint
paths, pooled integer counts, eight curve memberships, linear axes and all-point
axis coverage. The one-page PDF was rendered and visually inspected, with all
fonts embedded. All prior figures and reductions remain unchanged.

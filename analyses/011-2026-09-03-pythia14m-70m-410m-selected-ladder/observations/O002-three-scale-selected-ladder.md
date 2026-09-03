# O002 - Selected-ladder comparison across 14M, 70M, and 410M

## Question

Which features of the selected validation-loss versus `R_model` structure
persist from Pythia-14M and 70M to Pythia-410M?

## Method and coverage

Sources: verified Runs 014, 015, 018, and 019 and the verified A0/A1-H TEAL
artifacts at all three scales. Each scale uses one seed, one MiniPile pass, and
the complete 338-block validation workload. Trained loss and `R_model` are
paired from one eager logical-product pass; integer logical and activation
counts are reconciled before fractions are displayed.

## Figure caption and legend

Pythia-14M, 70M, and 410M validation loss versus measured model-wide logical
zero-product opportunity. Color and shape identify the two post-hoc control
families and two trained families exactly as in Analysis 010. Filled markers
with white edges denote 14M, open markers denote 70M, and filled markers with
dark edges denote 410M. Lines connect target/dose order only. The loss axis is
truncated at 6; off-scale points remain in the reduction and trajectories show
where they leave the visible area.

## Result

Across all three sizes, A7-OL1 dominates A4-OL1 at `kappa=0.5`. The
zero-threshold ordering is not scale-persistent: A4-OL1 dominates A7-OL1 at
14M and 70M, whereas A7-OL1 dominates A4-OL1 at 410M. Larger models also reach
substantially larger absolute `R_model` values under the selected high-dose
ladders, but three one-seed observations do not establish a scaling law.

## Caveats and nonclaims

The comparison is descriptive and has no between-seed uncertainty estimate.
TEAL and trained intervention semantics differ. `R_model` is not measured
runtime improvement. Marker fill encodes scale; color remains reserved for
scientific family. No manuscript finding is promoted.

Source script: `01_build.py`.

Output: `figures/02-pythia14m-70m-410m-selected-ladder.pdf`.

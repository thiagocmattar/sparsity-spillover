# Pythia-14M versus 70M selected ladder

This cross-run analysis asks whether the selected validation-loss versus
measured-`R_model` tradeoff observed at Pythia-14M persists at Pythia-70M.
It reduces the matched A4-OL1 and A7-OL1 five-point trained ladders from Runs
014, 015, and 018, plus the A0 and A1-H ten-point post-hoc TEAL frontiers from
Analysis 005 and Run 018.

Run `01_build.py` from the repository environment to regenerate
`figure_data.json`, `tables.md`, and the unified-frontier publication PDF. The builder
fails closed on source hashes, complete validation coverage, pooled integer
logical counts, exact-zero site counts, training coverage, and OL1 boundary
records. `test_build.py` checks the complete grids, integer reconciliation,
matched crossover, and PDF-only figure contract.

The comparison is descriptive: there is one seed at each scale, and
`R_model` is logical zero-product opportunity rather than measured speedup.
It establishes neither a scaling law nor causal mediation.

## Outputs

- `figure_data.json`: verified machine-readable reduction with source hashes.
- `tables.md`: paper-style complete trained A4-OL1/A7-OL1 and ten-point
  A0/A1-H TEAL tables, including loss, paired TEAL loss deltas, `R_model`, and
  every recorded count-pooled exact-zero site mass. Unmeasured TEAL Q/K/V
  fields are explicitly marked `n.m.`.
- `figures/01-pythia14m-vs-70m-selected-ladder.pdf`: single absolute
  `R_model` versus validation-loss frontier across both model sizes, using the
  Analysis 008 visual grammar and a loss-6 display cap.
- `observations/O001-pythia14m-vs-70m-selected-ladder.md`: interpretation and
  evidence provenance.

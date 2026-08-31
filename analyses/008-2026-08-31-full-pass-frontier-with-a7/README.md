# Analysis 008 - Full-pass frontier with A7 and A7-OL1

This analysis presents the corrected Pythia-14M full-pass frontier. It retains
Analysis 007's unaffected trained series and post-hoc controls, replaces its
historical Run 012 `h`-only series with the corrected four-site Run 015
A4-OL1 endpoints reduced by Analysis 009, and includes the five Run 013 A7 and
five Run 014 A7-OL1 endpoints. Run 012 is intentionally absent from this main
frontier because it did not realize four-site A4-OL1 pressure.

The A7/A7-OL1 table still reports threshold `kappa`, complete-validation loss,
measured `R_model`, and count-pooled exact-zero mass at `a`, `m`, `h`,
`q_post`, `k_post`, `v`, `z`, and post-`W_o` `attention_output`. A7 and A7-OL1
measurements are rebuilt from Runs 013 and 014's verified attempts. Every
imported fraction is checked against pooled integer counts.

## Files

- `01_build.py`: validates sources and generates the figure, data, and table.
- `figure_data.json`: plotted values, integer reductions, hashes, and coverage.
- `tables.md`: the interleaved A7/A7-OL1 results table.
- `figures/01-full-pass-frontier-with-a7.pdf`: corrected single-panel frontier
  with Run 015 A4-OL1, A7, and A7-OL1.
- `observations/O001-full-pass-frontier-with-a7.md`: caption, result, and caveats.
- `observations/INDEX.md`: observation index.
- `test_build.py`: focused source-reduction and output-contract tests.

## Reproduce

From the repository root:

```powershell
.venv\Scripts\python.exe analyses\008-2026-08-31-full-pass-frontier-with-a7\01_build.py
.venv\Scripts\python.exe -m pytest -p no:cacheprovider analyses\008-2026-08-31-full-pass-frontier-with-a7\test_build.py --tb=short
```

This is a one-seed descriptive analysis. `R_model` is logical-product
opportunity, not measured runtime speedup.

## Consolidation

On 2026-08-31, the user approved this matched result as tentative Finding F002:
`research/findings/F002-a7-extends-a4-logical-opportunity.md`. That finding
applies only to the A4/A7 comparison. The added A7/A7-OL1 comparison remains
descriptive and has not been promoted to a finding.

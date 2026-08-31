# Analysis 008 - Full-pass frontier with A7

This analysis extends Analysis 007's verified Pythia-14M full-pass frontier
with the five Run 013 `A7-Z-POST` endpoints. It also reports, in one table,
Run 013 threshold `kappa`, complete-validation loss, measured `R_model`, and
count-pooled exact-zero mass at `a`, `m`, `h`, `q_post`, `k_post`, `v`, `z`,
and post-`W_o` `attention_output`.

The Analysis 007 series and post-hoc control curves are reused from its
machine-readable figure data. A7 measurements are rebuilt from Run 013's
verified attempts. Every fraction is recomputed from pooled integer counts.

## Files

- `01_build.py`: validates sources and generates the figure, data, and table.
- `figure_data.json`: plotted values, integer reductions, hashes, and coverage.
- `tables.md`: the requested single A7 results table.
- `figures/01-full-pass-frontier-with-a7.pdf`: Analysis 007 frontier plus A7.
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
opportunity, not measured runtime speedup. No finding or manuscript claim is
promoted by this analysis alone.

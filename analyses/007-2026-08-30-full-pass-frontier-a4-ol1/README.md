# Analysis 007 - Full-pass frontier with A4-OL1

This analysis augments Analysis 005's full-validation frontier with the five
Run 012 A4-OL1 endpoints. It also reports Run 012 validation loss, measured
`R_model`, and count-pooled exact-zero mass at sites `a`, `m`, `h`, and `z`.

The source Analysis 005 series are reused from its machine-readable figure
data. A4-OL1 measurements are rebuilt from Run 012's verified attempt
artifacts; per-site fractions are computed as pooled integer zero counts divided
by pooled integer activation counts.

## Files

- `01_build.py`: validates source artifacts and generates the figure, data, and table.
- `figure_data.json`: machine-readable plotted values, counts, hashes, and coverage.
- `tables.md`: requested A4-OL1 result table.
- `figures/01-full-pass-frontier-with-a4-ol1.pdf`: augmented frontier figure.
- `observations/O001-full-pass-frontier-with-a4-ol1.md`: interpretation and caveats.
- `observations/INDEX.md`: observation index.
- `test_build.py`: focused source-reduction and output-contract tests.

## Reproduce

From the repository root:

```powershell
.venv\Scripts\python.exe analyses\007-2026-08-30-full-pass-frontier-a4-ol1\01_build.py
.venv\Scripts\python.exe -m pytest -p no:cacheprovider analyses\007-2026-08-30-full-pass-frontier-a4-ol1\test_build.py --tb=short
```

This is a one-seed descriptive comparison. `R_model` is a logical-product
opportunity, not a measured runtime speedup.

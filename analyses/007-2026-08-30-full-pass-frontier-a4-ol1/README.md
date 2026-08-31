# Analysis 007 - Full-pass frontier with A4-OL1

Correction (2026-08-31): Run 012 captured OL1 pressure only at `h` despite
declaring `a,m,h,z`. This analysis remains reproducible for its source numbers,
but its Run 012 series must be interpreted as **A4-Z + OL1@h**, not four-site
A4-OL1. It does not support Finding F001, which is discarded. Corrected
four-site evidence is designed in Run 015 and has not been launched.

This analysis augments Analysis 005's full-validation frontier with the five
historical Run 012 A4-Z + OL1@h endpoints. It also reports Run 012 validation loss, measured
`R_model`, and count-pooled exact-zero mass at sites `a`, `m`, `h`, and `z`.

The source Analysis 005 series are reused from its machine-readable figure
data. Historical Run 012 measurements are rebuilt from its verified attempt
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

## Consolidation

On 2026-08-30, the user approved this matched result as tentative Finding F001.
The 2026-08-31 implementation audit invalidated the four-site premise, so F001
is now discarded: `research/findings/F001-a4-ol1-improves-moderate-threshold-frontier.md`.

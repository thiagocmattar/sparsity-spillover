# Analysis 009 - Historical h-only versus corrected four-site A4-OL1

This analysis directly compares the five verified Run 012 endpoints with the
five corrected Run 015 endpoints at matched `kappa`. Both use A4-Z one-sided
gates at `a,m,h,z`, OL1 with `lambda=1` and trust budget 1, the same random
initialization, data order, optimizer schedule, training budget, and complete
validation workload. The scientifically different field is the realized
pressure capture: Run 012 inherited `h` only, whereas Run 015 verifies all 24
`{a,m,h,z}.layer_{0..5}` tensors at every optimizer boundary.

Run 012's stale declared four-site pressure metadata is retained as historical
provenance but is not used as its series identity. The reduction checks the
inherited `ActivationCapture(model, ["h"], ...)` implementation directly and
checks Run 015's boundary-complete capture count and name hash.

## Files

- `01_build.py`: validates both source cohorts, pools integer counts, writes the
  machine-readable reduction, and generates the figure.
- `figure_data.json`: plotted values, pooled integer counts, matched deltas,
  coverage, hashes, and realization audit.
- `figures/01-rmodel-vs-validation-loss.pdf`: direct two-series comparison.
- `observations/O001-h-only-vs-four-site-a4-ol1.md`: question, method, figure
  caption, result, caveats, provenance, and the all-site exact-zero table.
- `observations/INDEX.md`: observation index.
- `test_build.py`: focused realization, matching, count, table, and output tests.

## Reproduce

From the repository root:

```powershell
.venv\Scripts\python.exe analyses\009-2026-08-31-run012-vs-run015-a4-ol1-pressure-sites\01_build.py
.venv\Scripts\python.exe -m pytest -p no:cacheprovider analyses\009-2026-08-31-run012-vs-run015-a4-ol1-pressure-sites\test_build.py --tb=short
```

This is a one-seed, one-model-scale descriptive comparison. `R_model` is a
logical-product opportunity, not measured runtime speedup. No finding or
manuscript claim is promoted by creating this analysis.

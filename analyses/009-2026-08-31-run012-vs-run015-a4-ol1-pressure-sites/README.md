# Analysis 009 - Historical h-only versus corrected four-site A4-OL1

This analysis plots the five verified Run 011 A4 endpoints together with the
five verified Run 012 endpoints and five corrected Run 015 endpoints at matched
`kappa`. All use A4-Z one-sided gates at `a,m,h,z`, the same random
initialization, data order, optimizer schedule, training budget, and complete
validation workload. Run 011 has no pressure. The pressure variants use OL1
with `lambda=1` and trust budget 1; Run 012 inherited `h` only, whereas Run 015
verifies all 24 `{a,m,h,z}.layer_{0..5}` tensors at every optimizer boundary.

Run 012's stale declared four-site pressure metadata is retained as historical
provenance but is not used as its series identity. The reduction checks the
inherited `ActivationCapture(model, ["h"], ...)` implementation directly and
checks Run 015's boundary-complete capture count and name hash.

## Files

- `01_build.py`: validates both source cohorts, pools integer counts, writes the
  machine-readable reduction, and generates the figure.
- `figure_data.json`: plotted values, pooled integer counts, matched deltas,
  coverage, hashes, and realization audit.
- `figures/01-rmodel-vs-validation-loss.pdf`: A4 baseline plus the two realized
  OL1 pressure trajectories.
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

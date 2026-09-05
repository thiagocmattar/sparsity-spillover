# Analysis 014 — matched 70M/410M sparse-kernel sentinels

This analysis compares the six verified Run-023 and Run-024 sentinels on the
same physical H100 NVL. It is a descriptive systems calibration, not a model
scaling law: the two model sizes have different losses and activation
distributions.

- `condition-comparison.csv`: condition-level quality, logical opportunity,
  adapter control, full-model speedup, and kernel summaries.
- `matched-speedup-change.csv`: direct 70M→410M full-model changes.
- `operation-comparison.csv`: per-condition/per-operation medians across layers.
- `attention-comparison.csv`: separate A7 composition medians.
- `comparison-summary.json`: device identity, positive controls, and break-even
  counts.
- `observations/001-pythia70m-vs-410m-sparse-kernel-sentinels.md`: interpreted
  result and caveats.

Regenerate from the verified run reductions with:

```powershell
python .\01_compare.py
```


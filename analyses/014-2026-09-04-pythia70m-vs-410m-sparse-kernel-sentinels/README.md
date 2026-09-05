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
- `figures/01-rmodel-vs-speedup.pdf`: all 24 valid full-model measurements
  against canonical `R_model`, coloured by model size, with matched-scale
  panels for batch sizes 1 and 32.
- `rmodel-speedup-points.csv`: plotted coordinates, integer logical-product
  counts, all seven paired timing ratios per point, and raw-source hashes.
- `observations/002-rmodel-vs-speedup.md`: figure caption, coverage, exclusions,
  and interpretation limits.

Regenerate from the verified run reductions with:

```powershell
python .\01_compare.py
python .\02_plot_rmodel_speedup.py
```

The figure generator checks all 12 unique sentinels, both batches, source
verification status, complete canonical coverage, count-derived `R_model`,
and agreement between raw paired timing medians and the comparison table.
The PDF was rendered and visually checked. No additional GPU work was needed.

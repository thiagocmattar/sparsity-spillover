# Analysis 012 - Paper-facing synthesis through Pythia-410M

## Question

What is the clearest evidence-preserving presentation of the selected
Pythia-14M, 70M, and 410M activation-sparsity results for a submission draft?
In particular:

1. which validation-loss--`R_model` relationships persist across sizes;
2. which relationships reverse at 410M under the shared one-pass token budget;
3. how do exact-zero patterns differ by site; and
4. which transformer operations contribute the observed logical-product
   opportunity?

This is a re-analysis of completed artifacts. It introduces no training run,
new checkpoint evaluation, or promoted finding.

## Sources and method

`01_build.py` consumes the verified machine-readable reduction from Analysis
011 and, for the four trained endpoints at `kappa` 0 and 0.5, the source
activation and logical-product diagnostics named and hashed by that reduction.
It verifies source hashes, complete 338-block validation coverage, complete
condition grids, count reconciliation, and the Run 021 learning-rate decision.

Every plotted loss is paired with `R_model` from the same eager full-validation
pass. Exact-zero values pool integer counts before division. Operation
contributions divide each operation's zero-product count by the complete model
denominator, so stacked bars sum to `R_model`. Lines order tested doses or
post-hoc target sparsities; they are not fitted curves.

## Outputs

- `figure_data.json`: paper-facing reduction, source hashes, baseline exposure,
  within-family deltas, sitewise zeros, and operation contributions.
- `tables.md` and `tables/*.tex`: compact baseline and endpoint tables.
- `figures/01-absolute-frontiers-by-scale.pdf`: three aligned absolute
  validation-loss--`R_model` panels.
- `figures/02-within-scale-deltas.pdf`: trained and post-hoc within-family
  changes relative to `kappa=0` or `p=0`.
- `figures/03-sitewise-zero-structure.pdf`: exact-zero mass by site for the
  selected low/high-dose trained endpoints.
- `figures/04-operation-contributions.pdf`: measured `R_model` decomposed by
  counted transformer operation.
- `observations/`: figure-specific interpretation and limitations.

## Interpretation boundary

The 410M cohort is used as a fixed-token boundary condition. Equal total tokens
imply sharply unequal tokens per parameter, and the higher-LR screen does not
repair the 410M A0 endpoint. That supports reporting a regime dependence; it
does not identify undertraining as the cause. The A4-OL1/A7-OL1 comparison is
between complete recipes that change both gate topology and pressure sites.
Only the 14M no-pressure A4/A7 comparison isolates topology expansion.

`R_model` is exact logical zero-product opportunity, not removed FLOPs or
measured speedup. All empirical comparisons have one seed per scale.

## Reproduce

```powershell
.venv\Scripts\python.exe analyses\012-2026-09-04-paper-synthesis\01_build.py
.venv\Scripts\python.exe -m pytest -p no:cacheprovider analyses\012-2026-09-04-paper-synthesis\test_build.py --tb=short
```

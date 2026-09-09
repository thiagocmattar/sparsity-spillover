# Analysis 019: existing-record audit for the manuscript rewrite

Authorized by `manuscript/draft/task.md` in CURRENT_RESULTS_ONLY mode. This
analysis derives quality-budget, runtime-attribution, historical-control and
diagnostic summaries from retained measurements. It does not train, run
inference, launch compute, or alter original run records. Manuscript claims
remain bounded by one training seed, the tested recipes and the stated workloads.

## Scope and outputs

- `training-audit.json`: all 54 endpoints, 29 paired contrasts, 15 scale pairs,
  operation sums, common A7 normalization, retrospective quality budgets over
  594 evaluated points, and clipping dominance with a p=0 drift sensitivity.
- `runtime-audit.json`: 30 manuscript checkpoints; reconciliation of 201,600
  paired timings across five candidates, absolute host latencies, qualifications,
  sparse-path factors, regression sensitivity and retained BF16 row-NNZ counts.
  It verifies 3,542 source artifacts. Timings and canonical FP16 product counts
  remain distinct measurements; BF16 scalar counts are only a lower bound.
- `historical-audit.json`: five historical Run 012 h-only-pressure controls,
  separately compared with corrected Run 015. They remain outside the main
  30-condition cohort; existing per-term pressure coefficients also differ.
- `tables/`: twelve TeX fragments and their source hashes.
- `figures/`: five revised data figures and two recipe PDFs. Source hashes are
  in `figures/SOURCES.json` and `figure-source/SOURCES.json`.
- `figure-source/`: revised recipe/composite TeX and the unchanged architecture
  panel needed to compile it without the ignored historical artifacts folder.
- [observations/INDEX.md](observations/INDEX.md): question, method, coverage,
  results, figure captions and limitations for each output.

## Regenerate displays from retained records

From the repository root, use the existing project Python environment (NumPy,
Matplotlib, PyYAML and PyTorch; the latter two enter through Analysis 018 imports).
LaTeX with TikZ is needed for the recipe figure. These commands use the checked-in
numerical records and require no model weights, token caches or GPU inference:

```powershell
.venv/Scripts/python.exe analyses/019-2026-09-09-manuscript-rewrite-audit/04_make_tables.py
.venv/Scripts/python.exe analyses/019-2026-09-09-manuscript-rewrite-audit/05_make_figures.py
.venv/Scripts/python.exe analyses/019-2026-09-09-manuscript-rewrite-audit/06_make_recipe_figure.py
```

Keep the repository layout: the generators import Analysis 018's plot/binning
functions and read its frozen coordinates/table fragments. Density regeneration
reads `manuscript/draft/supplementary-data/histograms/`. Script 06 writes build
intermediates under ignored `tmp/rewrite-recipe-build/`. It compiles the retained
source; original site placement and twelve analytic values are preserved.
The source PDFs/tables are copied into the draft as recorded by its SOURCES files.
Rebuilding a PDF may change LaTeX metadata without changing text or pixels.

## Re-audit original measurements

These read-only reductions additionally require the original run/analysis
artifacts at their recorded paths, including ignored local evidence. A fresh
Git checkout without those records cannot perform these raw-source audits.
They do not call model evaluation or original reducers' executable entry points:

```powershell
.venv/Scripts/python.exe analyses/019-2026-09-09-manuscript-rewrite-audit/01_audit_training.py
.venv/Scripts/python.exe analyses/019-2026-09-09-manuscript-rewrite-audit/02_audit_runtime.py
.venv/Scripts/python.exe analyses/019-2026-09-09-manuscript-rewrite-audit/03_audit_history.py
```

Audits retain exact values and hashes; percentages and rounded prose are display
conventions. Original source observations remain in Analysis 018 and Runs 029?031.
The manuscript uses architectural reach `R_arch`; raw `R_model`, `R_block`,
`R_model_max*` and historical `U_arch` keys are unchanged.

## Verification and manuscript tracking

All three source audits passed. Scientific checks covered site placement,
pressure mathematics, counters and reach. Display regeneration was repeated in
a separate directory containing retained inputs but no original-run artifacts or
checkpoints, using the installed environment. This verifies a clean directory,
not a newly provisioned Python environment or a reproduced training run.

The [draft revision record](../../manuscript/draft/revision/README.md) contains
task state, claim ledger, exact numerical checks, final verification and the
readiness assessment. Optional experiments remain proposed and unlaunched.

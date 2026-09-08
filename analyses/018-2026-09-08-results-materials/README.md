# Analysis 018 — Results materials for the revised argument

Revised after the user's figure review on 8 September 2026. This package reads
the current manuscript and reprocesses retained experiments. It contains eight
publication PDFs, numerical tables, proposed arguments, captions and provenance.
All artwork is drawn at the ICLR template's 5.5-inch text width and now includes a title.

Figure 01 now shows all 30 trained endpoints with 150 retained post-hoc
clipping evaluations as faint trajectories. Pressure curves are dashed; its
smaller legend sits below the plot. No training point is filtered. The activation
distribution grid is restored. **Finer activation bins remain pending:** the stored counts
only define four bins. The middle κ=.05 row is included now; the exact proposal
for new counts is [ACTIVATION-DIAGNOSTIC.md](ACTIVATION-DIAGNOSTIC.md).

Start with [RESULTS.md](RESULTS.md) for the argument and
[FIGURE-REVIEW.md](FIGURE-REVIEW.md) for the critical review of every figure.
The [checklist](CHECKLIST.md) records completion; [verification](VERIFICATION.md)
records the checks actually performed. Original runs and manuscript files are unchanged.

## Suggested paper order

| Placement | Figure | Purpose |
| --- | --- | --- |
| Main: overview | [01 — 14M recipes](figures/01-14m-overview.pdf) | 30 trained endpoints with faint post-hoc trajectories for all 15 retained source checkpoints |
| Main: matched effects | [02 — intervention effects](figures/02-blocked-intervention-effects.pdf) | Aligned loss and sparsity changes for 25 matched comparisons |
| Main: structural context | [08 — ceiling versus size](figures/08-ceiling-vs-model-size.pdf) | The architecture ceiling at each model size and topology |
| Main: transfer | [03 — scale transfer](figures/03-scale-transfer-and-ceilings.pdf) | Absolute loss/raw sparsity and theoretical ceiling lines above; relative loss/ceiling utilization below |
| Main case study, or appendix if space is tight | [05 — activation mass](figures/05-activation-mass-grid.pdf) | Site columns h,m,q,k,v and κ=0,.05,.5 rows; retained four-bin distributions, pending finer measurements |
| Supporting appendix | [04 — operation accounting](figures/04-operation-accounting.pdf) | Six solid-color bars decomposing high-threshold results into operations |
| Supporting appendix | [06 — post-hoc clipping](figures/06-complete-posthoc-comparison.pdf) | Explicit 14M coverage: 15 source checkpoints, each at 10 clipping targets |
| Systems subsection | [07 — kernel realization](figures/07-kernel-realization.pdf) | Qualified search progress and the final kernel on the same 30 included checkpoints |

Numbering preserves existing filenames; Figure 08 belongs before Figure 03 in
the argument. It need not become paper Figure 8. The observations contain
self-contained captions and evidence links: [observation index](observations/INDEX.md).

## Scope and tables

A4-OL1[h] is excluded from all current numerical results and artwork, including
runtime summaries. The corrected four-site A4-OL1 remains. There are 54 trained
conditions (30/12/12 at 14M/70M/410M) and 190 clipping evaluations (150/20/20).
These 244 evaluated conditions are not independent random seeds.

- Compact table candidates: [pressure and gate effects](tables/pressure-and-gate-effects.md)
  and [scale endpoints](tables/scale-endpoints.md), both also available as TeX.
- Complete numerical results: [trained endpoints](tables/all-trained-endpoints.md),
  [clipping evaluations](tables/all-clipping-points.md), [matched effects](tables/blocked-effects.md),
  [cross-scale pairs](tables/scale-paired-recipes.md), [numerical frontier membership](tables/frontiers.md), [complete overview series](tables/overview-series.md).
- Supporting evidence: [ceilings versus size](tables/ceiling-vs-model-size.md),
  [architecture counts](tables/architecture-counts.md), [normalization audit](tables/normalization-audit.md),
  [operation counts](tables/operation-counts.md), [activation statistics](tables/activation-statistics.md), [case-study magnitude bins](tables/activation-case-mass-bins.md),
  [training protocol](tables/training-protocol.md), [runtime summary](tables/runtime-summary.md).

[results.tex](results.tex) is an optional analysis-owned insertion fragment.
It does not modify the draft. The revised runtime cohort requires replacing
the old 35-checkpoint summary if this material is adopted in the manuscript.

## Reproduce

```powershell
.venv/Scripts/python.exe analyses/018-2026-09-08-results-materials/01_build.py
.venv/Scripts/python.exe -m pytest -p no:cacheprovider analyses/018-2026-09-08-results-materials/test_evidence.py tests/test_ceilings.py tests/test_metrics.py -q
```

`evidence.py` reconciles retained artifacts; `plots.py` draws only this analysis's
figures; `01_build.py` writes tables, PDFs and data. `figure_data.json` retains
integer counts, endpoint identities, normalization definitions, runtime
replicates and 230 directly hashed source files. `artifact_inventory.json`
identifies every generated PDF/table. No training, inference or cloud work was run.

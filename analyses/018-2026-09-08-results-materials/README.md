# Analysis 018 — Results materials for the revised argument

Figure 05-v3 is complete: [signed activation distributions](figures/05-v3-activation-density-grid.pdf)
now overlay FFN and attention densities from seven full-validation Run 031
measurements, with dashed trained thresholds. Exact-zero mass is retained in
the accompanying data and observation table.
See [O011](observations/O011-activation-density-v3.md) for the caption, pooling
and tail coverage, and [the data release](../../runs/031-2026-09-08-signed-activation-density/results/README.md).
This adds a third alternative PDF (eleven PDFs total). The original Figure 05
and Figure 05-v2 are preserved; older pending-bin notes below concern the original
coarse magnitude plot. Generate v3 with `02_activation_density_v3.py`.

The user selected the original horizontal [Figure 02](figures/02-blocked-intervention-effects.pdf).
Its final title is **Paired intervention effects**. A reusable paper caption is
in [CAPTIONS.md](CAPTIONS.md#figure-02-paired-intervention-effects).
Its metric axes show delta loss and delta model sparsity (pp). Beneath the
plots, two rows have horizontal labels on the left: **Intervention:** for the
action/site descriptions and **Paired:** for treatment-minus-reference names
such as A1-H − A0 and A4-OL1 − A4. All 29 paired differences are retained; the
[O002 reference map](observations/O002-blocked-effects.md#reference-map) defines
each comparison. The [rotated Figure 02-v2](figures/02-v2-blocked-intervention-effects.pdf)
is retained as an unselected layout alternative, documented in
[O009](observations/O009-blocked-effects-v2.md).

Revised after the user's figure review on 8 September 2026. This package reads
the current manuscript and reprocesses retained experiments. It contains eight
main publication figures plus alternative Figures 02-v2 and 05-v2, numerical
tables, proposed arguments, captions and provenance.
All artwork is drawn at the ICLR template's 5.5-inch text width and now includes a title.

Figure 01 shows 16 trained endpoints from A0/A1-H and the three OL1 families,
with post-hoc clipping drawn only for A0 as a clearly visible dashed path.
The title is "Quality vs. model-wide sparsity frontier for train-time and post-hoc interventions (Pythia-14M)", wrapped over two lines. Its compact legend remains below the plot.
The complete 30-checkpoint/300-point 14M measurements remain available. The activation
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
| Main: overview | [01 — 14M recipes](figures/01-14m-overview.pdf) | 16 trained endpoints: A0/A1-H and OL1 variants, with a visible dashed A0 clipping path |
| Main: matched effects | [02 — intervention effects](figures/02-blocked-intervention-effects.pdf) | Aligned loss and sparsity changes for 29 matched comparisons, with all five A1-H→A4 thresholds |
| Retained alternative | [02-v2 — intervention rows](figures/02-v2-blocked-intervention-effects.pdf) | Unselected vertical layout of the same 29 comparisons |
| Main: structural context | [08 — ceiling versus size](figures/08-ceiling-vs-model-size.pdf) | The architecture ceiling at each model size and topology |
| Main: transfer | [03 — scale transfer](figures/03-scale-transfer-and-ceilings.pdf) | A4-OL1/A7-OL1 and A0 clipping; raw sparsity with rounded 30/50/90% axis limits above, common A7-reference utilization below |
| Main case study, or appendix if space is tight | [05 — activation mass](figures/05-activation-mass-grid.pdf) | Site columns h,m,q,k,v and κ=0,.05,.5 rows; retained four-bin distributions, pending finer measurements |
| Alternative for review | [05-v2 — zeros and small activations](figures/05-v2-activation-mass-grid.pdf) | Six panels: exact zeros above, small nonzero mass below; threshold columns and linear 0–100% scales. See [O010](observations/O010-activation-mass-v2.md). |
| Supporting appendix | [04 — operation accounting](figures/04-operation-accounting.pdf) | Six solid-color bars decomposing high-threshold results into operations |
| Supporting appendix | [06 — post-hoc clipping](figures/06-complete-posthoc-comparison.pdf) | Explicit 14M coverage: 15 source checkpoints, each at 10 clipping targets |
| Systems subsection | [07 — kernel realization](figures/07-kernel-realization.pdf) | Qualified search progress and the final kernel on the same 30 included checkpoints |

Numbering preserves existing filenames; Figure 08 belongs before Figure 03 in
the argument. It need not become paper Figure 8. The observations contain
self-contained captions and evidence links: [observation index](observations/INDEX.md).

## Scope and tables

A4-OL1[h] is excluded from all current numerical results and artwork, including
runtime summaries. The corrected four-site A4-OL1 remains. There are 54 trained
conditions (30/12/12 at 14M/70M/410M). [Run 030](../../runs/030-2026-09-08-all-models-posthoc-clipping/README.md)
provides all 540 clipping evaluations (300/120/120), with A0's ten clipping evaluations used in
Figure 01. The other seven figures and legacy clipping table retain their
previous 190-point subset (150/20/20). These are repeated evaluations of fixed
checkpoints, not independent seeds. Full measurements, CSV, and numerical
frontier memberships are available in Run 030's results folder.

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
replicates and directly hashed source files, including Run 030's complete measurement release. `artifact_inventory.json`
identifies every generated PDF/table. This analysis only plots retained evidence; Run 030 performed the additional
user-authorized checkpoint evaluations.

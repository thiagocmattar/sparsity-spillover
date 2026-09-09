# Analysis 018 — Results materials for the revised argument

The current 26-page manuscript uses the all-variant overview v2 and places the
architecture/recipe diagram in the main experimental setup. See the
[manuscript record](../../manuscript/README.md). The snapshots below preserve
the earlier revisions; the 9 September closeout checks are in [VERIFICATION.md](VERIFICATION.md).

The 8 September manuscript revision added the fixed-threshold motivation and kernel
specialization argument. [O015](observations/O015-threshold-kernel-argument.md)
records primary-source checks, the corrected historical-replay coverage,
unchanged figures/data and the verified 27-page
[source/PDF snapshot](provenance/manuscript-20260908-threshold-kernel/README.md).

The manuscript has now received paragraph-level technical-reader, scientific
and literature review. [O014](observations/O014-manuscript-polish.md) records the
stronger argument, standardized terminology, concise captions, unchanged
figures/numbers and verified 26-page [source/PDF snapshot](provenance/manuscript-20260908-polished/README.md).
The initial results snapshot below remains a historical record.

Figure 01 has a separate [v2 with all eight training variants](figures/01-v2-14m-overview.pdf):
30 trained checkpoints, adding A1-H-L1, A4 and A7 while retaining A0-only
post-hoc clipping. The original figure is retained; the current manuscript uses v2.
Generate it with `04_overview_v2.py`; [O013](observations/O013-overview-all-variants.md)
records its caption and verification.

Figure 05-v3 is complete: [signed activation distributions](figures/05-v3-activation-density-grid.pdf)
now overlay FFN and attention densities from seven full-validation Run 031
measurements, with dashed trained thresholds. Exact-zero mass is retained in
the accompanying data and observation table.
See [O011](observations/O011-activation-density-v3.md) for the caption, pooling
and tail coverage, and [the data release](../../runs/031-2026-09-08-signed-activation-density/results/README.md).
This was the third alternative PDF; Figure 01 v2 brings the current total to twelve PDFs. The original Figure 05
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

Latest revision follows the user's request for small readability/label fixes.
The existing eight layouts are retained. Figure 06 connects each checkpoint's
clipping trajectory, and Figure 02 now includes all five A1-H→A4 thresholds.
The broader redesign was rolled back.

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
records the checks actually performed. Original runs are unchanged; the manuscript adoption is recorded in O012.

## Selected paper figures and supporting material

The user selected five results figures. The implemented manuscript order is
overview, paired effects, distribution reshaping, transfer across sizes, then
kernel realization on the corrected 30-checkpoint cohort.
[MANUSCRIPT-PLAN.md](MANUSCRIPT-PLAN.md) specifies each goal, draft location,
main-table recommendation and appendix coverage; [CAPTIONS.md](CAPTIONS.md)
contains the five source captions, adapted with resolved references in the draft. The working plan is in
[the manuscript draft](../../manuscript/draft/results-plan.md).

| Placement | Figure | Purpose |
| --- | --- | --- |
| Main: overview | [01-v2 - 14M recipes](figures/01-v2-14m-overview.pdf) | 30 trained endpoints and A0 post-hoc clipping |
| Main: paired effects | [02 - intervention effects](figures/02-blocked-intervention-effects.pdf) | 29 matched changes in loss and model-wide sparsity |
| Main: distributions | [05-v3 - signed activation densities](figures/05-v3-activation-density-grid.pdf) | FFN/attention reshaping under A0, A4-OL1 and A7-OL1 |
| Main: transfer | [03 - scale transfer](figures/03-scale-transfer-and-ceilings.pdf) | Test persistence of the high-threshold recipe ordering with a common A7 reference |
| Main: kernel realization | [07 - kernel realization](figures/07-kernel-realization.pdf) | Qualified search progress and measured acceleration across the corrected 30-checkpoint cohort; replaces the older kernel figure and its 35-checkpoint prose |
| Appendix support | [04 - operation accounting](figures/04-operation-accounting.pdf), optionally [08 - ceiling versus size](figures/08-ceiling-vs-model-size.pdf) | Interpret operation contributions and architectural reach |
| Complete clipping appendix | [Run 030 full-range plots and observations](../../runs/030-2026-09-08-all-models-posthoc-clipping/observations/INDEX.md) | All 54 checkpoints and 540 clipping evaluations |
| Retained alternatives/subsets | [02-v2](figures/02-v2-blocked-intervention-effects.pdf), [05 original](figures/05-activation-mass-grid.pdf), [05-v2](figures/05-v2-activation-mass-grid.pdf), [06](figures/06-complete-posthoc-comparison.pdf) | Available records, not additional selected main-text figures; Figure 06 contains 150 clipping points |

File numbering is unchanged and does not prescribe paper numbering. The
[observation index](observations/INDEX.md) retains evidence and source links.
Results-section insertion is complete. [O012](observations/O012-manuscript-results.md)
records the argument-led prose, seven generated manuscript tables, complete
appendix and local evidence release. The [tracked source/PDF snapshot](provenance/manuscript-20260908-results/README.md)
preserves the 24-page reading copy while `manuscript/draft/` remains ignored.
No figure artwork or numerical measurement changed during manuscript assembly.

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
The adopted, hand-written prose is in the manuscript snapshot; it supersedes
this optional fragment for the current draft. The draft now uses the corrected
30-checkpoint runtime summary. Generate its seven retained-evidence tables with
`.venv/Scripts/python.exe analyses/018-2026-09-08-results-materials/03_manuscript_tables.py`.

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

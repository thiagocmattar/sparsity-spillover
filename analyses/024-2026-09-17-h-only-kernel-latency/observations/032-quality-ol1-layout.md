# Quality overview: OL1 pressure, independent loss scales and three-row legend

## Question and method

The author requested removal of the naive-L1 results from the main 14M/70M
quality figure, a taller/narrower layout, independent Y scales, and finally
a return to the three-row legend. This revises
[14-14m-70m-quality-sparsity.pdf](../figures/14-14m-70m-quality-sparsity.pdf)
and its byte-identical manuscript copy. Source:
[23_plot_quality_main_appendix.py](../23_plot_quality_main_appendix.py).
Exact plotted records, exclusions, limits and hashes:
[quality-main-appendix.json](../data/quality-main-appendix.json).

The figure now contains 36 trained conditions at 14M and 22 at 70M, with
all twenty Base/ReLU clipping settings per size. Four historical 14M
naive-L1 points are omitted from this view and explicitly preserved under
`excluded_trained_points` in the export. The T1/P1 legend label drops the
OL1 qualifier; all displayed pressure recipes use OL1. Controls and
unpressured threshold recipes remain. All retained values, colors and
line styles are unchanged. Original all-model data and the complete 410M
appendix PDF remain unchanged.

The PDF measures 9.72 by 5.20 inches, versus 10.80 by 4.55 previously
(10% narrower and 14.3% taller). Its three-column, three-row legend groups
the controls/one-site recipes, four-site recipes and seven-site recipes.
The manuscript includes it at 90% of line width. Loss limits are [5.0, 6.2]
at 14M and [4.0, 5.5] at 70M. Every retained trained point is visible;
eight/seven high-loss clipping points continue above the views. Complete
clipping trajectories remain in the appendix.

## Caption

**Quality-sparsity trade-offs at 14M and 70M.** 36/22 trained conditions,
with loss scales fitted separately to each size. Colors distinguish executed
recipes; lines connect separate threshold settings (kappa = 0, .01, .05,
.1, .5), or pressure weights (lambda = .05, .1, .5, 1) for T1/P1. Dotted
paths apply post-hoc clipping at a, m, h, z to the fixed base and ReLU
controls. Vertical guides mark the T4/T7 analytic sparsity ceilings. All
trained points are visible; 8/7 high-loss clipping evaluations continue
above the view (complete paths in the appendix). Trained losses use ordinary
final-checkpoint evaluation; sparsity uses pooled FP16 logical counts over
all 338 complete validation blocks.

## Associated manuscript writing and caveats

`manuscript/draft/training-results.tex` updates the caption, include width
and the overview's condition count from 62 to 58. The appendix's evidence
description now records the four omitted naive-L1 conditions explicitly.
The empirical argument is unchanged: interventions approach sparsity
ceilings at a quality cost. Different Y scales improve within-size reading;
slopes and vertical distances must be interpreted using their axis labels.
The study's full 74-condition inventory remains available in the source data.

Coverage is unchanged: all 338 complete 2,048-token blocks from 500 validation
documents, 692,224 input tokens, with the 1,444-token tail excluded. Loss-pass,
single-seed and logical-versus-runtime caveats from
[observation 031](031-quality-main-appendix.md) still apply.

## Verification

ID-based record equality verifies that the 70 plotted records across main
and appendix views plus four exclusions recover all 74 original records.
All sixty control-clipping records are unchanged. Source/output hashes and
manuscript copies reconcile. Every trained point fits its declared panel
limits. The final PDF was rendered and checked for legend, annotation and
label spacing. Other analysis PDFs and ongoing author edits are unchanged.
The final manuscript-size view was checked in a temporary 34-page build,
with no overfull boxes. The pre-existing unresolved `\ref{tab}` in the
author's opening setup paragraph remains outside this edit.
The canonical `main.pdf` was not replaced during this figure-only edit.

Rebuild with:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/024-2026-09-17-h-only-kernel-latency/23_plot_quality_main_appendix.py
```

# Figure 3: Pressure effects relative to matched no-pressure controls

PDF: [03-pressure-scope-threshold.pdf](../figures/03-pressure-scope-threshold.pdf).
Manuscript placement: [Section 4.2](../../../manuscript/draft/training-results.tex),
`sec:paired-results` (proposed replacement for the pressure-scope portion of
`fig:paired-intervention-effects`).

## Question, method, and coverage

How does adding h-only or all-site OL1 pressure change loss and model-wide
sparsity relative to no pressure at the same threshold topology and κ?
Twenty exact pairs use 30 unique 14M checkpoints: two threshold scopes,
two pressure treatments, and five κ values. Every contrast subtracts P0;
OL1(all) is not referenced to OL1(h). Subtract ordinary-final loss and
count-pooled logical sparsity at fixed size/scope/κ. All pairs share
initial-parameter and training-schedule hashes, with λ=b=1 for OL1.

The user selected a two-panel 14M layout: (a) loss effect and (b) sparsity effect.
No T4/P0 or T7/P0 checkpoints are available at 70M, so the earlier cross-size
all-minus-h view is replaced by the supported 14M comparisons. κ is categorical,
with all five values explicit. Validation covers 338 complete 2,048-token blocks
from 500 documents, excluding the 1,444-token tail. The ordinary-final losses
and canonical FP16 counters are reused without re-evaluation.

## Publication caption

**Pressure effects relative to matched no-pressure controls at 14M.**
(a) Validation-loss change and (b) model-wide sparsity change when adding
OL1 pressure, holding threshold placement and κ fixed. All four curves use
the matching T4/P0 or T7/P0 no-pressure checkpoint as reference, as stated
in the legend. Green/teal and purple dashed curves show T4/Ph and T7/Ph;
blue and orange solid curves show T4/Pall and T7/Pall. Colours, line styles,
and uniform circular markers match Figure 6. Negative loss changes are better;
positive sparsity changes mean more logical zero-operand products, in percentage
points. The horizontal line denotes zero change. Lines join separately trained
settings, not training trajectories or independent replicates. Each setting
has one seed and one final checkpoint, with matched initialization, data order,
and training budget. No independent-seed uncertainty is represented.

## Result and proposed manuscript writing

> Relative to matched no-pressure checkpoints, h-only pressure lowers validation
> loss under both threshold scopes at κ≤0.1. At κ=0.5, seven-site all-pressure
> adds 12.096 percentage points of model-wide sparsity for +0.1265 loss, whereas
> four-site all-pressure adds 2.498 points for +0.3783 loss. At that threshold,
> h-only pressure adds 1.277 points for +0.0291 loss under seven-site thresholding,
> and 0.012 points for +0.0630 loss under four-site thresholding. Pressure scope
> therefore changes the quality-sparsity effect of adding pressure, even when
> threshold placement and strength are held fixed.

## Caveats and provenance

Pall denotes pressure on all selected threshold sites: a,m,h,z for T4, and
additionally q,k,v for T7. Changing pressure scope also changes the equal-tensor
normalization of its objective. These data do not isolate direct Q/K/V pressure
or attribute effects to normalization versus target selection. Five κ values
are five settings, not five seed replications. Logical sparsity is not runtime
speedup. The absent 70M P0 controls prevent these matched contrasts at 70M.

Source: [paper_effect_figures.py](../paper_effect_figures.py),
[13_rebuild_paper_figures.py](../13_rebuild_paper_figures.py),
[checkpoint table](../data/paper-checkpoints.json), and
[20 unrounded contrast pairs](../data/paper-derived.json).
Proposed text is retained here for manuscript adoption, not silently inserted.

Regenerate only this figure with `13_rebuild_paper_figures.py --only-pressure-scope`.
The seven focused evidence tests pass, including direct integer-count contrast
checks and agreement with the retained pressure-versus-none appendix. The PDF
was rendered and visually inspected; its fonts are embedded. Other figure PDFs
and the manuscript are unchanged by this revision.

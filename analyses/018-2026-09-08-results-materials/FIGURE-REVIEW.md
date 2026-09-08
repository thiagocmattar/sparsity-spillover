# Figure review after the second user revision

The previous revision removed useful structure. This revision restores the
requested curves and distribution layout while retaining paper-width lettering.
Every figure now has a descriptive title. The artwork remains 5.5 inches wide;
ordinary text is at least 8 pt, with conventional smaller math scripts.
Figures 01 and 03 use compact 7.5-pt legends below the plots.

The [official ICLR template](https://github.com/ICLR/Master-Template/blob/master/iclr2026/iclr2026_conference.tex)
asks for clean, legible artwork and uses a 5.5-inch text width. The 8-pt
criterion is our review choice, not a claimed numeric conference requirement.

| Figure | Current decision | Remaining interpretation or layout limit |
| --- | --- | --- |
| 01 | User-supplied two-line title; keep the five training families. Show A0 clipping only with a more visible grey dashed path and open markers. | 16 trained endpoints plus ten A0 clipping inputs; six clipping coordinates in view. Full-cohort data and frontier membership remain separate. |
| 02 | User-approved final horizontal layout, titled "Paired intervention effects". The metric y-axes show only delta loss and delta model sparsity (pp). Below the plots, Intervention: and Paired: label two rows from the left. Actions describe operations/sites; paired cells show treatment minus reference, wrapped over two lines. Dose/action/pair text is 6.5 pt; λ/κ labels are 7 pt; the two row headings are 7.5 pt. Paired cells use black text. | The seven pair expressions identify the actual comparators; O002 explains matching doses. Doses have ordinal spacing and are not replicates. The common sparsity scale compresses small L1→OL1 effects; exact values remain in the table. |
| 02-v2 | Retained, unselected alternative: shared vertical Intervention axis, loss on the left and sparsity on the right. Numeric doses form a vertical column beside separate λ/κ group labels. | The user preferred the horizontal Figure 02. This alternative retains the same evidence and its original labels; doses increase downward with ordinal spacing. |
| 03 | Keep A4-OL1/A7-OL1 and A0 clipping only. Use the same-size A7 ceiling for every bottom-row curve; round the top-row x limits up to 30%, 50% and 90%, with round-number ticks. Move the compact legend below, remove loss units, and apply the new two-line title in the existing font/size. | 60 evaluations shown twice. The A4 vertical guide remains structural context; the plotting denominator is A7. Separate topology-specific table values are unchanged. Full clipping losses compress fine trained differences. |
| 04 | Add title; use solid operation fills and thin white boundaries. Remove all hatch patterns. | Segment order and the legend support interpretation; very small contributions remain easier to compare in the table. |
| 05 | Restore site columns and κ rows, with κ=.05 added between 0 and .5. One shared recipe legend and four retained magnitude bins on a symlog mass axis. | Finer bins at .05/.5 need new checkpoint measurements. The grid is explicitly coarse until that diagnostic is confirmed and run. Five common sites are shown; new measurements would also supply baseline a/z. |
| 06 | Title explicitly says 14M; subtitle says 15 checkpoints, ten clipping targets each. Legend includes the checkpoint count for each source family and matches the open markers. | It is repeated evaluation of 15 fixed pretrained models, not a scale comparison or 150 newly trained models. |
| 07 | Add overall title and eight-recipe legend; use “Search progress” above the left panel and “Final kernel” above the right. | The search curve still admits only fully qualified speedups. Runtime scope remains the 30 included checkpoints. |
| 08 | Add title “Theoretical sparsity ceiling by model size”; retain topology curves and actual parameter-count positions. | These are analytic reach ceilings at a fixed workload, not quality-constrained attainability or runtime speedup. |

The fine-bin request remains a data requirement rather than a styling choice.
[ACTIVATION-DIAGNOSTIC.md](ACTIVATION-DIAGNOSTIC.md) records the exact seven-
checkpoint measurement design and the repository's confirmation boundary.
No unsupported mass was assigned to a finer bin.

## Figure 01 complete-cohort review

Independent sub-agents reviewed the actual 30-checkpoint, 300-clipping-point
proof. Scientific contribution: **20/25**; design: **20/25**. These are review
rubrics, not statistical evidence or publication acceptance probabilities.

| Scientific criterion | Score | Design criterion | Score |
| --- | --- | --- | --- |
| Fit to manuscript question | 4/5 | Visual hierarchy | 4/5 |
| Matched-comparison interpretability | 3/5 | Legibility at 5.5-inch width | 4/5 |
| Completeness and fairness | 5/5 | Visual encoding | 4/5 |
| Evidence strength and claim limits | 4/5 | Overlap and layout | 4/5 |
| Scientific insight communicated | 4/5 | Standalone clarity | 4/5 |

The science audit checked every clipping point against its source sweep and
checkpoint hash, full validation coverage, integer counts and measured p=0.
The maximum p=0 loss discrepancy is 0.00005517, below the 0.0005 guard. The
PDF contains 30 distinct source paths with no substituted starting points.
The design review found no material cropping, misidentification or overlap
requiring another layout change. All trained markers remain above faint
clipping paths. Caption qualifications identify the one-seed scope, retained
A7 query/key/value gates and the 78 evaluations above the displayed loss range.

A follow-up scientific audit finds 29 strictly nondominated joint records at
18 distinct coordinates. Twenty records come from newly evaluated A7/A7-OL1
sources; none comes from corrected A4-OL1. Many are identical zero-threshold
outcomes. For A7-OL1 kappa=.5, clipping p=.7 adds 0.15665 sparsity percentage
points for 0.01619 loss versus its own measured p=0. The only prior frontier
record displaced is the A7-OL1 kappa=.1 trained endpoint, through a tiny
p=0 numerical difference; this is not claimed as a substantive improvement.

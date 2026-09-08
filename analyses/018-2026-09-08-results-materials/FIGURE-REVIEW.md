# Figure review after the second user revision

The previous revision removed useful structure. This revision restores the
requested curves and distribution layout while retaining paper-width lettering.
Every figure now has a descriptive title. The artwork remains 5.5 inches wide;
ordinary text is at least 8 pt, with conventional smaller math scripts.
Figure 01 uses the user-requested smaller 7.5-pt legend below the plot.

The [official ICLR template](https://github.com/ICLR/Master-Template/blob/master/iclr2026/iclr2026_conference.tex)
asks for clean, legible artwork and uses a 5.5-inch text width. The 8-pt
criterion is our review choice, not a claimed numeric conference requirement.

| Figure | Current decision | Remaining interpretation or layout limit |
| --- | --- | --- |
| 01 | Title “14M quality-sparsity trade-offs”; small legend below; thin grey x/y grid; dashed pressured training curves; all 15 retained clipping trajectories drawn faintly behind training points. | All 30 trained and 150 clipping inputs remain; the quality-focused range omits high-loss clipping points from view. Corrected A4-OL1/A7 clipping is unavailable. No jitter or interpolated frontier. |
| 02 | Title “Matched intervention effects (14M)”; shared categorical axis “Intervention”; numeric doses directly below each point, then separate λ/κ group labels and intervention names. Uniform circular markers replace the mixed dose key. A1-H→A4 and A4→A7 identify the paired recipes. | Doses have ordinal spacing and are not replicates. The common sparsity scale compresses small L1→OL1 effects; exact values remain in the table. The comparisons are not additive. |
| 02-v2 | Alternative orientation: shared vertical Intervention axis, loss on the left and sparsity on the right. Numeric doses form a readable vertical column beside separate λ/κ group labels. The same 29 pairs use aligned rows and horizontal stems. Original Figure 02 is retained. | Doses increase downward with ordinal spacing. Common effect scales still compress the smallest L1→OL1 changes. This is a layout alternative using the same evidence. |
| 03 | Keep the absolute/normalized rows and clipping controls; restore dashed A4/A7 theoretical ceiling lines in the raw row and identify them in the shared legend. | The A0/A1-H clipping reach is A4, not their source checkpoint topology. Full clipping losses compress fine trained differences. |
| 04 | Add title; use solid operation fills and thin white boundaries. Remove all hatch patterns. | Segment order and the legend support interpretation; very small contributions remain easier to compare in the table. |
| 05 | Restore site columns and κ rows, with κ=.05 added between 0 and .5. One shared recipe legend and four retained magnitude bins on a symlog mass axis. | Finer bins at .05/.5 need new checkpoint measurements. The grid is explicitly coarse until that diagnostic is confirmed and run. Five common sites are shown; new measurements would also supply baseline a/z. |
| 06 | Title explicitly says 14M; subtitle says 15 checkpoints, ten clipping targets each. Legend includes the checkpoint count for each source family and matches the open markers. | It is repeated evaluation of 15 fixed pretrained models, not a scale comparison or 150 newly trained models. |
| 07 | Add overall title and eight-recipe legend; use “Search progress” above the left panel and “Final kernel” above the right. | The search curve still admits only fully qualified speedups. Runtime scope remains the 30 included checkpoints. |
| 08 | Add title “Theoretical sparsity ceiling by model size”; retain topology curves and actual parameter-count positions. | These are analytic reach ceilings at a fixed workload, not quality-constrained attainability or runtime speedup. |

The fine-bin request remains a data requirement rather than a styling choice.
[ACTIVATION-DIAGNOSTIC.md](ACTIVATION-DIAGNOSTIC.md) records the exact seven-
checkpoint measurement design and the repository's confirmation boundary.
No unsupported mass was assigned to a finer bin.

# O002: Absolute comparison and T2/Ph change boxplots

## Question

How do the three model scales compare when their absolute execution regimes
are shown alongside the distribution of T2/Ph loss and latency changes across
the five threshold settings?

## Method and coverage

The figure reuses all 36 execution points from 33 checkpoints in
`data/scale-figure.json`: Base under kernel and PyTorch execution, and T2/Ph
and T7/Ph at kappa 0, 0.01, 0.05, 0.1 and 0.5 for 14M, 31M and 70M.
Panel (a) retains these 36 execution points. Panel (b) uses only the 15 T2/Ph
checkpoints: five threshold settings per size. No new training or timing is added.

In panel (b), X is categorical model size: 14M, 31M and 70M. Each size has
two boxes. Loss change is `validation loss - Base validation loss`, plotted
on the left Y axis. Latency change is
`kernel full-model latency - PyTorch Base full-model latency`, in milliseconds,
plotted on the right Y axis. Both references come from the same model size.
Latency is now an absolute difference, replacing the earlier ratio panel.

Each box has five observations, one per kappa. The box spans Q1 to Q3 and
the line marks the median. Quartiles use linear interpolation; for n = 5,
Q1, median and Q3 are the second, third and fourth sorted observations.
Whiskers reach the most extreme observed values within 1.5 IQR of the box;
observations outside those bounds appear as hollow circular outliers.
All settings contribute equally, without pooling the timing replicates as
additional observations.

| Scale | Base loss | PyTorch Base latency (ms) | Reference session |
| --- | ---: | ---: | --- |
| 14M | 5.208582500028893 | 0.6554423980056099 | Run029 |
| 31M | 4.565514347256993 | 1.0238611173053367 | Run054 |
| 70M | 4.099767337889361 | 1.6110477085782202 | Run045 |

Quality retains complete canonical validation: 500 documents, 338 complete
2,048-token blocks and a 1,444-token excluded tail. Timing retains RTX 5090,
BF16, batch one, 2,048 tokens, full logits and CUDA graphs. The provenance and
qualification boundaries of [O001](001-scale-frontier.md) apply unchanged.

## Legend and caption

**Targeted sparsification exhibits scale-dependent quality–latency trade-offs.**
(a) Absolute validation loss and full-model latency, on logarithmic axes.
(b) Paired boxplots of T2/Ph changes across five kappas per model size.
Purple boxes show loss change from Base on the left axis; teal boxes show
latency change from PyTorch Base in milliseconds on the right axis. Both
axes are linear and zero-aligned. Boxes show the interquartile range and
median, with 1.5 IQR whiskers and explicit outliers. In panel (a), circles,
triangles and squares denote 14M, 31M and 70M; dashed blue curves denote
T2/Ph and solid orange curves T7/Ph. Hollow gray markers denote Base kernel
execution and filled gray markers PyTorch Base. Lines join threshold settings
within each recipe and size. Vertical guides in panel (a) mark Base loss;
the horizontal guide in panel (b) marks zero change. No curve is fitted.

## Result and checks

| Scale | Median loss change | Median latency change (ms) |
| --- | ---: | ---: |
| 14M | -0.074203 | -0.065319 |
| 31M | 0.137406 | -0.054478 |
| 70M | 0.082550 | 0.526220 |

Negative changes indicate improvement relative to the corresponding Base
reference. All 30 changes from the 15 checkpoints are represented through
the boxes, whiskers and outliers; the data artifact retains every value.

The generator checks source hashes, the full 36-point/33-checkpoint inventory,
complete threshold families, matching Base checkpoint identities, reversible
subtraction and visibility of every coordinate. Independent checks recompute
all six sets of quartiles, medians, whiskers and outliers from the source values.
The original single-panel PDF is preserved byte-for-byte. The new PDF was
rendered and visually inspected, including both panels, the two colored axes
and their legends. Panel (a)'s source coordinates are unchanged.

## Caveats

The boxes describe variation across deliberately chosen thresholds, not
uncertainty across seeds or timing replicates. Five settings give a small,
discrete distribution; the threshold-ordered curves remain available in
panel (a). Separate axes have different units, so box heights are read against
their matching colored axis. Zero is aligned on both axes.

Base subtraction is by model scale, not by timing session. It does not remove
session variability or differences in the specialized kernels. One seed and
three fixed-budget model sizes support a descriptive comparison. No fitted
scaling law, new finding promotion or manuscript text update is implied.

## Source script and artifacts

- `03_absolute_relative.py`
- `data/scale-figure.json`: retained absolute source coordinates.
- `data/absolute-relative-figure.json`: exact differences and boxplot statistics,
  Base references, source coordinates and source/script/PDF hashes.
- `figures/02-absolute-relative-scale-frontier.pdf`

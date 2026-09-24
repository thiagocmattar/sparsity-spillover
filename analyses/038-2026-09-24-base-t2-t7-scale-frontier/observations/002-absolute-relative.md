# O002: Absolute and Base-relative quality-latency comparison

## Question

How do the three model scales compare when their absolute execution regimes
are shown alongside the loss change and latency ratio relative to each scale's
native PyTorch Base?

## Method and coverage

The figure reuses all 36 execution points from 33 checkpoints in
`data/scale-figure.json`: Base under kernel and PyTorch execution, and T2/Ph
and T7/Ph at kappa 0, 0.01, 0.05, 0.1 and 0.5 for 14M, 31M and 70M.
No training, timing, model selection or pooling is added.

For each size, the right panel uses
`x = validation loss - Base validation loss` and
`y = full-model latency / PyTorch Base full-model latency`.
The denominator is the same scale's retained PyTorch Base coordinate, not the
custom-kernel Base or the native execution of an intervened checkpoint.

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
(b) Loss change from the same-scale Base and latency divided by the same-scale
PyTorch Base latency, on linear axes. Circles, triangles and squares denote
14M, 31M and 70M. Dashed blue curves denote T2/Ph and solid orange curves
T7/Ph. Lines join threshold settings within each recipe and size. Hollow gray
markers denote Base kernel execution; filled gray markers denote PyTorch Base.
The three PyTorch Base points coincide at (0, 1) in panel (b) and are drawn
as one filled gray marker. Vertical guides mark Base loss; the horizontal
guide in panel (b) marks native Base latency. No curve is fitted.

## Result and checks

Every PyTorch Base maps exactly to (0, 1). The custom-kernel Bases remain at
x = 0 with latency ratios 0.994097, 1.080474 and 1.695860 for 14M, 31M and
70M, respectively. Ratios below one indicate lower latency than the reference;
negative loss changes indicate lower validation loss. These values are kept
without clipping or forcing the custom-kernel references to unity.

The generator checks source hashes, the full 36-point/33-checkpoint inventory,
complete threshold families, matching Base checkpoint identities, exact native
Base coordinates, reversible normalization and visibility of every coordinate.
The original single-panel PDF is preserved byte-for-byte. The new PDF was
rendered and visually inspected, including both panels and the shared legend.

## Caveats

The normalization is by model scale, not by timing session. It does not remove
session variability or differences in the specialized kernels. One seed and
three fixed-budget model sizes support a descriptive comparison. No fitted
scaling law, new finding promotion or manuscript text update is implied.

## Source script and artifacts

- `03_absolute_relative.py`
- `data/scale-figure.json`: retained absolute source coordinates.
- `data/absolute-relative-figure.json`: exact transformed coordinates,
  denominators, reference sessions and source/script/PDF hashes.
- `figures/02-absolute-relative-scale-frontier.pdf`

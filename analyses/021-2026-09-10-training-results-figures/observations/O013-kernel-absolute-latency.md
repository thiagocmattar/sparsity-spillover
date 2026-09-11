# O013 - Native baseline versus optimized absolute latency

## Question

Why does greater native-relative speedup not necessarily mean lower optimized
K050 latency for matched four-/seven-site checkpoints?

## Method, coverage and source

[09_kernel_appendix_figures.py](../09_kernel_appendix_figures.py) selects the 20
multisite checkpoints from the retained 30-checkpoint [investigation data](../investigation/data/checkpoints.json).
Rows distinguish no pressure and OL1; each curve connects the five separately
trained thresholds 0, 0.01, 0.05, 0.1 and 0.5 at equal categorical positions.
Native latency uses the reference paired with full K050. Candidate columns
use all-skips-off and full-K050 geometric-mean host latencies. Values derive
from the original 64 matched inputs, seven passes and three processes per
implementation on RTX5090, BF16 batch-one uncached 2,048-token inference with
full-vocabulary logits. No timing, training or model evaluation is rerun.

## Figure and proposed caption

[Appendix Figure 3](../figures/appendix/03-kernel-absolute-latency.pdf).

**Native-relative speedup can reflect differences in the native baseline.**
Blue diamonds and orange triangles compare four-site and seven-site recipes
at matched thresholds, without OL1 (top) and with OL1 (bottom). Columns show
native PyTorch/SDPA, K050 with all sparse skips disabled, and full K050.
Each column shares its y-range across rows; column scales differ. Lines join
separately trained settings and are not training trajectories. The native
reference is measured alongside full K050. All entries are geometric-mean
full-model latencies in milliseconds under the original matched timing protocol.

## Result and caveats

At every positive threshold, seven-site native execution is slower, while the
all-skips-off implementations are much closer. Full K050 benefits strongly
at high threshold. In all eight positive-threshold pairs, however, seven-site
has both higher native-relative speedup and higher absolute K050 latency.
At κ = 0.5 with OL1, native latencies are 0.73944/0.84409 ms and K050 latencies
0.45948/0.47337 ms for four-/seven-site respectively.

The code trace and all-skips-off control support native gate overhead as an
explanation; they do not isolate the native Q/K/V gate cost. The middle column
uses four decimal places to represent its much tighter scale without rounding
distinct tick locations into misleading labels. No speedup ratios, fits,
counter values or search-history annotations are included in the plot.

## Verification

All 12 plotted paths match the retained pressure groups, threshold order and
absolute latency fields (60 coordinates). Y-limits match within each column
and differ across columns. The five investigation tests pass; the PDF was
rendered and visually checked, including row-label placement, embedded fonts
and page bounds. No manuscript file or existing figure was modified.

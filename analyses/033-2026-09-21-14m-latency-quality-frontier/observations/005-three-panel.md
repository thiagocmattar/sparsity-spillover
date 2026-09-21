# Quality, conditional time savings and sparsity

## Question and method

How do the 14M quality-latency and sparsity-latency trade-offs relate to the
controlled evidence for which execution paths save time?

Panels (a) and (c) reuse the focused figures' same 40 checkpoints across nine
recipes. Panel (b) reproduces the six conditional controls from manuscript
Figure 24(b), measured on T7/Pall at kappa=0.5 in Run037. No measurements are
added or recomputed from rounded table entries. The generating script is
[`../02_three_panel.py`](../02_three_panel.py); coordinates, unrounded controls,
source hashes and verification results are in
[`../data/three-panel.json`](../data/three-panel.json).

Coverage and timing for (a,c): final-checkpoint validation on all 338 complete
2,048-token blocks from 500 MiniPile documents, excluding the 1,444-token tail.
Retained K050 full-model host timings use RTX5090, BF16, batch one, full logits
and 1,344 samples per checkpoint (64 inputs x seven passes x three processes).
The four sessions supply Run029:30, Run033:5, Run041:4 and Run044:1 checkpoints.
Sparsity is the pooled zero-product count divided by the full-model logical
product count, including the dense output head, expressed as a percentage.

Panel (b) uses Run037's separate controlled session. It subtracts full-execution
latency from latency with one execution path disabled, holding the checkpoint,
thresholding and other paths fixed. Whiskers span differences between the
extrema of three process means. Values and spans are checked against Run037's
raw summary, whose qualification covers 30 processes and 26,880 timing samples.
This panel is identical in data and bar styling to
[Analysis031's panel (b)](../../031-2026-09-20-quality-sites-quality/figures/01-quality-sites-quality.pdf).

## Figure and caption

[Publication PDF](../figures/04-14m-quality-savings-sparsity.pdf).

**Quality, conditional time savings and sparsity on Pythia-14M.**
(a) Validation loss and full-model latency across 40 checkpoints.
(b) Conditional time saved by each sparse execution path for T7/Pall at
kappa=0.5, with thresholding retained and other paths enabled. Positive values
indicate faster execution with that path enabled; whiskers show the span of
differences between process means, not confidence intervals. The q,k and v
bars correspond to the QK and PV paths. (c) Model-wide sparsity and full-model
latency for the same checkpoints as (a). Blue highlights T2/Ph and orange
T7/Pall; other recipes are gray. Hollow markers and vertical guides denote
Base. Lines connect settings within each recipe. Panel (b) uses a separate
controlled timing session; its conditional savings are not additive.

## Result and limits

The side panels show the quality cost of the highlighted recipes alongside
their sparsity and latency. The controlled comparison localizes the largest
conditional savings to h (+150.1 microseconds) and z (+29.0 microseconds);
the other paths do not reliably save time in this checkpoint. These controls
do not measure T2/Ph or decompose the historical latencies plotted in (a,c).
Model-wide sparsity is a logical-product measure, not the fraction of runtime
saved. Cross-checkpoint differences are descriptive: one training seed,
multiple timing sessions and a fixed full-sequence workload limit inference.

The layout follows Figure 24's three-panel proportions, typography and embedded
TrueType fonts, with one shared three-entry legend and no subtitles. Earlier
standalone PDFs and the manuscript are unchanged.

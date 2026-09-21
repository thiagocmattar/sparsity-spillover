# Latency-quality comparison near Base validation loss

## Question

How do the lower-loss 14M operating points compare when the dense cluster
around Base validation loss is shown at readable scale?

## Method and coverage

This is a close-up of [Observation 001](001-all-recipes.md), using the same
41 checkpoints and historical K050 latencies. Naive L1 and the Pareto overlay
are omitted at the user's request. The view covers validation loss
5.085--5.285 and latency 0.517--0.662 ms; 18 markers fall within these limits.
Recipe styles remain in the legend, and trajectories toward off-screen
settings are clipped at the axes. The overview shows the complete cohort.

Sources: Analysis028 `data/full-trained-results.json` and Analysis027
`data/figure-data.json`; exact coordinates, limits and visible checkpoint
identities are in [`../data/frontier.json`](../data/frontier.json).
Generating script: [`../01_plot.py`](../01_plot.py).
Coverage is identical to Observation 001: 338 complete validation blocks and
a 1,444-token excluded tail; RTX5090, BF16, batch one, 2,048 tokens, full
logits and 1,344 host samples per checkpoint across three processes,
retaining four original timing sessions.

## Figure and caption

[Publication PDF](../figures/02-14m-latency-quality-frontier-near-base.pdf).

**14M latency-quality trade-offs near Base validation loss.** A close-up of
the same full-model K050 measurements, showing 18 of the 41 checkpoints.
Colors and line patterns match the overview, with light connections and
opaque markers. Four T2/Ph and T4/Ph operating points are labeled with recipe
and dose. The vertical dotted line marks Base loss. Lower values are preferable
on both axes; the overview shows the higher-loss settings outside this view.

## Result and limits

At or below Base loss (5.208583), the lowest displayed latency is T4/Ph at
kappa=.05: 0.561801 ms with loss 5.195590. T2/Ph at kappa=.1 has lower loss
(5.151098) and takes 0.573427 ms. T4/Ph at kappa=.1 reduces latency to
0.529309 ms with loss 5.228687, above Base. These are distinct trade-offs.

The comparison inherits the cross-session, single-seed and workload limits
from Observation 001. It does not establish significance for small latency
gaps or certify quality preservation. No new benchmark or manuscript edit
is performed.

# Version 2: latency on X and validation loss on Y

## Question and method

The user requested a second version of the full overview with the axes swapped.
This figure transposes the earlier overview's exact 41 points, axis ranges
and ten recipe styles. It is unchanged by the later focused styling of
[Observation 001](001-all-recipes.md). The recipe lines remain 0.85 pt at 40%
opacity, markers remain opaque, and neither naive L1 nor a Pareto overlay is
shown. This version retains the ReLU control and earlier full-color legend.

Generating script: [`../01_plot.py`](../01_plot.py). Inputs and hashes are in
[`../data/frontier.json`](../data/frontier.json), with per-output axis metadata.
Sources are Analysis028's complete results and Analysis027's manuscript
coordinates and styles. Coverage is unchanged: 338 full validation blocks
from 500 documents, excluding the 1,444-token tail; K050 host latency on
RTX5090, BF16, batch one, 2,048 tokens and full logits, 1,344 samples per
checkpoint across three processes, retaining the four original sessions.

## Figure and caption

[Publication PDF](../figures/01-14m-latency-quality-frontier-v2.pdf).

**Quality-latency trade-offs on Pythia-14M.** Final validation loss (Y) versus
full-model K050 latency in milliseconds (X) across 41 trained checkpoints and
ten recipes. Lower values are preferable on both axes. Colors and line
patterns follow the manuscript, with thin translucent connections between
separately trained dose settings. Selected points are labeled with recipe
and dose. The horizontal dotted line marks Base validation loss.

## Result and limits

This changes only the orientation of the same measured comparison. Numerical
results and the cross-session, single-seed and workload qualifications in
Observation 001 apply unchanged. No new measurement or manuscript edit is made.

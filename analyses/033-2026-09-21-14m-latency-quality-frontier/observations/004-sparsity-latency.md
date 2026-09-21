# Model-wide sparsity and full-model latency

## Question and method

How does observed full-model latency vary with model-wide sparsity across the
14M recipes, highlighting T2/Ph and T7/Pall?

This is the same 40-checkpoint, nine-recipe cohort as the focused quality-latency
overview. Base, T2/Ph and T7/Pall retain their highlighted styles; other recipes
are gray. Naive L1, GeLU-to-ReLU and point callouts remain excluded.
No new benchmark or model evaluation is performed.

The X coordinate is 100 times the pooled integer zero-product count divided
by the full-model logical-product count, including the dense vocabulary head.
It is model-wide logical sparsity, not the fraction of zero activations or
the fraction of runtime saved. No percentages are averaged across layers.
The Y coordinate retains the historical K050 host latency in milliseconds.

Sources: Analysis028 `data/full-trained-results.json`, Analysis027's manuscript
coordinates/palette and Analysis030's Figure 23 style. The generating script
is [`../01_plot.py`](../01_plot.py); exact coordinates, integers, source hashes
and per-output checkpoint identities are in [`../data/frontier.json`](../data/frontier.json).

Coverage: all 338 complete 2,048-token validation blocks from 500 MiniPile
documents, with a 1,444-token excluded tail. Timing uses RTX5090, BF16, batch
one, 2,048 tokens, full vocabulary logits and 1,344 synchronized host samples
per checkpoint (64 inputs x seven passes x three processes). The four retained
sessions contribute Run029:30, Run033:5, Run041:4 and Run044:1 checkpoints.

## Figure and caption

[Publication PDF](../figures/03-14m-sparsity-latency.pdf).

**Model-wide sparsity and full-model latency on Pythia-14M.** Forty trained
checkpoints across nine recipes. T2/Ph is blue and T7/Pall orange, using the
manuscript's line and marker styles. Base is a hollow marker with a vertical
sparsity guide. Other recipes appear in gray. Lines connect separately trained
settings and are guides to their ordering, not measured interpolation. The
legend labels only Base and the two highlighted recipes.

## Result and limits

The highlighted recipes occupy different sparsity ranges while both show
lower latency toward their high-threshold endpoints. T2/Ph at kappa=.5 has
5.339% model-wide sparsity and latency 0.506323 ms; T7/Pall at kappa=.5 has
27.483% sparsity and latency 0.473366 ms. The comparison does not hold model
quality fixed; the companion quality-latency figure displays that trade-off.

This descriptive cross-checkpoint plot does not isolate the causal effect of
sparse execution. Small timing differences are not significance claims;
different sessions, one training seed and the fixed full-sequence workload
limit interpretation. There is no manuscript edit or finding promotion.

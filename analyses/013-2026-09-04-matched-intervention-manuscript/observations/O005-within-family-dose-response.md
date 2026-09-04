# O005 - Gate dose relative to each family's zero-threshold endpoint

## Question and method

Does raising kappa have the same direction of quality change at each size?
For A4+OL1@4 and A7+OL1@7 at 14M, 70M, and 410M, subtract the family's
kappa=0 loss and opportunity from every tested threshold. Use all 30
pressured checkpoints and same-pass full-validation measurements. Each point
covers 338 complete blocks; the excluded tail contains 1,444 tokens.

## Figure and caption

[Figure PDF](../figures/05-dose-response-by-size.pdf).
Within-family change in validation loss against change in model-wide
opportunity. Lines order kappa=0,.01,.05,.1,.5. All panels share a loss-change
range that includes both improvements and deteriorations.

## Result and caveats

The high-threshold 14M/70M conditions raise loss relative to their own
zero-threshold pressured baselines, while the 410M high-threshold conditions
lower it. This isolates threshold-dose response from adopting the one-sided
gate architecture. It does not equal the cost relative to A0; the full
absolute-loss plots and tables supply that comparison. The cross-size
differences remain confounded by exposure per parameter and architecture.

Source scripts: [evidence.py](../evidence.py), [plots.py](../plots.py),
[01_build.py](../01_build.py). All source identities are in figure_data.json.

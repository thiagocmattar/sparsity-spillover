# 001 — A4/A7 with none, all-site, or h-only OL1

## Question

How do pressure target sets change the 14M quality–sparsity operating points
at fixed gate topology and threshold? This observation preserves the complete
six-recipe, five-threshold table approved by the user for paper use.

## Method, coverage, and caption

Source script: `../01_build_tables.py`. Outputs: `../TABLE.md`,
`../TABLE-ordinary-final.md`, `../TABLE-logical-pass.md`, and `../results.json`.
Sources are Runs 011, 015, 012, 013, 014, and 032, respectively. Run 012 is
identified through Analysis 009's realized h-only pressure audit. All 30
source endpoints and identities are reconciled; full source hashes and
checkpoint identities accompany the machine-readable table.

**Table caption.** Pythia-14M model-wide logical sparsity and final-checkpoint
validation loss for A4 and A7 with no pressure, pressure on every gate site,
or pressure only at h. Each cell gives `(S_model %, loss in nats)`; lower loss
is better. A4 uses one-sided gates at a,m,h,z. A7 additionally uses symmetric
gates at q_post,k_post,v. OL1 uses lambda = 1 and b = 1. Five thresholds,
kappa = 0, 0.01, 0.05, 0.1, 0.5, are evaluated at one matched seed (1234),
712 updates, and 1,493,172,224 training tokens. Validation includes all
500 documents through 338 complete blocks, excluding the 1,444-token tail.
Sparsity is calculated from pooled integer product counters, including the
dense LM-head denominator. It is not measured runtime acceleration.

The archival table preserves the previously displayed loss-pass mixture:
the four original overview curves use eager logical-pass loss, while the two
h-only columns use ordinary final loss. The largest difference between these
passes is 0.000105806356 nats. Both uniform-pass tables are provided explicitly;
the loss convention must be named when inserting a table into the paper.

## Observed result

At kappa = 0.5, A7 rises from 15.3868% without pressure to 16.6636% with
h-only pressure and 27.4827% with seven-site pressure. The corresponding
archival losses are 5.702895, 5.732049, and 5.829407. Thus h-only pressure
recovers about 10.56% of the all-site sparsity increment at that threshold;
it also incurs a smaller validation-loss increase.

At the same threshold, A4 reaches 10.2155%, 10.2274%, and 12.7134% for
none, h-only, and all-four-site pressure, with archival losses 5.659678,
5.722666, and 6.037982. H-only pressure changes A4 sparsity very little there.

With h-only pressure fixed, A7 exceeds A4 sparsity by 1.0904 percentage points
at kappa = 0.05 (10.1261% versus 9.0356%), at nearly equal ordinary losses
(5.194959 versus 5.195590). At kappa = 0.5, the sparsity gap becomes
6.4362 points (16.6636% versus 10.2274%), with ordinary losses 5.732049
and 5.722666. At the four smaller thresholds, each h-only recipe has lower
loss than its corresponding pressure-free and all-site-pressure endpoints.

## Interpretation limits and status

These are single-seed, fixed-budget, one-scale descriptive comparisons.
Poor recovery of the all-seven-site pressure increment by h-only pressure
does not establish direct Q/K/V pressure as necessary: all-site pressure
also adds a,m,z and changes the objective's target averaging. The h-only
scalar averages six site/layer tensor means; A4-all averages 24 and A7-all
averages 42. No new runtime, causal-mediation, or scaling-law claim follows.

The user approved the data for paper use on 17 September 2026. The analysis
records that approval and its evidence; manuscript prose has not been edited.
Relevant sections are the quality–sparsity overview and paired-pressure
discussion in `manuscript/draft/training-results.tex`.

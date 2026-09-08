# Proposed results argument

The evidence supports a conditional result: pressure's added value depends on
the gate threshold and where it acts. Selected high-threshold recipe orderings
survive larger model sizes, while low-threshold responses change. Architecture
ceilings explain part of the increase in raw model-wide sparsity; activation
mass distinguishes exact zeros from magnitude shrinkage. Runtime gains require
a separately qualified implementation.

## 1. Establish the 14M trade-off

Use Figure 01: all 30 included training conditions plus A0/A1-H post-clipping.
Each legend entry has its own dose sweep: eight trained recipe families and
two separate clipping-control series. Lines connect all evaluated points in
increasing dose or target order, including dominated points. These curves show
the response to dose, not a Pareto envelope. The loss window remains 5.04–6.15; the eight control
clipping evaluations above it are retained in Figure 03 and the tables.
The [series table](tables/overview-series.md) records every point and its connection order.

Suggested text:

> The 14M study spans distinct quality–sparsity regimes. Local L1 pressure on
> the ReLU FFN hidden activation produces the lowest observed validation loss:
> at λ=1, loss is 5.1023 with 3.9493% model-wide sparsity, compared with
> 5.2696 and 2.7141% for unpressured A1-H. Broader gate placement reaches a
> different region: A7-OL1 at κ=.5 reaches 27.4827% at loss 5.8294.

Corrected four-site A4-OL1 contributes no globally nondominated point among
these 30 conditions, despite improving parts of the within-A4 comparison.
Do not describe a within-family improvement as a global frontier improvement.
The excluded historical h-only A4 pressure cohort contributes no current result.

The appendix's Figure 06 shows all 150 available 14M clipping evaluations.
Clipping the λ=1 L1 checkpoint gives 4.7982% sparsity at loss 5.1070 (p=.1)
and 5.6441% at loss 5.1441 (p=.2). The retained post-hoc study does not clip
A7 or corrected A4-OL1. It therefore cannot identify an optimal joint
training-plus-clipping recipe. Uniform clipping is not TEAL's greedy allocation.

## 2. Explain the added effect of each intervention

Figure 02 preserves the aligned two-row design and is titled “Intervention
effects along the sparsification ladder.” The gate additions are labeled
G+(x) at a,m,z and Gpm(x) at q,k,v. Points increase in dose left to right. Every value is
treatment minus the stated reference. The sequence builds an explanation,
not an additive decomposition or a curriculum used to train one model.

> At fixed A4 gates, adding OL1 improves both outcomes at κ=0 and .01.
> At κ=.5 it adds 2.4979 model-sparsity percentage points at a .3783 loss
> cost. A7 responds differently: at zero threshold OL1 slightly worsens
> both outcomes, while at κ=.1 it adds 1.3717 points with a .000819 loss
> increase. At κ=.5 its gain rises to 12.0959 points at a .1265 loss cost.
> Pressure has no uniform benefit across doses and topologies.

The seven comparisons are A0→A1-H; A1-H→A1-H-L1; L1→OL1 at h;
A1-H→A4 at κ=0; A4→A4-OL1; A4→A7; A7→A7-OL1. The table identifies
the exact source endpoints for all 25 pairs. Within fixed A4/A7 topology,
only the specified pressure objective is added. Across topologies, its sites
and equal-tensor normalization change, so this is a recipe response rather
than a gate-by-pressure interaction under one fixed objective.

Small loss differences are descriptive, not evidence of equivalence or
significance. At κ=0, added Q/K/V gates are mathematical identities, yet the
trained no-pressure A4/A7 endpoints differ by −.002123 loss and +.005632
sparsity points. Treat this residual as a numerical/implementation control
discrepancy. A1-H→A4 also changes the boundary derivative convention at h.

## 3. Introduce the ceiling, then assess transfer

Figure 08 supplies the missing structural view. Its y axis is the analytic
all-zero reach ceiling, not an observed zero rate. It uses the actual parameter
counts of the three architectures and the same uncached T=2048 workload.
OL1 changes training, not the selected-site ceiling, so A4/A4-OL1 and
A7/A7-OL1 share curves. The integer-count table fixes the numerator and unit.
The same A4/A7 ceilings now appear as dashed vertical lines in Figure 03's
raw-sparsity row; clipped controls share the A4 evaluation-site reach.

> A7's all-block ceiling rises from 29.952% at 14M to 49.424% at 70M and
> 87.245% at 410M. The dense output head consumes a much larger fraction of
> the counted workload at 14M. Thus an increase in raw model-wide sparsity
> across sizes need not imply a proportional increase in learned zero rates.

Figure 03's top row plots absolute loss against raw model-wide sparsity.
The bottom row plots A0-relative loss against ceiling utilization. Both rows
include A4-OL1/A7-OL1 at all five κ values and A0/A1-H with all ten uniform
clipping targets p=0,.1,…,.9. The full loss range makes the clipping cost
and the nonmonotonic absolute quality across model sizes visible.

> At κ=.5, A7-OL1 improves on A4-OL1 by 14.7692, 5.0057 and 9.0241
> model-sparsity points at 14M, 70M and 410M, reducing loss by .2086,
> .1736 and .0703 respectively. At κ=0, A4-OL1 is better on both outcomes
> at 14M and 70M, while A7-OL1 is better at 410M. The result supports
> partial transfer of complete recipes, with a regime-dependent response.

The larger cohorts lack no-pressure A4/A7 controls, so they do not replicate
the isolated pressure effect. The same 1.493 billion input-token budget gives
106.1, 21.2 and 3.7 tokens/parameter. Peak learning rate is 1e-3 at 14M/70M
and 3e-4 at 410M. A0 loss is 5.2086, 4.0998 and 4.5475 respectively.
Do not treat these three one-seed points as a scaling law or diagnose
undertraining as the cause of the 410M response.

### How to use the proposed normalization

Keep `U_arch = R_model / R_model_max` as an explanatory second view. It is
the ratio of observed zero products to selected-site reachable products,
not a model-size-invariant quality or speed score. At κ=.5, A7 realizes
91.75%, 82.15% and 92.40% of its respective ceilings. A4-OL1 realizes more
of its own narrower ceiling at all 15 matched dose/size pairs; this does not
reverse A7-OL1's high-threshold advantage in raw sparsity and loss.

For the clipped controls, use the evaluation clipping sites {a,m,h,z}; these
have A4 reach, even when the source checkpoint is A0 and even at p=0.
Normalizing by the source A0 topology would divide by zero and would describe
the wrong intervention. Unmodified A0 retains an undefined ratio in the
trained table. The bottom-row loss reference remains the unmodified same-size A0.

Natural zeros outside selected reach remain in the U_arch numerator, so the
ratio need not be bounded by one. Save `U_reach`, which excludes those zeros,
as a sensitivity measure. High-threshold 14M A4-OL1 gives 99.07% versus
98.63%, with .05657 raw sparsity points outside reach. A7 reaches all counted
block operations, so both ratios equal the block-only zero-product fraction.
Even that fraction changes its operation weights with architecture/workload.
Figure 04 decomposes the high-threshold raw totals; its full rates remain in tables.

## 4. A focused activation case

Figure 05 restores the distribution grid: columns h,m,q,k,v and rows κ=0,
.05,.5. Each panel shows four measured magnitude-bin masses: exact zero,
(0,.001], (.001,.01] and >.01. A shared legend identifies A0/A4-OL1/A7-OL1;
the same A0 is repeated across rows. The y axis is logarithmic above .01%
and linear near zero to retain empty bins. Lines connect categorical masses,
not a reconstructed continuous density.

The requested finer bins at .05 and .5 are not available in the stored
statistics. Figure 05 remains a coarse distribution view until the proposed
checkpoint diagnostic is confirmed and run. The diagnostic would also provide
the missing A0 a/z counts and permit all seven site columns.

> At κ=.5, both recipes make over 99.8% of FFN hidden activations exactly
> zero. Their attention operands differ: A4-OL1 has 47.64% of query values
> and 56.12% of key values within magnitude .01, yet fewer than .001%
> exact zeros at either site. A7-OL1 instead makes 93.54% of queries,
> 94.54% of keys and 98.71% of values exactly zero. The contrast explains
> how A7 can supply zeros to additional attention operations.

Smaller magnitude does not imply more exact zeros: query RMS is .146 for
A4-OL1 and .483 for A7-OL1 at this dose. RMS remains in the table rather
than in per-panel annotations. Marginal mass alone does not identify a
causal spillover mechanism; gate and pressure placements both differ.

These are retained count summaries, not full histograms. A0 lacks a/z
near-zero measurements. The common post-Wo diagnostic is not pre-Wo z and
is omitted from the figure. The all-site table preserves it under its exact
operational name. Activation and logical-product counts come from separate
complete validation passes and are not forced to be bitwise equal.

## 5. Close with qualified execution evidence

Figure 07 is redrawn from Run 029, restricted to the same 30 included
checkpoints. Its left curve retains the qualified incumbent on fixed c30
across 42 eligible proposals; the excluded cohort was absent from this search
subset. Its right panel and all runtime summaries are recomputed for 30 models.
The figure now has a shared recipe legend, an overall title and the panel
label “Search progress”; the qualified-incumbent definition remains in the caption.

> The qualified incumbent reaches 1.7830× on the search checkpoint.
> Final K050 qualifies on all 30 included checkpoints, with 1.2340× geometric
> mean speedup (range 1.0059–1.7832×). The descriptive sparsity–speedup fit
> has R²=.8167. Sparse paths add 4.32% geometrically averaged acceleration
> over the fused no-skip ablation, but help only 14/30 checkpoints.
> Disabling attention skipping improves all 30 checkpoints.

The last ablation has 1.2506× geometric-mean speedup; K050 is .9868× as
fast as that ablation geometrically. P0 qualifies on only four included
checkpoints, so its .8910× mean is not a whole-cohort ranking.

This is a BF16, batch-one, uncached 2048-token RTX5090 workload. Canonical
sparsity is measured in FP16. Weights, gates and model quality vary across
checkpoints; the association does not establish causal proportionality or
equal-quality gains. Ablation ratios compare separately measured normalized
timings. No cached-decoding or larger-model speedup follows. If adopted, these
30-checkpoint numbers must replace the draft's original 35-checkpoint summary.

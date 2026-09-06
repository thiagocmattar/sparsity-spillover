# Observation 002: fixed logical opportunity does not fix runtime behavior

## Question

For the same checkpoint, inputs, and canonical `R_model`, how much can changing
the kernel/site/layer policy alter speed and correctness?

## Method and coverage

Figure 2 joins 25 same-condition transitions from four stages of Run 025:

- four 14M development comparisons from the minimally adapted P0 path to fused
  signed-exact warp compaction K001;
- six 14M complete-validation comparisons from fast K012 to robust K013;
- twelve 70M complete-validation comparisons from generic K009 to
  topology-specific K016; and
- three 410M development comparisons from early K004 to frozen K010.

Every endpoint is first paired against native eager timing in its own session.
The plot then joins the two endpoint ratios for the same checkpoint. This keeps
checkpoint and `R_model` fixed but does not make separate GPU sessions a paired
fresh-process experiment.

## Legend and caption

Marker shape identifies intervention family and colour identifies model size.
Red crosses are policies that fail their declared numerical gate. The native
break-even line is shown in each panel. A rising line means the later policy has
a higher native-relative ratio; it does not by itself identify which source
change caused the improvement.

## Result

The 14M P0-to-K001 change improves all four development sentinels, with a median
optimized/baseline speedup ratio of 1.0536x; three of four input-cluster ratio
intervals exclude 1. The later K012-to-K013 change illustrates the correctness
cost of search: it repairs the three invalid K012 endpoints, but only one of six
raw speed ratios rises and none has an interval wholly above 1.

At 70M, K016 is faster than K009 on 9/12 checkpoints and seven ratio intervals
exclude 1, but the final K016 A4-OL1 interior checkpoints at `kappa` 0.01, 0.05,
and 0.1 fail complete numerical validation. At 410M, all three recorded K010
development comparisons improve. The A0 jump from 0.4712x to 0.9431x is caused
by native fallback and is therefore a deployment-policy improvement, not a
sparse-kernel gain. The two sparse high-`kappa` changes improve only modestly,
from 1.0158x to 1.0205x and from 1.0034x to 1.0160x.

The central observation is therefore both positive and cautionary: specialized
implementation choice materially changes realized latency at unchanged
`R_model`, while an aggressive or transferred choice can be fast, slow, or
numerically invalid.

## Caveats

These transitions mix development and complete-validation scopes, called out in
each panel. They come from a single bounded trajectory rather than replicated
agent and human baselines. The policy transitions sometimes change which
eligible operations use sparse code, and dense fallback is included because it
is a legitimate frozen deployment decision. QK/PV attention never became
sparse.

## Source

- Figure: [`../figures/02-same-rmodel-kernel-search-transitions.pdf`](../figures/02-same-rmodel-kernel-search-transitions.pdf)
- Generator: [`../02_plot.py`](../02_plot.py)
- Transition table: [`../same-rmodel-optimization.csv`](../same-rmodel-optimization.csv)
- Reduction/verification: [`../01_reduce.py`](../01_reduce.py)

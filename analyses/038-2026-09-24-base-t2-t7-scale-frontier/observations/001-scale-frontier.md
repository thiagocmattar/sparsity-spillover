# O001: Base, T2/Ph and T7/Ph across scales

Status: complete, with five verified new endpoints and 15 qualified timing
processes. All agreed artifacts are retained locally; task Pods are terminated.

## Question

Does extending the threshold topology from T2 to T7 under h-only OL1 improve
the measured validation-loss versus full-model latency frontier at the middle
Pythia scale, and how does the comparison relate to14M and70M?

## Method and coverage

Run055 adds five randomly initialized31M T7/Ph endpoints, kappa0,.01,.05,.1,.5,
matched to Run054 on initial parameters/RNG, data order,712 updates,1.493B input
tokens, optimizer, precision and retained measurements. T7 uses one-sided a,m,h,z
and symmetric post-RoPE q,k plus v; OL1 targets only h. All500 MiniPile validation
documents contribute338 complete2048-token blocks; the1444-token tail is excluded.
The loss coordinate is the ordinary final-checkpoint evaluation with FP16
autocast, matching the training recipe. BF16 native-versus-kernel losses are
retained separately for latency qualification; they do not replace that coordinate.

New latency points require three fully qualified RTX5090 processes per checkpoint,
BF16,B1/T2048,full50304 logits,CUDA graphs,64 validation inputs and seven paired
passes. The geometric mean of448 samples is computed per process, then the
geometric mean across the three processes. New kernel policy is the31M Run045
opt073 port with a/m gates wired into paired LayerNorm. All338 blocks qualify
against native eager under the inherited numerical bounds. Runtime work counts
are untimed and checked independently; logical zero-product opportunity remains
a distinct quantity.

Historical Base/T2/Ph/T7/Ph coordinates are selected unchanged from Analysis037.
Its source hashes verified before launch; all five kappas are present in each
historical recipe/scale. New31M T7 completes36 execution points/33 checkpoints.
Base PyTorch means CUDA-graph replay with the same full-model workload.

## Legend and caption

**Quality–latency trade-offs across model scales.**
Validation loss versus full-model latency for Pythia-14M, Pythia-31M and
Pythia-70M. Circles, triangles and squares denote the three scales. Dashed blue
curves denote T2/Ph and solid orange curves T7/Ph; hollow gray points denote
Base kernel and filled gray points Base PyTorch. Colored lines connect
kappa 0, 0.01, 0.05, 0.1 and 0.5 within each recipe and scale. Three vertical
gray guides, labeled by model size, mark the Base losses. Both axes are logarithmic. Timings use
RTX 5090, BF16, batch one, 2,048 tokens and full logits. No curve is fitted.

The paper-ready presentation adopts the reference manuscript figure's
typography, colors, line weights and compact layout, with no kappa labels,
workload subtitle or footer. The user corrected the earlier 30M label to 31M;
the middle model still has 30,494,720 parameters. All numerical coordinates
and the measured Pareto set are unchanged.

## Results

| 31M T7/Ph kappa | Validation loss | Kernel latency (ms) | On measured cross-scale frontier |
| --- | ---: | ---: | --- |
| 0 | 4.894259 | 0.986853 | No |
| 0.01 | 4.874077 | 0.985217 | No |
| 0.05 | 4.893721 | 0.895587 | Yes |
| 0.1 | 4.942717 | 0.800644 | Yes |
| 0.5 | 5.520965 | 0.606869 | No |

Two new middle-scale T7/Ph endpoints extend the discrete measured Pareto set.
Kappa 0.05 is 1.41% faster than the previous 31M T2/Ph kappa 0.1 point,
with 0.136513 higher loss; kappa 0.1 gives a larger latency reduction at
loss 4.942717. Neither point is dominated by any of the 36 execution points.
The 0 and 0.01 points are dominated by 31M T2/Ph 0.1; the 0.5 point is
dominated by 14M T2/Ph 0.05. This supports a limited extension of the observed
frontier, not uniform superiority of the wider topology.

At each matched kappa, 31M T7/Ph is 3.15-11.86% faster than 31M T2/Ph, with
0.172793-0.300547 higher validation loss. Across the full sweep, T7/Ph latency
ranges from 0.606869 to 0.986853 ms. The exact data retain all three process
values and both backends. Base kernel/PyTorch references remain present at
all three sizes; the 31M Base is (4.565514, 1.106255/1.023861 ms).

The complete 36-point Pareto set contains 15 execution points. This uses
literal coordinate dominance with no uncertainty margin; it is a description
of the measurements, not a statistically established ranking. In particular,
the 1.41% gap above is between different timing sessions. Connecting lines
within recipes and the vertical Base-loss guides are not a fitted scaling curve.

## Caveats

One seed, three sizes, fixed token budget, different timing sessions and
size-specific kernels. Small latency differences require caution; the point
ordering is descriptive. Greater logical sparsity does not imply lower latency.
T7-vs-T2 changes five gate sites together and does not isolate an individual
site's causal effect. The user-approved manuscript integration is documented
separately in [O003](003-manuscript-integration.md).

Numerical qualification is specific to the reported checkpoints and validation
coverage. Run055 retained an unqualified H200 preflight and one failed block of
the untrained kappa0.5 RTX5090 checkpoint; the latter isolates to short-row
projection rounding. The trained kappa0.5 checkpoint passed a separate complete
RTX5090 preflight with bitwise-equal logits. All 15 final measurements passed
their own full338-block qualification under unchanged bounds. The kernel is not
claimed bitwise-equivalent for arbitrary unseen checkpoints or inputs.

## Sources and scripts

- Run055 training/verifier and latency receipts.
- Analysis037 data/combined-figure.json and its unchanged source coordinates.
- `01_collect.py`: new endpoint verification and raw timing reduction.
- `02_plot.py`: scale-figure.json, empirical Pareto set and publication PDF.

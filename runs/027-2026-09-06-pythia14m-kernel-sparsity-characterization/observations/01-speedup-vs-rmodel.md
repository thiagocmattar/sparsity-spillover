# Speedup versus achieved R_model

## Question

How does the fixed K019+K018 algorithm perform at each existing trained 14M
endpoint, and how much of its latency benefit is due specifically to skipping
zero-valued operands?

## Sources, method and coverage

`../prelaunch/inputs.json` pins all 35 trained 14M endpoints of Analysis 013:
30 main-study endpoints plus five separately labeled historical A4+OL1@h
endpoints. This is not a screen of every archived 14M run or of post-hoc
clipping settings. Source logical counts use all 500 MiniPile validation
documents / 338 complete 2048-token blocks; 1444 tail tokens are excluded.
Counts are pooled before dividing. The model-wide denominator includes the
dense language-model head. Canonical R_model is the source diagnostic value,
not an estimate inferred from measured latency or this BF16 execution.

Four modes run in each of three fresh processes on one RTX 5090, BF16 B1 T2048,
full vocabulary logits, eager causal SDPA, no KV cache or compilation.
Time the full synchronized forward, including dynamic gates and kernel input
packing. Exclude equal resident-input staging, loading, extension compilation,
and one-time weight transpose. Retain both synchronized host and CUDA-event
times; the plotted estimand uses host time. All modes see the same 64 randomly
selected validation blocks (seed 2504), seven randomized paired timing passes,
and the same pretraining checkpoint. Checkpoint order is randomized per process
round, seeds 2702/2703/2704. No mode is selected after observing performance.

Each ratio is the exponential of the mean paired log latency ratio over
3 x 64 x 7 observations. Intervals use 10,000 percentile bootstrap draws with
crossed process and input resampling, retaining the seven repeats within each
cluster (seed 2706). These intervals characterize timing variability for these
inputs/processes, not independent training seeds, hardware populations, or
variation across cloud hosts. Three process clusters limit interval precision.

Numerical qualification compares all full logits and pooled shifted-label loss
on all 338 validation blocks / 691886 prediction targets per process. Fixed
gates: elementwise atol .25 + rtol .02, relative L2 <= .02, loss difference
<= .001 nat/token. A failure remains visible but is not claimed as a valid
speedup. The BF16 validation losses are reported alongside the speed table;
cross-checkpoint speed changes are not claims of matched language-model quality.

## Figure caption and legend

**Pythia-14M full-model performance versus model-wide logical sparsity.**
Each point is one trained endpoint; colors and markers identify intervention
families, and thin segments connect dose-ordered endpoints without asserting
a fitted relationship. Open historical markers denote A4 gates with h-only
OL1 pressure, not corrected four-site OL1. (a) Stock eager latency divided by
the fixed fused sparse implementation's latency. (b) Latency of the identical
joint-fusion implementation with zero skipping disabled, divided by its
zero-skipping latency. The latter is a mechanistic control, not an optimized
dense baseline. Values above one indicate faster sparse execution; the dashed
line is parity. Error bars are paired 95% timing bootstrap intervals. Grey
crosses, if present, indicate numerical gate failures. All complete validation
blocks qualify numerical fidelity; timing uses the fixed 64-block subset.

## Result

Awaiting execution and reduction; no measured claim is made yet.

## Caveats and nonclaims

R_model counts potential zero products across six transformer operations;
this kernel skips only h/W2 and z/Wo products. It is neither removed FLOPs nor
an Amdahl/runtime ceiling. Actual BF16 eligible zeros are measured separately
in Observation 02. Fusion/layout changes can produce gains even at negligible
sparsity. The controlled no-skip comparison isolates skip behavior conditional
on this particular kernel; it does not show that this kernel is globally
optimal. A cross-checkpoint association can be affected by activation topology,
gate overhead, operand structure, and CPU/GPU behavior. It cannot by itself
establish a causal effect of training pressure or isolate training quality.

## Reproduction

Source: `../01_benchmark.py`, `../03_matrix.py`, `../10_reduce.py`, `../11_plot.py`.
Raw evidence: `../artifacts/c??-r?/`; audited table: `../results/summary.csv`.
Output: `../figures/01-speedup-vs-rmodel.pdf`.
No manuscript section changed; paper-facing motivation is recorded in the run README.

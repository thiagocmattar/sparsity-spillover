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
not an estimate inferred from measured latency or this BF16 execution. The
source logical passes use FP16 autocast (`Run004/diagnostics.py`, reused by
Runs009/011/012/013/014/015). Thus the main x-axis is the canonical training-study
diagnostic, not a newly counted BF16 R_model. Observation02 separately measures
native BF16 operands eligible for this kernel.

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
a fitted relationship. Grey hexagonal markers identify historical A4 gates with
h-only OL1 pressure, not corrected four-site OL1. Filled markers qualify in
all three processes; open crossed markers fail the fixed numerical gate and
are measured latency ratios, not qualified equivalent-model speedups.
(a) Stock eager latency divided by
the fixed fused sparse implementation's latency. (b) Latency of the identical
joint-fusion implementation with zero skipping disabled, divided by its
zero-skipping latency. The x-axis retains the source FP16-autocast logical
diagnostic; timing is BF16. The latter ratio is a mechanistic control, not an optimized
dense baseline. Values above one indicate faster sparse execution; the dashed
line is parity. Error bars are paired 95% timing bootstrap intervals; intervals
smaller than the markers are not separately visible. Numerical fidelity is
tested on all complete validation blocks; timing uses the fixed 64-block subset.

## Result

All 35 checkpoints / 105 processes completed. Only six checkpoints qualify:
A1-H+OL1 lambda 0.1, A4 kappa 0.5, A4+OL1@4 kappa 0.5, A7 kappa 0.5,
A7+OL1@7 kappa 0.5, and historical A4+OL1@h kappa 0.5. The last five have
bitwise-identical full-validation logits in all three processes. Their
qualified overall speedups span 1.490-1.808x; every qualified interval lies
above parity. All 35 QKV-fusion-only variants qualify.

The 29 sparse rejections fail the elementwise logit criterion on 1-68 distinct
blocks out of 338. All relative-L2 errors are below 0.00304 (bound 0.02), and
all absolute pooled loss changes are below 0.000092 nat/token (bound 0.001).
Small average errors do not override the fixed elementwise gate. Across all
35, measured native/sparse ratios span 0.9697-1.8093x; the endpoints of this
unfiltered range are not both qualified. A0 is a 0.9697x ratio and is unqualified.

The largest qualified speedup is A7+OL1@7 kappa 0.5: 1.807615x, 95% CI
[1.802063, 1.812120], at canonical R_model 27.482684%. Its median native/sparse
latencies are 3.679201/2.033881 ms. Qualified A7 kappa 0.5 has R_model 15.386813%
but almost the same speedup, 1.803428x [1.798824, 1.808976]. Their native BF16
eligible zero fractions are 99.8838% and 99.8605%, respectively. The data show
a topology-dependent plateau, not a proportional conversion of global logical
opportunity into speedup. This is consistent with the kernel's restricted h/z
reach, but does not isolate every CPU/GPU bottleneck.

Values and uncertainty for every dose appear in `../results/per-variant.md`
and `../results/summary.csv`. Near-identical points may overlap in the figure.

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
Both final PDFs were rendered with Poppler and inspected at 180 dpi; labels,
markers, legends, uncertainty and the complete zero-inclusive ranges are intact.

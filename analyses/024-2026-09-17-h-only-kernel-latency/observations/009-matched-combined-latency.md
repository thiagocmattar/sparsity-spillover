# 009: Matched 14M and 70M conditions on one latency panel

## Question

Compare model-wide sparsity and final-kernel full-model latency on one shared
panel, retaining only recipe/kappa conditions available at both model sizes.

## Method and coverage

Select 22 checkpoints at each size: baseline A0, the unpressured A1-H ReLU
control, and A4/A7 with OL1(h) or OL1(all), each at kappa
`{0, 0.01, 0.05, 0.1, 0.5}`. The 14M-only pressure-free A4/A7 grids and
single-site pressure grids are excluded. The 14M source names `A4+OL1@4` and
`A7+OL1@7` map to their respective OL1(all) families. All 22 normalized
recipe/kappa keys must match exactly between sizes; no checkpoint is imputed.

Read the unchanged verified reductions `data/results.json` and Run035's
`results/70m-final-kernel.json`, checking their hashes against Figures03/04's
retained provenance. The x coordinate is canonical FP16 model-wide logical
product sparsity: pooled integer zero-product count divided by model product
count, expressed as a percentage. The y coordinate is the geometric mean of
all 1,344 raw synchronized host timings per checkpoint, as retained in each
source reduction. No averaging over checkpoints or new measurements occurs.

Both sizes use the RTX5090/BF16, batch-one, uncached 2,048-token causal
full-logit workload with vocabulary 50,304. Timings use 64 fixed validation
inputs, seven paired passes and three fresh processes. Numerical qualification
covers all 338 complete blocks from 500 validation documents, excluding the
1,444-token tail. All plotted checkpoints qualified. No training, evaluation,
benchmarking or cloud resources are launched for this display change.

## Legend and caption

**Matched Pythia-14M and Pythia-70M conditions: model-wide sparsity versus
full-model final-kernel latency.** Open markers represent 14M; filled markers
represent 70M. Colors and shapes distinguish baseline A0, 1-site A1-H,
4-sites A4* and 7-sites A7. Short-dashed lines connect OL1(h) checkpoints;
long-dashed lines connect OL1(all) checkpoints. Each of the eight curves
connects increasing kappa within one model size, topology and pressure recipe.
Baseline and ReLU controls are isolated. The single panel uses shared linear
axes, with the latency limits fitted to the data rather than forced to zero.
All latencies, including baseline markers, are from the respective candidate
kernel. There are 44 points representing 22 matching conditions per model.

## Result

The matched view retains the source observations while showing both model
sizes in common units on one pair of axes. Observed latency spans approximately
0.46-0.65 ms at 14M and 1.62-3.32 ms at 70M. The common linear latency scale
compresses the differences among 14M checkpoints. Pressure recipe and model
size are encoded in the legend without direct text annotations on the curves.

## Logarithmic latency variant

The user-requested [log-y version](../figures/09-14m-70m-matched-sparsity-latency-log-y.pdf)
uses the identical 44 points and eight family curves, with a linear sparsity
axis and logarithmic latency axis. Open/filled markers and both dashed pressure
styles remain unchanged. Tick labels retain milliseconds; equal vertical
distances now represent equal latency ratios rather than equal differences
in milliseconds. This gives the within-14M variation more vertical space
without changing any measurements, qualification or interpretation limits.

Reproduce with `09_plot_matched_combined_latency.py --log-y`. The separate
`data/matched-combined-latency-log-y.json` records the logarithmic scale, tick
values, multiplicatively padded positive limits, source hashes and PDF hash.
The script asserts exact point and curve equality with the retained linear
version. The one-page PDF was rendered and visually inspected, with all fonts
embedded. The original linear PDF and its data JSON remain byte-for-byte
unchanged. For this variant, use the caption above with "shared linear axes"
replaced by "a shared linear sparsity axis and logarithmic latency axis."

## Caveats

Matching means the same topology, pressure recipe and kappa, not identical
training budgets, checkpoint quality or execution implementations across sizes.
14M uses K050; 70M uses the qualified `k050-70m-v2` shape port. The port did
not receive an equal optimization search budget. Physical GPU/host sessions
also differ, including Run029 versus Run033 within the 14M cohort. This is a
descriptive comparison, not a causal effect of sparsity or a scaling-law test.
Logical sparsity is not measured acceleration; the paired native latency
denominator is not plotted. No seed uncertainty is inferred from the grid.
Overlapping markers remain separate observations in the retained data.

## Sources and reproduction

Run `09_plot_matched_combined_latency.py` to generate
[Figure09](../figures/09-14m-70m-matched-sparsity-latency.pdf) and
[`data/matched-combined-latency.json`](../data/matched-combined-latency.json).
The JSON retains source hashes, identities, original and normalized families,
kappas, qualification, pooled integer counts, plotted coordinates, curve
membership/order, axis limits and the PDF hash. Original benchmark and
implementation details remain in [Observation003](003-final-latency-topology.md)
and [Observation004](004-70m-final-latency-topology.md).

The script checks matching keys, 44 point identities, qualification, integer
sparsity reconstruction, eight separate family curves, one panel and axis
coverage. The PDF was rendered and visually inspected; all fonts are embedded.
Previous figures, including the two-panel Figure08, and source reductions
remain unchanged. No manuscript text or finding promotion is made.

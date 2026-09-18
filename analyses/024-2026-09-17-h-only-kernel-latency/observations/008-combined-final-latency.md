# 008: 14M and 70M sparsity versus final full-model latency

## Question

Display the retained 14M and 70M model-wide sparsity versus full-model latency
results together, preserving the family distinctions in Figures03 and04.

## Method and coverage

This figure combines the verified plotted cohorts from
`data/final-latency-topology.json` and `data/70m-final-latency-topology.json`.
The script verifies their upstream source hashes and numerical qualification,
then reproduces their points, family connections and subtle pressure labels.
There are 58 distinct checkpoints: 36 at 14M and 22 at 70M. The 14M panel keeps
the prior exclusion of four single-site naive-L1 checkpoints. The 70M panel
contains only available checkpoints; no pressure-free multisite or single-site
OL1 grid is imputed. Seven family curves appear at 14M and four at 70M.

Both panels use canonical FP16 integer-pooled model-wide logical sparsity and
the geometric mean of 1,344 raw synchronized host latency measurements per
checkpoint. The measurement workload is RTX5090, BF16, batch one, uncached
2,048-token causal inference with full 50,304-vocabulary logits. Numerical
qualification covers all 338 complete blocks from 500 validation documents,
with the 1,444-token tail excluded. This is a display-only change: no training,
new evaluation, benchmarking or cloud resources.

## Legend and caption

**Model-wide sparsity versus full-model final-kernel latency for Pythia-14M
and Pythia-70M.** Left: K050 on 36 retained 14M checkpoints. Right: the qualified
K050-derived 70M port on 22 retained checkpoints. Colors and markers identify
Baseline (A0), 1-site (A1-H), 4-sites (A4*) and 7-sites (A7). Dashed curves
connect increasing intervention strength within each recipe and model size;
small no-pressure, OL1(h) and OL1(all) labels identify the families. Both axes
are linear and independently fitted in each panel, with common units. All
latencies, including baseline markers, use the respective candidate kernel.

## Result

The combined view retains the observed latency-versus-sparsity patterns of
both source figures. Latencies are approximately 0.46-0.65 ms at 14M and
1.62-3.32 ms at 70M. The h-only curves reach low observed latency at less
model-wide logical sparsity than the corresponding all-site curves. This
does not establish equal-quality performance or a causal effect of sparsity.
The separate fitted ranges keep the 14M differences legible alongside 70M.

## Caveats

The implementations differ: 14M uses K050 and 70M uses `k050-70m-v2`, which
adapts dimensions and matches the native 70M attention reduction schedule.
The port did not receive the 14M kernel's optimization search budget. Physical
GPU/host sessions also differ. This is therefore a descriptive comparison
under matched measurement definitions, not an unchanged-kernel scaling test.
Independent axis ranges mean visual slopes and distances cannot be compared
directly between panels. Neither panel plots the paired native denominator;
candidate latency alone does not establish native-relative acceleration.
Training-seed and causal limitations from the source observations remain.

## Sources and reproduction

Run `08_plot_combined_latency.py` to generate
[`Figure08`](../figures/.archive/08-14m-70m-final-sparsity-latency.pdf) and
[`data/combined-final-latency.json`](../data/combined-final-latency.json).
The JSON records both source chains, all points and family connections, panel
limits and the PDF hash. Numerical and implementation details remain in
[Observation003](003-final-latency-topology.md) and
[Observation004](004-70m-final-latency-topology.md).

The script checks all 58 point identities, qualification, family membership,
source hashes, curve counts and axis coverage. The PDF was rendered and
visually inspected, and all fonts are embedded. Existing figures and source
reductions remain unchanged. No manuscript text or finding promotion is made.

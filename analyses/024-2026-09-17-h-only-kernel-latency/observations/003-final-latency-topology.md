# 003 - Final K050 latency grouped by topology

**Question.** How does full-model latency under the final kernel vary with
model-wide sparsity across the baseline, single-site, four-site and seven-site
checkpoint groups?

**Request.** On 18 September 2026 the user requested a new figure with
model-wide sparsity on x and final-kernel full-model latency on y, followed by
later styling guidance. The request said three labels but explicitly listed
four; this figure uses the four named groups. The user's next instruction
replaces the zero-based latency axis with a scale fitted to the data.

**Method and coverage.** Reuse all 40 qualified checkpoints in
`data/results.json`: 1 A0, 9 A1-H conditions (ReLU, naive L1 and OL1),
15 A4 conditions and 15 A7 conditions (no, all-site and h-only pressure).
Each point is one checkpoint, not an average across its topology group.
The final K050 latency is the geometric mean of 1,344 synchronized host
full-logit graph-replay timings, from 64 timing inputs, seven passes and
three independent processes. No new measurement or native latency is added.
Canonical FP16 model-wide sparsity uses pooled integer logical-product counts.
Numerical qualification covers 338 complete blocks from all 500 validation
documents; the 1,444-token tail is excluded.

**Caption and legend.** Full-model final K050 latency versus model-wide
logical sparsity for 40 Pythia-14M checkpoints. Gray stars: Baseline (A0);
green circles: 1-site (A1-H); blue diamonds: 4-sites (A4*); orange triangles:
7-sites (A7). Both multisite labels include no-pressure, all-site OL1 and
h-only OL1 variants. BF16 inference on RTX5090, batch 1, T=2048, full
50,304 logits; milliseconds per full-sequence forward. All points are shown,
the latency axis spans 0.44-0.67 ms to show the measured range with padding,
and no connecting line, fit or point annotation is used.

**Result.** The displayed K050 latencies span 0.459476-0.651573 ms.
The baseline is 0.651573 ms. Topology grouping changes only the presentation;
it does not replace individual values with group means or imply an ordering
of all checkpoints by topology.

**Limits.** Logical sparsity is distinct from measured runtime savings.
Checkpoints differ in validation quality and pressure settings. Five A7/h-only
points were measured on a second physical RTX5090/host session, so absolute
cross-session comparisons remain descriptive. Topology colors are not a causal
decomposition, and overlapping points are not independent training-seed replicates.

**Source and output.** `../03_plot_final_latency.py`, the unchanged source
`../data/results.json` and its SHA-256 in
`../data/final-latency-topology.json`. Output:
`../figures/03-14m-k050-sparsity-latency-topology.pdf`.

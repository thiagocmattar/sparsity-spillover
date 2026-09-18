# 004: 70M final port latency by topology

## Question

How does full-model latency vary with pooled model-wide logical sparsity for
the available 70M checkpoints using a correctly qualified K050-derived port?
This is the approved 70M counterpart of Figure03, without a new kernel search.

## Method and coverage

[Run035](../../../runs/035-2026-09-18-pythia70m-k050-port/README.md) owns the port
and measurements. All 22 retained step712 checkpoints from Runs018/034 are used:
A0 GeLU, A1-H ReLU, and A4/A7 with OL1(all) or OL1(h), each at kappa
0,.01,.05,.1,.5. No pressure-free A4/A7 or A1-H+OL1 grid is available at 70M.
Weights, initialization identity, training data order and training recipe are
unchanged; no retraining occurs.

The same Run033 protocol is used: RTX5090, BF16, batch 1, T2048, uncached causal
full 50304-logit inference, pinned Python3.12/Torch2.11.0+cu128/Transformers5.12.1,
runtime seed 2801 and timing seed 2504. One physical GPU executes all 66 fresh
processes sequentially. Each checkpoint pools 64 inputs x7 passes x3 processes,
1344 paired samples. Latency is the geometric mean of raw synchronized host
times; speedup uses paired native-SDPA-graph/candidate-graph ratios.

Every process qualifies all 338 complete blocks from 500 validation documents:
692224 input tokens,691886 prediction tokens,1444 excluded tail tokens. All 66
pass unchanged numerical bounds; maximum absolute loss delta is 3.95206e-8.
Full 338-block activation/weight/occupancy and issued/skipped-work diagnostics
exist for each checkpoint. All 705 returned files were size/SHA256 verified
before Pod deletion. Canonical FP16 S_model pools the original integer logical
counts; it is separate from BF16 operand and executed-work diagnostics.

The initial v1 port failed 14/338 baseline blocks. Substitution tests isolated
attention. The final `k050-70m-v2` preserves the exact-zero MMA bypass while
matching the observed native 70M unsplit M128/N128/D64 schedule. The six preflight
sentinels and all 22 final checkpoints qualify under this one frozen version.
Failed v1 records remain available and never enter this figure.

## Legend and caption

**Pythia-70M, K050-derived final port.** Full-model candidate latency versus
model-wide logical-product sparsity for 22 retained checkpoints. Gray stars,
green circles, blue diamonds and orange triangles denote Baseline(A0),
1-site(A1-H),4-sites(A4*) and7-sites(A7). Dashed lines connect increasing kappa
within each pressure recipe; small OL1(h)/OL1(all) labels identify the four
curves. All y values, including A0 and ReLU, use the candidate port. The native
graph is a checkpoint-specific speedup denominator and is not plotted.
The latency axis fits the data and does not start at zero. Connections are
visual guides, not regressions or connections between different recipes.

## Result

| Family at kappa 0.5 | S_model (%) | Candidate latency (ms) | Paired speedup |
|---|---:|---:|---:|
| A4+OL1(all) |35.596219|1.626444|1.0942x|
| A4+OL1(h) |29.235859|1.619968|1.1002x|
| A7+OL1(all) |40.601872|1.748420|1.0996x|
| A7+OL1(h) |32.232517|1.618370|1.1880x|

The h-only endpoints attain lower or similar observed latency despite lower
S_model than the corresponding all-site endpoints. A4 h-only/all-site latency
is close; no equivalence test is claimed. All four kappa 0.5 endpoints exceed
their own native graph reference. The other 18 checkpoints are slower, with
overall speedups ranging 0.4997–1.1880x. Native baseline latency is 1.663883ms,
whereas its candidate-port latency is 3.313247ms. A lower point on this plot
therefore does not by itself establish acceleration over native inference.

## Caveats

This is one training seed and one new GPU/host session. Three fresh processes
measure repeatability, not training-seed uncertainty; their largest speedup
range is 0.584%, and no confidence intervals or equivalence claims are made.
The 14M and 70M workloads use different physical GPU/host sessions, weights and
shape-specific implementations. The 70M port has not received the 14M kernel's
optimization search budget. This establishes compatibility and the measured
conditional benefit of this port, not a general scaling law or the best 70M
kernel performance. Loss/weights differ across trained conditions, and these
timings do not isolate the causal contribution of direct Q/K/V pressure.
S_model is logical opportunity, not activation zero fraction or runtime gain.
No manuscript text or finding promotion is made by this analysis.

## Source and reproduction

Run035's `16_reduce.py` writes
[`results/70m-final-kernel.json`](../../../runs/035-2026-09-18-pythia70m-k050-port/results/70m-final-kernel.json)
from verified raw attempts. [`04_plot_70m_final_latency.py`](../04_plot_70m_final_latency.py)
writes [`Figure04`](../figures/.archive/04-70m-k050-port-sparsity-latency-topology.pdf)
and [`data/70m-final-latency-topology.json`](../data/70m-final-latency-topology.json),
including source hash, plotted points and connection/annotation identities.
The PDF was rendered and visually checked; fonts are embedded. Figure03 and
its original 14M source reduction are unchanged.

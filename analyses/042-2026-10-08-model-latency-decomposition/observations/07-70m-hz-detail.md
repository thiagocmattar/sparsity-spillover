# 07-70m-hz-detail

**Question:** Which independently measured components form the 70M H/Z output block?

**Method and coverage:** BF16, batch 1, 2,048 tokens, resident inputs, uncached full 50,304-way logits. Each mode/threshold has 64 timing inputs x 7 paired passes x 3 fresh processes = 1,344 uninstrumented samples. Numerical qualification covers all 500 validation documents, 338 complete blocks (692,224 input tokens; 1,444 excluded tail tokens). CUPTI uses 4 inputs x 3 replays in process 1, giving 12 traced forwards per mode/threshold. 14M comes from Run064; 31M/70M from Run065. Both use RTX 5090, but the 14M physical GPU/session differs.

**Sources:** Run064 results/decomposition.json and Run065 results/decomposition.json; exact identities in ../provenance/sources.json. Main tables in ../README.md.

**Aggregation:** Host latency uses pooled geometric means and sample quantiles. GPU categories use arithmetic sums of measured CUPTI activity means. The H/Z output block pools disjoint named activities within each layer; the definition is fixed across models. Shares divide by that mode’s measured GPU-active union; no host-time rescaling. Exact definitions and fixed colors are in ../LABELS.md.

**Caption and legend:** 70M, kappa 0/0.1. Every bar is the same H/Z output block used in the common decomposition. All components keep the parent purple color; fixed patterns identify H gate/preparation, H down (W2), Z gate + down (Wo), residual, or an indivisible fused H/Z consumer. H gate/preparation sums separately measured H gate and feature compaction. Z gate + down sums the measured Z gate/projection or their fused equivalent. The solid fused category contains the complete measured consumer, including prepass and fallback where present. Its internal components are not assigned separate latencies. Missing components have no independent measurement, not a demonstrated zero cost. The same complete legend and microsecond axis are used in all three detail figures. All values are arithmetic sums of actual CUPTI activity means; these accounting groups are not independent contiguous timed intervals.

**Result:** H down (W2) dominates the skip-disabled H/Z output block; it is a component of the common purple category. All Z products stay dense.

**Caveats:** One training seed; different checkpoint/gate/kernel across model sizes, and separate physical 14M GPU/session. Tiled threshold is present only at 14M. Native is dense execution on the same gated checkpoint; no-skip retains the gate. No confidence interval, architecture-only scaling claim or independent internal fused-phase wall times are implied.

**Source script:** ../02_report.py (data: ../01_collect.py).

[Figure PDF](../figures/07-70m-hz-detail.pdf) · [Full tables](../README.md)

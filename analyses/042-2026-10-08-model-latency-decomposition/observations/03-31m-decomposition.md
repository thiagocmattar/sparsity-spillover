# 03-31m-decomposition

**Question:** Where does 31M forward GPU activity go, and what do the H/Z branches cost per layer?

**Method and coverage:** BF16, batch 1, 2,048 tokens, resident inputs, uncached full 50,304-way logits. Each mode/threshold has 64 timing inputs x 7 paired passes x 3 fresh processes = 1,344 uninstrumented samples. Numerical qualification covers all 500 validation documents, 338 complete blocks (692,224 input tokens; 1,444 excluded tail tokens). CUPTI uses 4 inputs x 3 replays in process 1, giving 12 traced forwards per mode/threshold. 14M comes from Run064; 31M/70M from Run065. Both use RTX 5090, but the 14M physical GPU/session differs.

**Sources:** Run064 results/decomposition.json and Run065 results/decomposition.json; exact identities in ../provenance/sources.json. Main tables in ../README.md.

**Aggregation:** Host latency uses pooled geometric means and sample quantiles. GPU categories use arithmetic sums of measured CUPTI activity means. The branch bundle pools disjoint named activities within each layer. Shares divide by that mode’s measured GPU-active union; no host-time rescaling.

**Caption and legend:** 31M at kappa 0 and 0.1. Upper panels sum per-stage CUPTI means from 12 traced forwards; color and hatching identify execution categories. Absent/fused categories are not independent measurements; zero CUDA-activity context layout is a view. Lower panels sum the H/Z gates, down projections, feature compaction and residual activities within each layer, making native separate stages comparable with an indivisible fused stage. The layer bundle is an accounting group, not an additional timed kernel. All three model figures share the upper microsecond axis and lower vertical axis. Gaps and host overhead are excluded from these active budgets. No event or CUPTI number is rescaled to the primary uninstrumented host total. CUPTI means are descriptive; distribution data are retained per original stage/kernel in the CSVs.

**Result:** The joint sparse consumer is slower than the equivalent native branch bundle; savings in other stages compensate.

**Caveats:** One training seed; different checkpoint/gate/kernel across model sizes, and separate physical 14M GPU/session. Tiled threshold is present only at 14M. Native is dense execution on the same gated checkpoint; no-skip retains the gate. No confidence interval, architecture-only scaling claim or independent internal fused-phase wall times are implied.

**Source script:** ../02_report.py (data: ../01_collect.py).

[Figure PDF](../figures/03-31m-decomposition.pdf) · [Full tables](../README.md)

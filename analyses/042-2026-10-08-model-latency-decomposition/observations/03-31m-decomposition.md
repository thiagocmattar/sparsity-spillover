# 03-31m-decomposition

**Question:** Where does 31M forward GPU activity go, and what do the H/Z branches cost per layer?

**Method and coverage:** BF16, batch 1, 2,048 tokens, resident inputs, uncached full 50,304-way logits. Each mode/threshold has 64 timing inputs x 7 paired passes x 3 fresh processes = 1,344 uninstrumented samples. Numerical qualification covers all 500 validation documents, 338 complete blocks (692,224 input tokens; 1,444 excluded tail tokens). CUPTI uses 4 inputs x 3 replays in process 1, giving 12 traced forwards per mode/threshold. 14M comes from Run064; 31M/70M from Run065. Both use RTX 5090, but the 14M physical GPU/session differs.

**Sources:** Run064 results/decomposition.json and Run065 results/decomposition.json; exact identities in ../provenance/sources.json. Main tables in ../README.md.

**Aggregation:** Host latency uses pooled geometric means and sample quantiles. GPU categories use arithmetic sums of measured CUPTI activity means. The H/Z output block pools disjoint named activities within each layer; the definition is fixed across models. Shares divide by that mode’s measured GPU-active union; no host-time rescaling. Exact definitions and fixed colors are in ../LABELS.md.

**Caption and legend:** 31M at kappa 0 and 0.1. Upper panels sum per-stage CUPTI means from 12 traced forwards into nine fixed semantic categories. All model sizes and modes use the same category order, colors, patterns and complete legend. The purple H/Z output block includes gates, sparse preparation, W2, Wo and residual combination, whether fused or separately measured. No fused consumer is split into unmeasured internal phases. Bars are activity budgets, not chronological timelines. Lower panels show the same H/Z output block per layer; line colors encode execution mode: blue native, green selected sparse, red selected no-skip, gray additional 14M generic. All three model figures share the upper microsecond axis and lower vertical axis. The category is an accounting group, not an additional timed kernel. Gaps and host overhead are excluded. No event or CUPTI number is rescaled to the primary uninstrumented host total. CUPTI means are descriptive; distribution data are retained per original stage/kernel in the CSVs.

**Result:** The sparse H/Z output block is slower than the equivalent native block; savings in other stages compensate.

**Caveats:** One training seed; different checkpoint/gate/kernel across model sizes, and separate physical 14M GPU/session. Tiled threshold is present only at 14M. Native is dense execution on the same gated checkpoint; no-skip retains the gate. No confidence interval, architecture-only scaling claim or independent internal fused-phase wall times are implied.

**Source script:** ../02_report.py (data: ../01_collect.py).

[Figure PDF](../figures/03-31m-decomposition.pdf) · [Full tables](../README.md)

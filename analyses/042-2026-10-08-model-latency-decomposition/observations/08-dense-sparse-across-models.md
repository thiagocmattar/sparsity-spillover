# 08-dense-sparse-across-models

**Question:** How do dense and sparse forward budgets compare across model sizes with one fixed semantic legend?

**Method and coverage:** BF16, batch 1, 2,048 tokens, resident inputs, uncached full 50,304-way logits. Each mode/threshold has 64 timing inputs x 7 paired passes x 3 fresh processes = 1,344 uninstrumented samples. Numerical qualification covers all 500 validation documents, 338 complete blocks (692,224 input tokens; 1,444 excluded tail tokens). CUPTI uses 4 inputs x 3 replays in process 1, giving 12 traced forwards per mode/threshold. 14M comes from Run064; 31M/70M from Run065. Both use RTX 5090, but the 14M physical GPU/session differs.

**Sources:** Run064 results/decomposition.json and Run065 results/decomposition.json; exact identities in ../provenance/sources.json. Main tables in ../README.md.

**Aggregation:** Host latency uses pooled geometric means and sample quantiles. GPU categories use arithmetic sums of measured CUPTI activity means. The H/Z output block pools disjoint named activities within each layer; the definition is fixed across models. Shares divide by that mode’s measured GPU-active union; no host-time rescaling. Exact definitions and fixed colors are in ../LABELS.md.

**Caption and legend:** Final direct comparison of native dense arithmetic and the selected sparse consumer at 14M, 31M and 70M, for kappa 0 and 0.1. Each size has an adjacent dense/sparse pair. Colours, hatching, category order, legend and microsecond scale are identical across both panels and model sizes. The purple H/Z output block always includes gates, sparse preparation, W2, Wo and residual combination. Labels identify the 14M tiled threshold and 31M/70M scalar thresholds. Dense means native execution of that same gated checkpoint, not a different ungated model; sparse selects the corner-specialized 14M consumer or the existing scalar consumers at 31M/70M. Bars sum actual CUPTI activity means from 12 profiled forwards; end labels show GPU-active microseconds. This is a component budget, not uninstrumented host latency; gaps and host overhead are outside these bars. Original timing distributions remain in the CSVs. Different gates/consumers and the separate physical 14M GPU session restrict cross-size interpretation to a descriptive comparison.

**Result:** The common categories expose the growing 31M sparse H/Z output cost and the smaller 70M sparse H/Z output cost versus native, while keeping vocabulary projection and attention comparable.

**Caveats:** One training seed; different checkpoint/gate/kernel across model sizes, and separate physical 14M GPU/session. Tiled threshold is present only at 14M. Native is dense execution on the same gated checkpoint; no-skip retains the gate. No confidence interval, architecture-only scaling claim or independent internal fused-phase wall times are implied.

**Source script:** ../02_report.py (data: ../01_collect.py).

[Figure PDF](../figures/08-dense-sparse-across-models.pdf) · [Full tables](../README.md)

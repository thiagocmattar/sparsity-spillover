# 01-full-forward-latency

**Question:** How does complete-forward latency vary by model, gate and execution mode?

**Method and coverage:** BF16, batch 1, 2,048 tokens, resident inputs, uncached full 50,304-way logits. Each mode/threshold has 64 timing inputs x 7 paired passes x 3 fresh processes = 1,344 uninstrumented samples. Numerical qualification covers all 500 validation documents, 338 complete blocks (692,224 input tokens; 1,444 excluded tail tokens). CUPTI uses 4 inputs x 3 replays in process 1, giving 12 traced forwards per mode/threshold. 14M comes from Run064; 31M/70M from Run065. Both use RTX 5090, but the 14M physical GPU/session differs.

**Sources:** Run064 results/decomposition.json and Run065 results/decomposition.json; exact identities in ../provenance/sources.json. Main tables in ../README.md.

**Aggregation:** Host latency uses pooled geometric means and sample quantiles. GPU categories use arithmetic sums of measured CUPTI activity means. The branch bundle pools disjoint named activities within each layer. Shares divide by that mode’s measured GPU-active union; no host-time rescaling.

**Caption and legend:** Bars show pooled host geometric means; black whiskers are pooled sample p10-p90; three black diamonds show fresh-process geometric means. These are descriptive distributions, not confidence intervals. Green bars identify the selected sparse implementation. Hatching means arithmetic skipping is disabled; the trained gate stays active. All panels share the millisecond axis. 14M has a tiled gate in every mode, including native and generic. 31M/70M have scalar gates in every mode. Different gates/consumers and physical GPU sessions make the cross-size comparison descriptive.

**Result:** The selected sparse implementations reduce latency versus their own gated native references; the 14M specialized-versus-generic comparison is distinct from either comparison with native.

**Caveats:** One training seed; different checkpoint/gate/kernel across model sizes, and separate physical 14M GPU/session. Tiled threshold is present only at 14M. Native is dense execution on the same gated checkpoint; no-skip retains the gate. No confidence interval, architecture-only scaling claim or independent internal fused-phase wall times are implied.

**Source script:** ../02_report.py (data: ../01_collect.py).

[Figure PDF](../figures/01-full-forward-latency.pdf) · [Full tables](../README.md)

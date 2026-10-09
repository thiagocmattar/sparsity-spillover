# Global quality latency

Question: Does vocabulary-head intervention extend the observed global quality–latency envelope?

Sources: Analysis037/038/041 cohort coordinates; Run063 process timing samples; Analysis042 Run064/065 decomposition and original checkpoint metadata; Runs066/068 canonical validation and Runs067/068 verified timing results. Exact identities are in [latency source receipts](../provenance/latency-sources.json); reduced values and source attempts are in [latency.json](../data/latency.json).

Coverage: 138 existing execution points at 14M, 31M and 70M; all measured modes retained. Validation coordinates/counts use all 500 MiniPile documents, 338 complete 2,048-token blocks, 691,886 predictions and the declared 1,444-token excluded tail. Geometry observations use training boundaries rather than validation batches.

Method: Canonical final validation loss versus synchronized complete-forward host geometric mean. Pareto dominance minimizes both, with at least one strict inequality.

Caption and legend: Two panels: all sizes and explicitly labeled 14M detail. Colors identify base, other scalar, scalar T2, scalar T7, tile T2, tile T7, head8 and head16 recipes. Model shapes are circle/triangle/square; head8/head16 are diamond/star. Open markers are native dense arithmetic with the trained gates, × is no-skip, and dots connect the observed envelope. Thin whiskers show process minimum/maximum where available; no confidence interval is inferred.

Observed result: 20 global nondominated points; none of the 18 head execution points extends the envelope. Run064 T2 tile κ=0.1 is nondominated at loss 5.596746 and 0.486441 ms.

Caveats: Mixed sessions, BF16 optimized body kernels versus FP16 native-body/head kernels. No combined body/head timing exists. The envelope is descriptive; dotted connections do not assert measured intermediate points. Expanded axes are labeled detail.

Source script: [03_plot_frontiers.py](../03_plot_frontiers.py), with [01_collect_latency.py](../01_collect_latency.py) collecting the evidence. Output: [01-global-quality-latency.pdf](../figures/01-global-quality-latency.pdf). Tables: [report](../results/REPORT.md). No manuscript edit or promoted claim.

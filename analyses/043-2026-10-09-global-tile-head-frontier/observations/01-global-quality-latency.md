# Global quality latency

Question: Does vocabulary-head intervention extend the observed 14M quality–latency envelope?

Sources: Analysis037/041 retained 14M cohort coordinates; Run063 process timing samples; Analysis042 Run064 14M decomposition and original checkpoint metadata; Runs066/068 canonical validation and Runs067/068 verified timing results. Plot style reference: Analysis041 Figure04 panel (b) and its 02_quality_latency_frontier.py script. Exact identities are in [latency source receipts](../provenance/latency-sources.json); reduced values and source attempts are in [latency.json](../data/latency.json).

Coverage: 87 existing 14M execution points; all measured modes retained. Validation coordinates/counts use all 500 MiniPile documents, 338 complete 2,048-token blocks, 691,886 predictions and the declared 1,444-token excluded tail. Geometry observations use training boundaries rather than validation batches.

Method: Canonical final validation loss versus synchronized complete-forward host geometric mean. Pareto dominance minimizes both, with at least one strict inequality.

Caption and legend: One 14M-only detail panel with linear axes, following Analysis041 Figure04 panel (b). Historical five-threshold T2/Ph is dashed blue; T7/Ph is solid orange. Base has a vertical loss guide. Green and purple mark T2/T7 body tiles with head OFF; pink/orange mark head8/head16. Open markers mean native dense arithmetic with the trained gates; × is no-skip. The dotted step envelope is the observed Pareto set. Whiskers show process min/max, not confidence intervals. The larger latency range includes every new head execution.

Observed result: 12 14M nondominated points; none of the 18 head execution points extends the envelope. Run064 T2 tile κ=0.1 is nondominated at loss 5.596746 and 0.486441 ms.

Caveats: Mixed sessions, BF16 optimized body kernels versus FP16 native-body/head kernels. No combined body/head timing exists. Connected sweep segments show threshold order, not interpolation; the descriptive step envelope does not assert measured intermediate executions. The detail axes do not start at zero.

Source script: [03_plot_frontiers.py](../03_plot_frontiers.py), with [01_collect_latency.py](../01_collect_latency.py) collecting the evidence. Output: [01-global-quality-latency.pdf](../figures/01-global-quality-latency.pdf). Tables: [report](../results/REPORT.md). No manuscript edit or promoted claim.

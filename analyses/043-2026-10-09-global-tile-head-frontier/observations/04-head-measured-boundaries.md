# Head measured boundaries

Question: How does each head intervention change projection, head-path, and complete latency?

Sources: Analysis037/038/041 cohort coordinates; Run063 process timing samples; Analysis042 Run064/065 decomposition and original checkpoint metadata; Runs066/068 canonical validation and Runs067/068 verified timing results. Exact identities are in [latency source receipts](../provenance/latency-sources.json); reduced values and source attempts are in [latency.json](../data/latency.json).

Coverage: Runs067/068, 6 checkpoints × 3 modes × 3 timing boundaries; 3 processes/checkpoint, 64 inputs × 7 paired timing passes/process. Validation coordinates/counts use all 500 MiniPile documents, 338 complete 2,048-token blocks, 691,886 predictions and the declared 1,444-token excluded tail. Geometry observations use training boundaries rather than validation batches.

Method: Use source synchronized host geometric means unchanged. Error bars are process min/max, not confidence intervals. Gate/consumer occupancy fractions pool full-validation integer counts.

Caption and legend: Projection-only, gate+cast+projection, and complete-forward panels use separate labeled zero-based scales. Blue native dense, orange dotted custom no-skip, green hatched custom sparse. Every tick declares head tile size, Ph/Ph+head, body κ and head κ. Body projections remain native dense FP16.

Observed result: Head16 κ=0.5 sparse saves 15.55% on the head path and 3.10% on full forward versus native dense; κ=0.1 changes full forward by +1.03%. Head8 full sparse execution is 7.19–8.86% slower than dense despite bypass savings.

Caveats: Boundaries overlap and are measured separately; never stack or subtract them. No full head-model CUPTI decomposition or head-gate-off ablation is retained. Absolute process variability is visible for Run068, and the matched ratios are retained in data.

Source script: [04_plot_latency.py](../04_plot_latency.py), with [01_collect_latency.py](../01_collect_latency.py) collecting the evidence. Output: [04-head-measured-boundaries.pdf](../figures/04-head-measured-boundaries.pdf). Tables: [report](../results/REPORT.md). No manuscript edit or promoted claim.

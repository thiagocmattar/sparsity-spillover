# Execution cohort frontiers

Question: Does the global conclusion survive separating the execution packages?

Sources: Analysis037/041 retained 14M cohort coordinates; Run063 process timing samples; Analysis042 Run064 14M decomposition and original checkpoint metadata; Runs066/068 canonical validation and Runs067/068 verified timing results. Plot style reference: Analysis041 Figure04 panel (b) and its 02_quality_latency_frontier.py script. Exact identities are in [latency source receipts](../provenance/latency-sources.json); reduced values and source attempts are in [latency.json](../data/latency.json).

Coverage: 14M only: 69 BF16 block executions and 18 FP16 native-body/head executions. Validation coordinates/counts use all 500 MiniPile documents, 338 complete 2,048-token blocks, 691,886 predictions and the declared 1,444-token excluded tail. Geometry observations use training boundaries rather than validation batches.

Method: Recompute strict Pareto dominance independently inside each declared cohort using original measured coordinates; no latency rescaling.

Caption and legend: Two explicitly labeled detail panels have their own axes because latency ranges differ. Colors, symbols, native/no-skip encoding and process-range whiskers match Figure01. Each panel has its own dotted point-estimate envelope.

Observed result: 12 BF16 frontier points, 3 FP16 head frontier points. All 3 head-cohort frontier points use native dense arithmetic. Head16 κ=0.5 sparse is point-estimate dominated by Head16 κ=0.1 dense.

Caveats: Timing sessions and checkpoint weights still differ inside the cohorts. This separation does not create a paired gate-off ablation or statistical frontier.

Source script: [03_plot_frontiers.py](../03_plot_frontiers.py), with [01_collect_latency.py](../01_collect_latency.py) collecting the evidence. Output: [02-execution-cohort-frontiers.pdf](../figures/02-execution-cohort-frontiers.pdf). Tables: [report](../results/REPORT.md). No manuscript edit or promoted claim.

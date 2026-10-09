# Existing full decomposition

Question: Where is complete-forward CUDA work spent under common cross-model labels?

Sources: Analysis037/038/041 cohort coordinates; Run063 process timing samples; Analysis042 Run064/065 decomposition and original checkpoint metadata; Runs066/068 canonical validation and Runs067/068 verified timing results. Exact identities are in [latency source receipts](../provenance/latency-sources.json); reduced values and source attempts are in [latency.json](../data/latency.json).

Coverage: Runs064/065, 6 checkpoints, κ=0/0.1, native and selected sparse modes; 12 CUPTI traces per condition/mode. Validation coordinates/counts use all 500 MiniPile documents, 338 complete 2,048-token blocks, 691,886 predictions and the declared 1,444-token excluded tail. Geometry observations use training boundaries rather than validation batches.

Method: Preserve Analysis042 original arithmetic CUDA activity means; group into nine fixed categories. Stack activity durations without rescaling. Independent complete-host geometric means appear as text.

Caption and legend: Three model panels share the activity-sum scale. Input setup gray; block normalization sky blue; QKV blue; RoPE/context orange; attention yellow; MLP up green; H/Z output pink; final norm black; vocabulary projection vermillion. Patterns reinforce colors. 14M h/z thresholds are 8×16 tiles; 31M/70M are scalar. Every head threshold is OFF.

Observed result: Vocabulary projection remains dense and prominent in the optimized profiles. The H/Z aggregate makes the fused 14M/31M and separate 70M implementations comparable without inventing internal timings.

Caveats: Bars are summed activities, not a timeline or host latency. Fusion prevents independent timing of several H/Z internals. New head experiments have no full CUPTI stage profile; do not infer it by subtracting independent boundary means.

Source script: [04_plot_latency.py](../04_plot_latency.py), with [01_collect_latency.py](../01_collect_latency.py) collecting the evidence. Output: [03-existing-full-decomposition.pdf](../figures/03-existing-full-decomposition.pdf). Tables: [report](../results/REPORT.md). No manuscript edit or promoted claim.

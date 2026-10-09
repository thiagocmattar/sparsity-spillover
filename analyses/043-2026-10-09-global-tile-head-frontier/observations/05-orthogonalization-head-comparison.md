# Orthogonalization head comparison

Question: Does adding head-input pressure increase global orthogonalization resistance relative to Ph?

Sources: Runs060/066/068 raw events.jsonl and manifests; Analysis024 original OL1 appendix and its 20 14M source conditions; Analysis021 retained geometry auditor. Manuscript draft/05-experimental-study.tex and draft/08-appendix-interventions.tex define the contextual metrics. Exact identities are in [geometry source receipts](../provenance/geometry-sources.json); reduced values and source attempts are in [geometry.json](../data/geometry.json).

Coverage: 8 retained 14M conditions from Runs060/066/068, 712 optimizer boundaries each (5,696 records). Validation coordinates/counts use all 500 MiniPile documents, 338 complete 2,048-token blocks, 691,886 predictions and the declared 1,444-token excluded tail. Geometry observations use training boundaries rather than validation batches.

Method: Recompute per-boundary geometry from raw training events and audit the original adaptive dot/norm/cap identities. Display medians and IQRs for norms/ratios, integer counts/712 for conflict. Include the initial zero-LR boundary.

Caption and legend: Six panels: manuscript removed component/task norm; removed component/pressure norm; raw/adaptive conflict; absolute pressure norm before projection; pressure/task norm before projection; post-projection pre-cap r/b. Ph blue, Ph+head vermillion; no-head circle, head8 diamond, head16 star. Error bars are step quartiles. The dotted r/b=1 line marks the cap.

Observed result: Matched head8 κ=0.1: removed/task norm 2.227→8.503%, removed/pressure norm 9.377→23.625%, raw conflict 65.03→85.39%; adaptive conflict stays approximately 99%. Cap is active only 1–3/712 boundaries in these recent conditions.

Caveats: Global pooled eligible-parameter geometry, not isolated head resistance. Tensor averaging changes 1/6→1/7 and eligible pool 65→75. No head-only pressure condition. One seed; IQRs describe training evolution rather than seed uncertainty. Norm fraction is not energy or coordinate count.

Source script: [05_plot_geometry.py](../05_plot_geometry.py), with [02_collect_geometry.py](../02_collect_geometry.py) collecting the evidence. Output: [05-orthogonalization-head-comparison.pdf](../figures/05-orthogonalization-head-comparison.pdf). Tables: [report](../results/REPORT.md). No manuscript edit or promoted claim.

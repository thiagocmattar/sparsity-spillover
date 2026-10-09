# Manuscript style geometry

Question: How do the newer tile/head conditions compare with the manuscript OL1 geometry view?

Sources: Runs060/066/068 raw events.jsonl and manifests; Analysis024 original OL1 appendix and its 20 14M source conditions; Analysis021 retained geometry auditor. Manuscript draft/05-experimental-study.tex and draft/08-appendix-interventions.tex define the contextual metrics. Exact identities are in [geometry source receipts](../provenance/geometry-sources.json); reduced values and source attempts are in [geometry.json](../data/geometry.json).

Coverage: 20 original manuscript 14M conditions plus 8 retained tile/head conditions: 19,936 optimizer-boundary records, 712 each. Validation coordinates/counts use all 500 MiniPile documents, 338 complete 2,048-token blocks, 691,886 predictions and the declared 1,444-token excluded tail. Geometry observations use training boundaries rather than validation batches.

Method: Audit raw logs and reproduce original manuscript medians. Left: condition median/IQR of 100ρopp on categorical κ positions. Right: unsmoothed pointwise median r/b over available thresholds in each family, with faint individual traces.

Caption and legend: Dashed context families are T4/Ph, T4/Pall, T7/Ph, T7/Pall (5 thresholds each); solid newer families are T2 tile/Ph, head8/Ph, head8/Ph+head, head16/Ph+head (2 each). Both metric axes are logarithmic. Head16 varies head κ at body κ=0; head8 varies both. r/b=1 marks the norm cap.

Observed result: New head-pressure recipes have larger removed components than the matched Ph head8 recipes, but rarely saturate the cap. Original T7/Pall cap activity is 3,453/3,560 boundaries (96.99%) across five κ conditions.

Caveats: Family medians pool different numbers/choices of thresholds and are descriptive; head16/body-threshold axes are not a matched common sweep. Manuscript r/b is after projection, not the pre-projection norm ratio. Global geometry and one-seed limitations remain; manuscript text is unchanged.

Source script: [05_plot_geometry.py](../05_plot_geometry.py), with [02_collect_geometry.py](../02_collect_geometry.py) collecting the evidence. Output: [06-manuscript-style-geometry.pdf](../figures/06-manuscript-style-geometry.pdf). Tables: [report](../results/REPORT.md). No manuscript edit or promoted claim.

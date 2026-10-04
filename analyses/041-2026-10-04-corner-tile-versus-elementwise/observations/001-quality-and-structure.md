# Quality cost of causal corner tiles

Question: does the approved 8x16 corner gate create empty tiles while preserving
the quality of matched elementwise T2/Ph at kappa 0/.1?

Sources: verified Run060 attempts `001-20261004-125905-e5a32225` and
`002-20261004-125906-45dd6a88`, compared with the two Ph endpoints of Run058.
The exact input paths, bytes and SHA256 are in `../data/source-identity.json`.
The analysis checks identical initial parameters, cache/data schedule, seeds,
architecture, 712-update budget, optimizer/FP16 recipe and h-only OL1 pressure.
Only the complete h/z gate operator changes; no a/m gate or pressure is present.

Coverage: all 500 MiniPile validation documents, 338 complete 2048-token blocks,
692224 input tokens and 691886 predictions; excluded tail 1444 tokens. One seed.
Loss is pooled prediction-weighted NLL. Perplexity is exp(loss). Empty-tile
percentages pool integer counts across all batches and six layers before division.
Each 8x16 tile stays within one sequence and has 128 exactly zero values if empty.

| kappa | Scalar loss | Tile loss | Change | Perplexity change | H empty tiles: scalar -> tile | Z empty tiles: scalar -> tile |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.127033 | 5.553770 | +0.426737 | +53.225% | 3.0442% -> 57.5056% | 0.0153% -> 44.9262% |
| .1 | 5.152346 | 5.596746 | +0.444400 | +55.955% | 43.6962% -> 84.3082% | 20.7041% -> 70.9567% |

Result: both tile endpoints have substantially more fully empty tiles and worse
validation quality. The observed penalty does not support preservation of quality
under this fixed recipe/budget. It does not show total model collapse: learning
and surviving per-layer branch activity are documented in Observation 002.
No numerical acceptable-loss bound was chosen before the run.

Caption: Final complete-validation quality and empty 8x16 activation tiles for
Pythia-14M T2/Ph at kappa 0 and .1. Solid blue bars are retained Run058 scalar
gates; hatched orange bars are Run060 corner gates. The left panel shows NLL,
and the middle/right panels show pooled empty-tile percentages in h/z. All
axes start at zero. Labels are rounded; unrounded values and counts are retained
in `../results/summary.json` and the wide comparison table.

Caveats: one seed, two thresholds, one model size and one fixed training budget.
The tile operator also retains signed/subthreshold noncorner values and rejects
equality, so the comparison does not isolate geometry. Structured emptiness does
not establish runtime gain or mechanistic redistribution across heads/layers.
Inference timing and tile pressure remain deferred. No manuscript claim changed.

Generator: `../01_analyze.py`. Figure: `../figures/01-quality-and-structure.pdf`.
The final PDF was rendered with Poppler and visually inspected.

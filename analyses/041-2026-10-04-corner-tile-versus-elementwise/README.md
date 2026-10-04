# 14M T2/Ph — causal corner tiles versus elementwise gates

Status: complete. Run060 passed remote and local scientific verification;
the archive and all 144 transferred members passed SHA256/byte checks.
All three final PDFs were rendered and visually inspected. The owned Pod was
terminated and provider inventory confirms its absence.

Both tile conditions learn and retain activity in every h/z layer, but their
final losses rise .426737/.444400 versus matched scalar controls (perplexity
+53.225%/+55.955%). Empty 8x16 tiles increase substantially, while total h/z
element sparsity decreases. This supports learning viability under the restriction
with a substantial quality cost in this recipe; it does not establish preservation
of quality, redistribution or speedup.

See the [wide comparison table](results/comparison.md),
[layerwise counts](results/per-layer.csv), and [observations](observations/INDEX.md).
The run's [compute closeout](../../runs/060-2026-10-04-pythia14m-temporal-corner-tile/results/compute-closeout.json)
records transfer, teardown, estimated cost and partial posted billing.

Compare the two Run060 corner-gate endpoints with matched retained Run058 Ph
endpoints at kappa 0/.1. The 14M architecture, untrained initial bytes, all training
data/cache identity, realized schedule, seeds, 712-update budget, FP16 recipe,
optimizer, checkpoints and h-only orthogonal L1 pressure are identical.
The gate at h/z changes from scalar equality-survives thresholding to an inclusive
single-corner 8x16 operator preserving all values in admitted tiles.

Primary outputs: full-validation loss/perplexity and paired changes; exact-zero
and empty 8x16-tile percentages beside each other at a,m,h,z; per-layer CSV and
pooled integer denominators. Full tile diagnostics at updates 0/32/128/712 and
full loss validation at 0/1/32/128/256/512/712 describe adaptation in Run060.
Additional 1x16/16x16/32x16 counters, corner certificates, per-position/per-feature
counts and logical-product opportunities remain in the source artifacts.

Three publication PDF figures show final quality/structure, learning and tile
evolution, and all four sites' scalar versus tile sparsity/occupancy. Their matching
observations record complete coverage, methods, captions and source scripts.

Coverage: all 500 MiniPile validation documents, 338 complete 2048-token blocks,
692224 inputs and 691886 predictions, with 1444 excluded tail tokens.
Pool integer counts before division. Report actual quality deltas: no numerical
acceptable-loss budget was specified. One seed and a fixed horizon give a
descriptive complete-operator comparison; equality and signed-value retention
also differ, so it does not isolate geometry alone or prove redistribution.

Source runs:
[Run058](../../runs/058-2026-10-03-pythia14m-t2-p2-versus-ph/README.md),
[Run060](../../runs/060-2026-10-04-pythia14m-temporal-corner-tile/README.md).
Regenerate from the retained, verified sources:

```powershell
.venv/Scripts/python.exe analyses/041-2026-10-04-corner-tile-versus-elementwise/01_analyze.py
```

`data/source-identity.json` records input bytes/SHA256; `results/summary.json`
retains unrounded values, paired deltas, counts and curves. Tables and PDFs
derive from those same source artifacts. No manuscript edit is requested.

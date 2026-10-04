# 14M T2/Ph and T7/Ph — causal corner tiles versus elementwise gates

Status: complete. Run060 passed remote and local scientific verification;
the archive and all 144 transferred members passed SHA256/byte checks.
The initial three PDFs were rendered and visually inspected. The owned Pod was
terminated and provider inventory confirms its absence.

Both tile conditions learn and retain activity in every h/z layer, but their
final losses rise .426737/.444400 versus matched scalar controls (perplexity
+53.225%/+55.955%). Empty 8x16 tiles increase substantially, while total h/z
element sparsity decreases. This supports learning viability under the restriction
with a substantial quality cost in this recipe; those training diagnostics do
not establish preservation of quality, redistribution or speedup. The separately
verified Run061 latency evidence is now included below.

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

## Quality-latency frontier with the two tile endpoints

The user's follow-up requested the manuscript's quality-on-X, latency-on-Y
comparison with the two new tile points, saved here. The current source asset
02-14m-70m-latency-quality.pdf comes from Analysis038: retain all 36 historical
Base, T2/Ph and T7/Ph execution points across 14M/31M/70M, append the two 14M corner
endpoints, and draw the observed frontier over this 38-point cohort. A linear 14M detail
also displays the two matched Run061 scalar references, excluded from the
frontier cohort. The other panel preserves the source's logarithmic axes.

- [Figure04](figures/04-quality-latency-frontier-with-corner-tiles.pdf).
- [Observation004: caption, coverage and caveats](observations/004-quality-latency-frontier.md).
- [Exact coordinates, Pareto set and source hashes](data/quality-latency-frontier.json).
- [Coordinate table](results/quality-latency-frontier.md).

Tile coordinates use the canonical FP16 loss above and the verified BF16
Run061 sparse host latency: (5.553770, .540220 ms) at κ=0 and
(5.596746, .508340 ms) at κ=.1. Neither extends the observed historical frontier.
Scalar T2/Ph κ=.5 previously measured (5.536255, .506323 ms) dominates both
by point estimates. Its .002016 ms gap against tile κ=.1 is small and comes from
different sessions/kernel variants; the comparison is descriptive.
The controlled Run061 gains of 10.990%/10.067% over matched scalar κ=0/.1
remain valid. Whiskers span three process means, not confidence intervals.
No normalization, new benchmark, cloud spend, manuscript edit or finding
promotion is part of this figure. The original three PDFs are unchanged.

Regenerate Figure04 with
.venv/Scripts/python.exe -X utf8 analyses/041-2026-10-04-corner-tile-versus-elementwise/02_quality_latency_frontier.py.

## Approved T7/Ph extension: in progress

[Run062](../../runs/062-2026-10-04-pythia14m-t7-corner-tile-ph/README.md) runs
the same corner operator at all seven T7 sites, kappa0/.1, keeping pressure at h.
`03_t7_training.py` will compare canonical FP16 quality and pooled seven-site
zero/empty8x16 counts against the retained Run032 scalar controls. The scalar
counts receive a separate complete FP16 post-hoc replay, preserving the original
canonical quality field. `04_t7_frontier.py` will append two qualified tile
endpoints and two auxiliary matched scalar references while retaining every
previous coordinate and PDF. No new results are asserted before verification.

The training extension is now verified: both712-update conditions learn and
retain activity at all42 final gate ports. Compared with scalar T7/Ph, losses
rise.501244/.597707 and perplexity65.077%/81.795%; empty8x16 tiles increase
substantially. See the [seven-site table](results/t7-comparison.md),
[Figure05 and observation005](observations/005-t7-site-sparsity-and-tiles.md),
and [Figure06 and observation006](observations/006-t7-learning-and-tile-structure.md).
RTX5090 numerical qualification and latency remain in progress; no new T7
optimized latency or frontier membership is asserted yet.

## T7/Ph follow-up: complete

Run062's six fresh RTX5090 processes passed full own-eager BF16 qualification;
all12168 model/graph/block comparisons have zero maximum logit error. All80
returned inference files are verified locally, and both owned training/inference
Pods are deleted. The independent provider guards are stopped. All checkpoints,
raw timings, diagnostics and gradient measurements remain local.

T7 tile latency is.564432/.516416ms versus matched scalar.635143/.556401ms:
reductions11.133%/7.187% (1.125277x/1.077430x). The original quality penalties
remain.501244/.597707. Neither tile endpoint extends the observed frontier:
previous orange T7/Ph kappa.1 dominates tile kappa0, and orange kappa.5 dominates
tile kappa.1. This is a descriptive cross-session comparison; the controlled
same-threshold runtime gains are measured separately.

See [Figure07, caption and caveats](observations/007-t7-quality-latency-frontier.md),
[all40 coordinates and the15-point frontier](data/t7-quality-latency-frontier.json),
and [runtime table](../../runs/062-2026-10-04-pythia14m-t7-corner-tile-ph/latency/results/tables.md).
Earlier six PDFs are preserved byte-for-byte. Regenerate the verified extension
with scripts03_t7_training.py and04_t7_frontier.py. No manuscript edit or finding
promotion is included.

# Verification record - second figure revision, 8 September 2026

## Scope and outstanding measurement

Eight figures were revised, titled and reviewed. The 54 trained conditions,
190 clipping evaluations and 30-checkpoint runtime cohort are unchanged.
The overview now has ten separately defined frontier series. Its numerical
membership is stored and tabulated. The activation grid now uses seven
checkpoints, five common sites and kappa=0,.05,.5 rows.

**The requested finer activation bins are not complete.** Stored statistics
only define the four bins {0}, (0,.001], (.001,.01] and >.01. No finer-bin
counts were inferred or fabricated. The separate ACTIVATION-DIAGNOSTIC.md
proposal awaits design confirmation under AGENTS.md; no new diagnostic code,
numbered run or model execution has been started. The checklist leaves this
item open. The restored distribution figure is explicitly a coarse view.

## Numerical checks

```powershell
.venv/Scripts/python.exe -m pytest -p no:cacheprovider analyses/018-2026-09-08-results-materials/test_evidence.py tests/test_ceilings.py tests/test_metrics.py -q
```

**39 passed in 1.65 seconds.** Checks cover complete cohorts and validation,
integer logical counts, count-first activation pooling, magnitude-bin
partitioning, source hashes, same-size identities, exact matched differences,
ceiling units and clipping-site normalization, scope of each overview frontier,
all three case-study thresholds, the 30-checkpoint runtime reduction and output
inventory hashes. The saved bundle still identifies 230 direct source files.
No shared scientific code changed. The full bootstrap launch suite is not
applicable until a new diagnostic is implemented for launch.

## Visual and TeX checks

All eight final PDFs were rendered at 115.2 dpi in color and grayscale and
visually inspected. Each is one page and exactly 396 points (5.5 inches) wide.
Every extracted text span lies inside its page bounds; ordinary text is at
least 8 pt, with conventionally smaller math scripts. Every figure has a title.

The overview uses separate trained-family and control-clipping frontiers,
without a pooled envelope. Eight high-loss clipping points lie beyond its
5.04?6.15 loss window; their complete range is visible in the scale figure
and the clipping appendix. The scale ceiling lines use both color and differing
line patterns. Operation fills are solid as requested; their stack order and
legend identify operations. The activation grid uses matched axes, one shared
legend and the original measured bins. The clipping legend states source
checkpoint counts. The kernel plot now includes the full recipe legend.

The optional results.tex and both compact generated tables compile in the
5.5-inch-text-width article wrapper. The final reference-resolved compile has
**no LaTeX warnings, unresolved references, overfull or underfull boxes**.
All ten preview pages were rendered and inspected as a syntax/caption/layout
proof; the wrapper's float pagination is not a final conference page budget.
The manuscript itself was not edited or rebuilt.

The final scale-line pattern change was rendered and checked separately.
Temporary renders and wrapper files stay under ignored tmp/analysis018-revision2.
Publication artwork remains PDF only. artifact_inventory.json records all
PDF/table hashes and sizes; the analysis .gitattributes preserves those bytes.

## Closeout

The coherent figure revision, updated captions/prose/tables and pending
measurement design are committed together. The open finer-bin item is not
represented as finished. Original runs, prior analyses, model weights, caches,
credentials, temporary outputs and unrelated work remain outside this commit.

# Verification record - 8 September 2026

## Numerical and source verification

The final local build completed successfully:

```text
59 trained conditions
190 raw clipping evaluations
30 matched contrasts (25 in Figure 02; 5 historical pressure-placement pairs in tables)
7 single-page PDF figures
250 directly hashed source files
```

The 35/12/12 trained counts reconcile with the admitted run cohorts; clipping
grids reconcile at 150/20/20 points. Raw logical numerators and denominators,
analytic per-sequence ceilings, full-validation coverage, pooled activation
counts/bands/RMS, and same-size initialization/order/cache identities pass
the checks in `evidence.py`. Larger-scale raw clipping values also match
the retained Analysis 011 reduction. Source manifests establish random
initialization, seed 1234, 712 boundaries, and 1,493,172,224 input tokens.

Final focused and affected mathematical tests:

```powershell
.venv/Scripts/python.exe -m pytest -p no:cacheprovider analyses/018-2026-09-08-results-materials/test_evidence.py tests/test_ceilings.py tests/test_metrics.py -q
```

Result: **35 passed**, 1.50 seconds on the final check. This covers frontier
dominance and ties; disjoint band construction; complete scientific grids;
rejection of corrupted logical counts; natural-zero normalization and undefined
A0 ratios; operation weighting; matched difference arithmetic/identities;
common-site diagnostic pooling; historical/corrected pressure identity;
the scoped cross-scale ordering claims; and saved-source/output hashes.
No new model execution or scientific shared-code change was made. The full
bootstrap launch suite was not run: there is no experiment launch in this task.

## Visual and LaTeX verification

The current manuscript PDF was extracted and its argument/setup read. Its
setup page was also rendered and inspected. All seven final figure PDFs were
rendered at 108 dpi and visually inspected. Iteration removed overlapping
dose labels in Figure 03 and separated measured stack values from ceiling
markers in Figure 04. Figure 01 now identifies both frontier pools explicitly
in its legend. All PDFs contain selectable text; each has exactly one page.

The optional `results.tex` fragment and both generated compact table fragments
were compiled in a temporary article wrapper. After the reference-resolution
pass, the five-page preview has **no LaTeX warnings, unresolved references,
overfull boxes, or underfull boxes**. Every preview page was rendered and
inspected. This checks insertion syntax and local appearance, not a final
conference page budget or float layout. Figures 02/05 are dense, wide assets;
use full-width placement and assess lettering at the eventual submission size.
The manuscript itself was neither edited nor rebuilt.

The figures are the final publication artifacts. PNG renders, LaTeX wrapper,
compiled preview, logs and auxiliary files remain under ignored `tmp/` and
are not part of the committed materials. Figure 07's bytes match Run 029's
final r03 figure exactly. `artifact_inventory.json` records each generated
PDF and table's SHA-256 and size; the tests verify those entries.
The analysis-local `.gitattributes` preserves these hash-inventoried bytes
across Git line-ending conversion on different platforms.

## Scope and closeout

The commit is limited to this analysis folder and the Analysis 018 pointer / next
analysis number in `research/INDEX.md`. Existing `.gitignore`, run-log changes,
and historical archive/deep-path state are outside this task. No credentials,
data cache, model checkpoint, temporary render, active-run artifact, prior
analysis modification, or manuscript source is staged. No push is performed.

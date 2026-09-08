# V2 working draft: results incorporated

The 8 September 2026 revision implements the approved five-figure plan.
The [argument map](results-argument-map.md) was written before the prose.
The results proceed from quality-sparsity regimes to paired intervention
effects, activation reshaping, cross-scale recipe transfer and measured
kernel acceleration. The introduction now includes a short result preview.

## Reading and source files

- [main.pdf](main.pdf): 24-page reading copy; main text on pages 1-10,
  references on page 11, appendices on pages 12-24.
- [experimental-study.tex](experimental-study.tex): setup and comparisons.
- [training-results.tex](training-results.tex): four training-results subsections,
  four figures, one compact cross-scale contrast table.
- [kernel-autoresearch.tex](kernel-autoresearch.tex): selected Figure 07 and
  the matching 30-checkpoint results; the [technical companion](kernel-implementation.md)
  clearly distinguishes its older 35-checkpoint audit.
- [results-appendix.tex](results-appendix.tex): realized protocol, all trained
  endpoints and contrasts, density mass/tails, operation accounting, full
  clipping plots, kernel qualification and ablations.
- [figures/SOURCES.json](figures/SOURCES.json): original figure paths and hashes.
  The five selected figures and five appendix assets are byte-identical copies.
- [tables/](tables/): seven generated tables with source hashes. The formatter
  belongs to Analysis 018: `03_manuscript_tables.py`.
- [supplementary-data/README.md](supplementary-data/README.md): 14 copied
  evidence files (13.68 MB), including all 540 clipping evaluations and seven
  signed histograms; units, fields, source paths and hashes are explicit.

General methods and formal definitions remain in [methodology.tex](methodology.tex)
and [methodology-appendix.tex](methodology-appendix.tex); Pythia-specific sites
and reach are in [experimental-appendix.tex](experimental-appendix.tex).
The original architecture/ladder is now an appendix figure. Related work
and bibliography are unchanged. The user's supplementary.md is unchanged.

## Reproduce and verify

From this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The complete build resolves all 15 citation keys and all references, without
box warnings or errors. All 24 rendered pages were inspected, with additional
full-size checks of dense figures and tables. All fonts are embedded; none
are Type 3. The seven tables reconcile 54 integer-count totals, 44 paired
differences and 14 pooled histograms. No scientific measurement was rerun.
This is the existing 5.5-inch reading wrapper, not an ICLR submission-template
fit claim.

## Provenance and earlier work

The [plan](results-plan.md) retains the pre-writing figure/caption plan and
its implementation status. Analysis 018 [O012](../../analyses/018-2026-09-08-results-materials/observations/O012-manuscript-results.md)
links the full source/PDF snapshot and verification. The draft remains
Git-ignored/local-only; the snapshot is version-controlled under the analysis.
The tracked parent [README](../README.md) records the current adoption.

Earlier reviews remain in [reviews/](reviews/); the previous paper and working
materials are in [archive/](archive/README.md). Their page counts and deferred
results statements describe those earlier revisions. The prior kernel source
snapshots remain under Run 029. Positioning and methods notes retain their
original literature and operational provenance.

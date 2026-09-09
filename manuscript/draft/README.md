# Manuscript draft: ICLR 2027

This directory is tracked for collaborative editing. Edit the section TeX files
and bibliography, rebuild `main.pdf`, and commit the intended source and PDF
changes together. The official ICLR styles remain unchanged in anonymous review
mode. LaTeX intermediates, backup archives and raster previews remain ignored.

## Current reading draft

The 9 September existing-results rewrite follows [task.md](task.md). Its two
central comparisons concern the conditional effect of pressure and the different
rankings obtained from local zeros, operation-weighted counts and execution.
The user approved **architectural reach**, `R_arch`, as the name and symbol for
the existing analytic quantity. Stored scientific identifiers remain unchanged.

[main.pdf](main.pdf) has 30 pages: main text on pages 1-14, the AI use statement
and references starting on page 14, and appendices on pages 16-30. It contains
six main figures, five appendix figures and fourteen numbered tables. It is a
reading draft, not a page-limit-compliant submission: shortening to the official
nine-page main-text limit and the author's framing/figure-budget decisions remain
open. The authors should also check the completeness of [ai-use.tex](ai-use.tex).

The [revision record](revision/README.md) contains the paragraph map, batch log,
claim ledger, numerical checks, author decisions and readiness assessment.
[Analysis 019](../../analyses/019-2026-09-09-manuscript-rewrite-audit/README.md)
owns the existing-record audits and revised figures/tables. No experiment was
launched and the original run records were not changed.

## Sources and evidence

- `abstract.tex`, `introduction.tex`, `related-work.tex`, `methodology.tex`:
  framing, related evidence and definitions.
- `experimental-study.tex`, `training-results.tex`, `kernel-autoresearch.tex`:
  matched comparisons, quality budgets, distributions, scale and execution.
- `conclusion.tex`, `ai-use.tex`: bounded conclusions and assistance disclosure.
- `methodology-appendix.tex`, `experimental-appendix.tex`, `results-appendix.tex`:
  exact algorithms, all 54 endpoints, 29 paired contrasts, 15 scale contrasts,
  distribution diagnostics, all 540 clipping evaluations and kernel records.
- [figures/SOURCES.json](figures/SOURCES.json) and
  [tables/SOURCES.json](tables/SOURCES.json): origins and hashes for copied assets.
  Older inactive PDFs are retained with their original provenance. The twelve
  table fragments include absolute latency, regression sensitivity and h/z rows;
  site and training-setting tables are inline TeX.
- [supplementary-data/README.md](supplementary-data/README.md): original numerical
  records, seven signed histograms, protocol and three revision audit summaries.
  This directory is not a standalone checkpoint or executable-kernel release.

The paper aliases raw `R_model` to model-wide sparsity, `R_block` to block-only
sparsity, and `R_model_max*` to architectural reach. Dividing model-wide sparsity
by the common A7 reach equals block-only sparsity for this declared Pythia graph;
this is not a change to every historical `U_arch` value.

## Build and verification

From this directory, with LaTeX and BibTeX installed:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

All active figure PDFs, table inputs, bibliography and official styles are
included. The revision's [verification](revision/verification.json) records the
clean-directory build, page comparison, figure/table regeneration, scientific
checks and source-copy hashes. The final thirty pages were visually inspected;
minor underfull vertical boxes retain the official page-stretching behavior.

## Retained history

The first tracked draft is Git revision
`1e14471bd4763dbfef84a162d4429021d2f7e2e6`, whose 27-page PDF matches the
runbook's frozen source. The [ICLR conversion record](reviews/2026-09-09-iclr2027-template/README.md)
describes that earlier layout. Older [reviews](reviews/),
[archive](archive/README.md), [argument map](results-argument-map.md) and
[figure plan](results-plan.md) document their own historical versions; claims
there that the draft was local-only or figures were unchanged describe those
versions, not the current revision.

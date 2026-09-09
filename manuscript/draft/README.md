# Manuscript draft: ICLR 2027

This directory is tracked for collaborative editing. Edit the section TeX files
and bibliography, rebuild `main.pdf`, and commit the intended source and PDF
changes together. The official ICLR styles remain unchanged in anonymous review
mode. LaTeX intermediates, backup archives and raster previews remain ignored.

## Current reading draft

The 9 September intervention-centered revision follows [task.md](task.md).
It explains fixed thresholds, OL1's conditional cap geometry and architectural
reach, then examines pressure interactions, ReLU followed by clipping,
cross-size recurrence, activation distributions and measured execution.
The user approved **architectural reach**, `R_arch`, as the name and symbol for
the existing analytic quantity. Stored scientific identifiers remain unchanged.

The author approved the empirical-design framing and the updated task.
[main.pdf](main.pdf) now has 33 pages: main text on pages 1-9, the AI use statement
and references on pages 10-11, and appendices on pages 12-33. Its six main figures,
nine appendix figures and fourteen numbered tables preserve the complete results.
The main text fits the nine-page budget without changing template fonts or margins.
The cross-size comparison is restored to the main text on page 7, with ReLU
clipping and a full-range appendix view. The runtime display uses checkpoint-level
quality and latency; retrospective quality budgets remain in the appendix.
The full recipe diagram and distribution grid also remain there. Authors should check the
completeness of [ai-use.tex](ai-use.tex) before submission.

The [current revision record](revision-v2/readiness.md) contains the paragraph map,
batch log, claim ledger, numerical checks, evidence limits and reproduction commands.
[Analysis 020](../../analyses/020-2026-09-09-intervention-revision/README.md)
owns the new log/reach/clipping reductions, four vector PDFs and generated dynamics table.
[Analysis 019](../../analyses/019-2026-09-09-manuscript-rewrite-audit/README.md)
owns the preceding existing-record audits and figures/tables. No experiment was
launched and the original run records were not changed.

## Sources and evidence

- `abstract.tex`, `introduction.tex`, `related-work.tex`, `methodology.tex`:
  framing, related evidence and definitions.
- `experimental-study.tex`, `training-results.tex`, `kernel-autoresearch.tex`:
  matched comparisons, quality budgets, distributions, scale and execution.
- `conclusion.tex`, `ai-use.tex`: bounded conclusions and assistance disclosure.
- `methodology-appendix.tex`, `experimental-appendix.tex`, `results-appendix.tex`,
  `training-dynamics-appendix.tex`:
  exact algorithms, all 54 endpoints, 29 paired contrasts, 15 scale contrasts,
  distribution diagnostics, all 540 clipping evaluations and kernel records.
- [figures/SOURCES.json](figures/SOURCES.json) and
  [tables/SOURCES.json](tables/SOURCES.json): origins and hashes for copied assets.
  Older inactive PDFs are retained with their original provenance. The twelve
  original table fragments include absolute latency, regression sensitivity and h/z rows;
  the new thirteenth fragment gives late training dynamics;
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
included. The revision's [verification](revision-v2/verification.json) records the
clean-directory build, page comparison, figure/table regeneration, scientific
checks and source-copy hashes. The final 33 pages were visually inspected;
minor underfull vertical boxes retain the official page-stretching behavior.

## Retained history

The preceding [existing-results rewrite](revision/README.md) records its own
28-page version. Its verification and author decisions remain historical records.

The first tracked draft is Git revision
`1e14471bd4763dbfef84a162d4429021d2f7e2e6`, whose 27-page PDF matches the
runbook's frozen source. The [ICLR conversion record](reviews/2026-09-09-iclr2027-template/README.md)
describes that earlier layout. Older [reviews](reviews/),
[archive](archive/README.md), [argument map](results-argument-map.md) and
[figure plan](results-plan.md) document their own historical versions; claims
there that the draft was local-only or figures were unchanged describe those
versions, not the current revision.

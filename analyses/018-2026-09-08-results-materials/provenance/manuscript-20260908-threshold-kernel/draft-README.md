# Manuscript draft: reviewed results and discussion

The 8 September 2026 revision preserves the user's first two introduction
paragraphs exactly and develops the existing argument: quality-sparsity
trade-offs, paired effects, activation reshaping, transfer across model sizes,
then measured execution. A concise abstract and discussion complete the paper.
The contribution is supported by the structure across interventions, parameter
sweeps and complementary measurements. Exact seed details remain in the protocol.

The latest revision motivates fixed symmetric thresholds using the Q-Sparse
and Spark selection rules, then develops the kernel argument from compatibility
through agent-assisted specialization, early gains, cohort association and
attention-skipping limits. See the [argument and review record](reviews/2026-09-08-kernel-argument/revision-log.md).

## Reading copy and sources

- [main.pdf](main.pdf): 27-page reading copy; main text pages 1-12,
  references page 13, appendices pages 14-27. Template fitting is not assessed.
- [abstract.tex](abstract.tex), [introduction.tex](introduction.tex),
  [related-work.tex](related-work.tex), [methodology.tex](methodology.tex):
  framing, literature and definitions.
- [experimental-study.tex](experimental-study.tex),
  [training-results.tex](training-results.tex),
  [kernel-autoresearch.tex](kernel-autoresearch.tex): matched protocol,
  five main figures, compact recipe key and scale-contrast table.
- [conclusion.tex](conclusion.tex): scientific contribution, agreement across
  evidence, execution implications and study scope.
- [methodology-appendix.tex](methodology-appendix.tex),
  [experimental-appendix.tex](experimental-appendix.tex),
  [results-appendix.tex](results-appendix.tex): definitions, complete results,
  diagnostics, clipping, kernel qualification and data provenance.
- [figures/SOURCES.json](figures/SOURCES.json): eleven unchanged figure copies.
  Ten are embedded; the all-variant overview v2 remains an unselected alternative.
- [tables/](tables/): seven reproducible analysis-owned tables. Numerical rows
  are unchanged; terminology uses threshold, activation sparsity and U_arch.
- [supplementary-data/README.md](supplementary-data/README.md): fourteen unchanged
  measurement copies plus a separately sourced [protocol](supplementary-data/protocol.json).
  Includes all 540 clipping evaluations, seven signed histograms, and source hashes.

The paper uses activation sparsity, model-wide sparsity, sparsity ceiling,
thresholding and nonlinearities consistently. Operational data keys retain
their original names. Figure artwork and numerical measurements are unchanged.

## Review and verification

The [latest verification](reviews/2026-09-08-kernel-argument/verification.json)
records the clean 27-page build, unchanged figures, tables and bibliography,
preserved introduction, primary-source checks and independent scientific and
visual review. New evidence wording separates the four-checkpoint historical
replay from the final implementation's 30-checkpoint manuscript cohort.

The [revision log](reviews/2026-09-08-polish/revision-log.md) records the
incremental edits and resolves technical-reader, scientific-reviewer and
literature-audit feedback. The [verification](reviews/2026-09-08-polish/verification.json)
records exact protected-paragraph preservation, all figure/data/protocol hashes,
53 resolved labels, 15 verified references, unchanged numerical table rows and
all 26 visually inspected pages. The build has no LaTeX or box warnings; all
fonts are embedded and none are Type 3.

From this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Provenance

The draft remains Git-ignored/local-only. Analysis 018
[O015](../../analyses/018-2026-09-08-results-materials/observations/O015-threshold-kernel-argument.md)
records the latest revision and its new source/PDF snapshot. The earlier
[O014 polish](../../analyses/018-2026-09-08-results-materials/observations/O014-manuscript-polish.md),
[argument map](results-argument-map.md), [figure plan](results-plan.md),
[reviews](reviews/) and [archive](archive/README.md) preserve prior decisions.
The original 24-page results snapshot remains unchanged. No experiment was launched.

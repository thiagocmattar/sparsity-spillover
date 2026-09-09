# Manuscript draft: ICLR 2027 format

## Collaborative editing

As of 9 September 2026, this directory is tracked in Git for collaborative
editing. Edit the section `.tex` files and `references.bib`, build from this
directory using the commands below, and commit the updated sources with
`main.pdf`. All required template files, figure PDFs and table fragments are
included, so the current paper builds from a fresh checkout with a LaTeX
installation and BibTeX. Supporting measurements, provenance, review notes and
archived manuscript sources/PDFs are also tracked.

LaTeX build intermediates, backup archives and raster QA previews remain
ignored. The small official ICLR template ZIP is retained as a provenance
exception. Source/evidence line endings are preserved to retain recorded hashes;
generated table fragments retain their existing LF convention. Historical
references to a local-only draft describe the policy before this change.

The initial tracked version was built from a separate checkout of the staged
draft files. All 27 pages matched the current PDF in extracted text and rendered
pixels. All 11 figure-copy hashes and 14 supporting-measurement hashes matched,
and checkout preserved every staged draft file byte-for-byte. The existing
minor underfull vertical box on page 3 remains; no references or inputs are
missing and no overfull boxes occur.

## Current format

The 9 September 2026 format conversion uses the official ICLR 2027 styles in
anonymous review mode, including the review header and line numbers. The title,
section headings, paragraph spacing, captions and bibliography now follow the
template. Section sources, bibliography data, figures and numerical tables are
unchanged. See the [conversion record](reviews/2026-09-09-iclr2027-template/README.md).

The 8 September 2026 revision preserves the user's first two introduction
paragraphs exactly and develops the existing argument: quality-sparsity
trade-offs, paired effects, activation reshaping, transfer across model sizes,
then measured execution. A concise abstract and discussion complete the paper.
The contribution is supported by the structure across interventions, parameter
sweeps and complementary measurements. Exact seed details remain in the protocol.

The 8 September argument revision motivates fixed symmetric thresholds using the Q-Sparse
and Spark selection rules, then develops the kernel argument from compatibility
through agent-assisted specialization, early gains, cohort association and
attention-skipping limits. See the [argument and review record](reviews/2026-09-08-kernel-argument/revision-log.md).

## Reading copy and sources

- [main.pdf](main.pdf): 27-page ICLR 2027 draft; main text pages 1-13,
  references pages 13-14, appendices pages 15-27. Before submission, shorten the
  main text to nine pages and add the required author-reviewed AI use statement.
- [abstract.tex](abstract.tex), [introduction.tex](introduction.tex),
  [related-work.tex](related-work.tex), [methodology.tex](methodology.tex):
  framing, literature and definitions.
- [experimental-study.tex](experimental-study.tex),
  [training-results.tex](training-results.tex),
  [kernel-autoresearch.tex](kernel-autoresearch.tex): matched protocol,
  six main figures, including the architecture/recipe diagram, and the scale-contrast table.
- [conclusion.tex](conclusion.tex): scientific contribution, agreement across
  evidence, execution implications and study scope.
- [methodology-appendix.tex](methodology-appendix.tex),
  [experimental-appendix.tex](experimental-appendix.tex),
  [results-appendix.tex](results-appendix.tex): definitions, complete results,
  diagnostics, clipping, kernel qualification and data provenance.
- [figures/SOURCES.json](figures/SOURCES.json): eleven unchanged figure copies.
  Ten are embedded. Figure 1 is the combined architecture map and sparsification
  ladder. Figure 2 uses the all-variant overview v2, with all 30 trained
  checkpoints and a matching caption; the original overview is retained.
- [tables/](tables/): seven reproducible analysis-owned tables. Numerical rows
  are unchanged; terminology uses threshold, activation sparsity and U_arch.
- [supplementary-data/README.md](supplementary-data/README.md): fourteen unchanged
  measurement copies plus a separately sourced [protocol](supplementary-data/protocol.json).
  Includes all 540 clipping evaluations, seven signed histograms, and source hashes.

The paper uses activation sparsity, model-wide sparsity, sparsity ceiling,
thresholding and nonlinearities consistently. Operational data keys retain
their original names. Figure artwork and numerical measurements are unchanged.

## Review and verification

The ICLR conversion was compiled with the official, byte-identical conference
and bibliography styles and their bundled `natbib.sty` and `fancyhdr.sty`.
All 27 pages were visually checked. All 52 labels and 15 citations resolve;
all fonts are embedded, no Type 3 fonts or overfull boxes occur, and no text
falls outside a page. A minor underfull vertical box on page 3 (badness 1383)
was visually checked; the official style's page stretching remains unchanged.
The [verification](reviews/2026-09-09-iclr2027-template/verification.json) records
the source ZIP and style hashes, final PDF hash and remaining submission work.

On 9 September, the user replaced the recipe table with the existing combined
architecture/ladder figure, relocated from the appendix to the experimental setup.
The byte-identical source is `../artifacts/pythia-architecture-sparsification-ladder.pdf`;
its companion Markdown and TeX record the two panels and analytic ceiling provenance.
The figure is now Figure 1 on page 5; its caption retains pressure targets and
model-size coverage, and the appendix points back to it. The 26-page build has
no LaTeX or box warnings; the affected layout was visually checked.

Earlier on 9 September, the user selected `figures/01-v2-14m-overview.pdf` for
the quality--sparsity overview, now Figure 2.
Its caption and source comments now follow Analysis 018
[O013](../../analyses/018-2026-09-08-results-materials/observations/O013-overview-all-variants.md)
and `04_overview_v2.py`. The PDF was rebuilt and the changed pages visually checked.

The [preceding verification](reviews/2026-09-08-kernel-argument/verification.json)
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

The current draft and retained manuscript history are now version-controlled.
Analysis 018
[O015](../../analyses/018-2026-09-08-results-materials/observations/O015-threshold-kernel-argument.md)
records the 8 September argument revision and its source/PDF snapshot. The earlier
[O014 polish](../../analyses/018-2026-09-08-results-materials/observations/O014-manuscript-polish.md),
[argument map](results-argument-map.md), [figure plan](results-plan.md),
[reviews](reviews/) and [archive](archive/README.md) preserve prior decisions.
The original 24-page results snapshot remains unchanged. No experiment was launched.

# Manuscript draft: ICLR 2027 format

Current compiled draft: [main-14m-70m.pdf](main-14m-70m.pdf). Analysis024 Figure08
is now Figure 1 on page 2, with 14M/70M quality-sparsity and full-model latency
panels. The caption defines the T/P legend and all four panels, including the
40 measured clipping settings. Its 44 trained checkpoints include the ten
70M h-only endpoints now recorded in the appendix. The setup and dependent
results/limitations have been reconciled to 74 total trained conditions.

The 35-page build has resolved references and no overfull boxes; all pages
were visually checked. The approved artwork is copied unchanged with source
and evidence hashes. The open `main.pdf` is locked against replacement, so
the current build uses the filename above. See the
[adoption record](reviews/2026-09-18-two-size-main-figure/README.md).
The following revision notes retain their historical scope.

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

The 17 September revision implements the updated pressure-placement review
task: 40/12/12 trained conditions, separate threshold/pressure scopes, six
replacement main figures, two supporting appendix PDFs and four tables.
The kernel figure uses absolute K050 latency versus quality and all 35
qualified historical checkpoints, including five restored four-site h-only
records. Seven-site h-only timing and h-only scaling remain pending.
The 32-page build has resolved references and no overfull boxes; four underfull
vertical boxes and one underfull horizontal box remain, with rendered pages
checked. Twelve focused tests pass. See the [review record](reviews/2026-09-17-pressure-placement/README.md)
and [analysis/task audit](../../analyses/021-2026-09-10-training-results-figures/pressure-scope/README.md).
The entries below describe earlier revisions.

On 11 September, the frozen Analysis 021 kernel v2 became Figure 7 on page 10.
Section 4.6 is now "When model-wide sparsity translates to speedup," with
Astra as the implementation tool. The section and Discussion distinguish
projection benefit, attention overhead, execution granularity and the native
denominator. Appendix D.4 retains the search history (Figure 14); D.5 adds
kernel diagnostics, all 30 checkpoint latencies (Table 11) and Figure 15.
Four kernel measurement exports accompany the draft with source hashes.
The 31-page build has resolved references, no overfull boxes, and five
underfull vertical-box warnings; affected pages were visually checked.
[Analysis 021 O010](../../analyses/021-2026-09-10-training-results-figures/observations/O010-kernel-manuscript-adoption.md)
records the evidence checks, unchanged first nine pages and interpretation
limits. Entries below describe earlier revisions.

The main-text boundary-contrast table (formerly Table 3) is removed without
replacement. Section 4.5 now states the two boundary-threshold conclusions
directly and refers to Appendix Table 7 for all five thresholds at each size.
The figure and numerical evidence are unchanged. The rebuilt 28-page PDF has
resolved references and no overfull boxes; the affected pages were visually
checked. The four underfull vertical-box warnings remain.

The frozen Analysis 021 cross-size figure is now Figure 6 on page 9.
Section 4.5 retains "Transfer across model sizes," narrows the claim to the
high-threshold complete-recipe ordering, and replaces the old normalized-row
explanation with endpoint fractions of the seven-site ceiling. The A0
optimization diagnostic is Figure 8 on page 18 in Appendix C.4, following
Table 5 on page 17. Its caption specifies centered nine-step arithmetic moving
means with partial edge windows. The appendix and Discussion describe realized
optimization regimes without asserting convergence or failure to converge.
Both copies match their approved source PDFs; hashes are in `figures/SOURCES.json`.
The 28-page build has resolved references and no overfull boxes; four underfull
vertical-box warnings remain. Affected pages were rendered and checked. See
[O006](../../analyses/021-2026-09-10-training-results-figures/observations/O006-scale-transfer.md)
and [O007](../../analyses/021-2026-09-10-training-results-figures/observations/O007-a0-optimization.md).
The following entries record earlier revisions.

On 11 September, the approved Analysis 021 composite replaced the main-text
distribution grid as Figure 5 on page 9. Section 4.4, "Local sparsity does not
determine model-wide sparsity," now uses the approved five paragraphs and
caption. The recent Q/K/V comparison retains Section 4.3. The full density
grid is Figure 8 on page 22 in Appendix D.1, explicitly supporting the
untreated-reference comparison; the three-size operation figure remains in
Appendix D.2 as Figure 9. The copied composite is byte-identical to Analysis 021,
with its hash in `figures/SOURCES.json`. The 28-page PDF builds with resolved
references and no overfull boxes; four underfull vertical-box warnings remain.
The affected pages were rendered and checked. See
[O005](../../analyses/021-2026-09-10-training-results-figures/observations/O005-distributions-and-operations.md).

On 11 September, Section 4.3 and Table 2 added the matched A7-versus-A4
comparison without pressure. The table reports raw losses and model-wide
sparsities at all five thresholds, highlighting kappa = 0.5. The text reports
the difference only in prose: +5.17 pp sparsity for +0.043 validation loss.
[Analysis 021 O004](../../analyses/021-2026-09-10-training-results-figures/observations/O004-qkv-thresholding.md)
records the retained evidence and checks. The 27-page PDF builds with resolved
references and no overfull boxes; the same three underfull vertical-box warnings
remain. Pages 6-8, including Table 2 on page 7, were rendered and checked.
The frozen figures are unchanged. Subsequent entries describe earlier revisions.

On 11 September, the frozen
[Analysis 021 paired-pressure figure](../../analyses/021-2026-09-10-training-results-figures/figures/03-14m-paired-interventions.pdf)
replaced the old 29-contrast overview as Figure 3 on page 6. Section 4.2 is
now "Paired effect of adding pressure," with the approved matched-threshold
caption, results, comparative summary and normalization caveat. Section 4.1
retains one local L1N/OL1 sentence linked to Appendix D, Table 6. The geometry
diagnostic remains, followed by the transition to activation distributions.
[O003](../../analyses/021-2026-09-10-training-results-figures/observations/O003-paired-interventions.md)
records the adoption. The 26-page PDF builds with resolved references and no
overfull boxes; three underfull vertical-box warnings remain. The affected
pages were rendered and checked. Earlier artwork and appendix tables are retained.

On 11 September, the frozen
[Analysis 021 OL1 figure](../../analyses/021-2026-09-10-training-results-figures/figures/02-14m-ol1-geometry.pdf)
was added as Figure 4 on page 7, with its approved caption and source hash.
Section 4.2 reports the task-relative opposing component and cap activity;
Section 3.1 makes saturation conditional, Appendix A.2 defines the diagnostic
and implemented cap condition, and the setup and discussion reflect the
target-set distinction. Evidence and scope are recorded in
[O002](../../analyses/021-2026-09-10-training-results-figures/observations/O002-ol1-geometry.md).
The 26-page draft builds with resolved references and no overfull boxes;
the affected pages were rendered and visually checked. Three underfull
vertical-box warnings remain. The frozen artwork is unchanged.

On 10 September, the quality-sparsity overview was moved into the introduction
using the approved polished
[Analysis 021 figure](../../analyses/021-2026-09-10-training-results-figures/figures/01-14m-quality-sparsity.pdf):
short labels, 26 trained checkpoints (naive L1 omitted), baseline and ReLU
clipping, and 12.83%/29.95% ceiling guides. It is Figure 1 on page 2, directly
above the Pythia paragraph that cites it. The caption and source manifest were
updated, and the experimental section refers back to it without repeating the
figure. Earlier overview PDFs remain retained. The entries below record earlier
revisions.

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

- [main.pdf](main.pdf): 32-page ICLR 2027 draft; main text pages 1-11,
  references pages 11-12, appendices pages 13-32. Before submission, shorten the
  main text to nine pages and add the required author-reviewed AI use statement.
- [abstract.tex](abstract.tex), [introduction.tex](introduction.tex),
  [related-work.tex](related-work.tex), [methodology.tex](methodology.tex):
  framing, literature and definitions.
- [experimental-study.tex](experimental-study.tex),
  [training-results.tex](training-results.tex),
  [kernel-autoresearch.tex](kernel-autoresearch.tex): matched protocol,
  seven main figures, including the architecture/recipe diagram, and matched comparisons.
- [conclusion.tex](conclusion.tex): scientific contribution, agreement across
  evidence, execution implications and study scope.
- [methodology-appendix.tex](methodology-appendix.tex),
  [experimental-appendix.tex](experimental-appendix.tex),
  [results-appendix.tex](results-appendix.tex): definitions, complete results,
  diagnostics, post-hoc thresholding, kernel qualification and data provenance.
- [figures/SOURCES.json](figures/SOURCES.json): figure-copy provenance.
  The pressure-placement figures use Analysis 021's six-recipe comparison;
  earlier overviews and diagnostics are retained with their original coverage.
- [tables/](tables/): analysis-owned tables, including the historical kernel
  latency ablations, original 64 endpoints and ten additional 70M h-only
  endpoints, with explicit threshold/pressure scope.
- [supplementary-data/README.md](supplementary-data/README.md): nineteen
  measurement copies plus a separately sourced [protocol](supplementary-data/protocol.json).
  Includes all 540 historical post-hoc evaluations, seven signed histograms,
  and the pressure-placement extension with source hashes.

The paper uses activation sparsity, model-wide sparsity, sparsity ceiling,
thresholding and nonlinearities consistently. Operational data keys retain
their original names. This revision adds retained h-only evidence; original
measurements are unchanged and the final-loss pass convention is disclosed.

## Review and verification

The introduction-figure adoption was rebuilt with resolved references and no
overfull boxes. Page 1 is pixel-identical to the preceding draft; pages 2-7
were rendered and checked after the reflow. The figure copy matches the approved
Analysis 021 PDF and its recorded SHA-256. Two underfull vertical boxes remain
on pages 3 and 7; both pages were visually checked.

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

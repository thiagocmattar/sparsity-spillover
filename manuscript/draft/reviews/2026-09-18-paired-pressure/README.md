# Paired pressure figure and lean Section 4.2

## Changes

At the author's request, [Analysis024 Figure03](../../../../analyses/024-2026-09-17-h-only-kernel-latency/figures/03-pressure-scope-threshold.pdf)
replaces the previous pressure-increment figure under the existing
`fig:paired-intervention-effects` label. The new paper copy is byte-identical
to the source: Figure 4 on page 6 of the rebuilt 34-page [main.pdf](../../main.pdf).
The subsection now contains two short paragraphs and a caption defining
the all-site-minus-h-only contrasts across 14M/70M, their signs and timing scope.

The detailed OL1 budget paragraph and figure move to Appendix D.6 (Figure 17
on page 33), retaining their values and artwork and making their 14M scope
explicit. The preceding subsection includes the author's edits that were
already present when this task started; its wording is preserved, with one
trailing space removed. This compiled source state is included in the commit
so the delivered PDF remains reproducible. Unrelated working files are excluded.

## Evidence and supported wording

The [source observation](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/015-pressure-scope-caption.md)
defines the methods, coverage, full caption and limits. The generator is
`paper_effect_figures.py`, invoked by `13_rebuild_paper_figures.py` in that
analysis. Existing artwork and scientific data are reused without regeneration.
Figure and data `SOURCES.json` manifests record source paths and copy hashes.

The copied [derived-data export](../../supplementary-data/paired-pressure-figure-data.json)
contains all twenty exact pairs, matched by model size, T4/T7 and five kappa
values. All forty endpoint identities and loss/sparsity/latency values match
the existing Figure 1 export. Loss uses ordinary final-checkpoint validation;
logical sparsity uses pooled integer counts over the full 338-block validation
set. Training initialization, data order and budget are matched within pairs.

At kappa <= .1, h-only pressure lowers loss in all sixteen pairs by
0.192102-0.324167, while absolute model-wide sparsity differences are at most
3.356191 percentage points. At kappa=.5, broader pressure increases sparsity
in all four pairs; for T7/70M it also lowers loss by 0.121262.

The requested universal latency statement is qualified by the saved data:
h-only pressure has lower geometric-mean latency in eighteen of twenty pairs,
including all ten 70M pairs. All-site pressure is nominally faster at 14M,
kappa=.5, by 2.776373 us for T4 and 0.521765 us for T7. These are reported as
small reversals, not erased or called statistically equivalent. The figure
retains the negative values. The two 14M T7 checkpoints span physical GPU
sessions; 70M uses the qualified shape-specific port. These are descriptive
single-seed contrasts, with no claim of robust timing improvement for tiny
differences or a causal attribution to one added pressure site.

## Verification

Seven existing focused Analysis024 tests pass, including checkpoint/count
reconciliation, pressure matching, latency subtraction, session identity,
source hashes and preservation of the original archived figures. A direct
adoption audit also checks every numerical claim and matches all forty
endpoints to the existing manuscript export.

BibTeX and three pdflatex passes complete in an isolated temporary directory.
References and citations resolve; there are no overfull boxes. Six underfull
vertical boxes and one underfull horizontal box remain. The adopted figure,
page 6 and the moved appendix diagnostics were inspected at high resolution;
contact sheets cover all 34 pages. The existing figure PDFs and source data
remain unchanged. No experiment, new diagnostic or cloud operation ran.

[verification.json](verification.json) records numerical checks, source and
output hashes, test results, build warnings and resolved label locations.

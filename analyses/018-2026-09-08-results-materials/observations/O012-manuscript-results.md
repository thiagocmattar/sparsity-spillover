# Results adopted into the manuscript draft

## Question

How should the approved figures support a readable chain of scientific arguments?

## Method

Map claims, evidence, boundaries and transitions before drafting, then write
the results from O001/O002/O003/O004/O007/O011 and Runs 030/031. Copy the five
selected PDFs without modifying their bytes. Generate seven small TeX tables
directly from retained records with `03_manuscript_tables.py`; add protocol,
complete results, distributions, clipping and kernel details to the appendix.
No experiment, diagnostic evaluation or kernel timing is launched.

## Coverage

Five main figures in order 01, 02, 05-v3, 03, 07; 54 trained endpoints,
29 paired 14M effects, 15 scale comparisons, 14 pooled density groups from
seven checkpoints, all 540 clipping evaluations, and the 30-checkpoint
runtime cohort. Validation is all 500 documents packed into 338 complete
2,048-token blocks, with the 1,444-token tail excluded.

## Figures and captions

Paper Figures 1-5 correspond to the five main captions in [CAPTIONS.md](../CAPTIONS.md).
The draft adds resolved appendix/table references and shortens the kernel
caption, retaining its qualification, denominator, precision and claim limits.
The original architecture/ladder and operation-accounting figure plus all
three full-range clipping PDFs appear in the appendix. Copy identities are
recorded in the snapshot's `figures/SOURCES.json`.

## Result

The argument is: quality-sparsity regimes; conditional paired intervention
effects; distribution reshaping and operation contributions; high-threshold
recipe transfer; qualified execution and sparse-path ablations. A short
introduction preview states the observed conclusions. The kernel prose and
appendix now consistently use the corrected 30-checkpoint cohort.

The reading copy has 24 pages: main text and five results figures on pages
1-10, references on page 11, and appendices on pages 12-24. One six-row
contrast table is in the main text. Local `supplementary-data/` contains
14 byte-identical evidence files (13.68 MB) and their source/hash manifest.

## Caveats

This is the existing lightweight reading wrapper, not an ICLR submission
template or a claim of submission-page fit. One seed per size is retained.
Density comparisons support a nonlocal response under a complete recipe,
not pressure-specific causal identification; larger cohorts test complete
recipes. Logical opportunities and measured BF16 speedups remain distinct.
The draft stays Git-ignored; a self-contained source/PDF snapshot is retained
under this analysis. Source runs remain unchanged; no finding is promoted.

## Source and verification

- [Argument map and complete manuscript snapshot](../provenance/manuscript-20260908-results/README.md).
- [Table formatter](../03_manuscript_tables.py) and [table source hashes](../manuscript-tables/SOURCES.json).
- [Verification record](../provenance/manuscript-20260908-results/VERIFICATION.json).

Table construction checks all 54 integer-count totals, 44 paired differences
and 14 histogram count/zero/tail reconstructions. LaTeX/BibTeX resolves all
references and citations without box warnings or errors. All 24 rendered
pages were inspected, including full-size checks of the dense figure and
table pages; fonts are embedded with no Type 3 fonts.

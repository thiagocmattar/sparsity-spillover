# v0 - Matched intervention ladder

Archived **2026-09-05** from the active draft rewritten on 2026-09-04.

**Paper:** Where to Sparsify a Transformer: Matched Interventions in Architecture
and Optimization.

**Main argument:** Gate placement and activation pressure change the
quality-sparsity trade-off. Attention-operand gates extend the available sparsity
opportunity, while broader pressure can increase its quality cost.

**Framing:** An inference-efficiency-motivated, matched intervention study:
detailed 14M evidence, selected 70M comparisons, operation-level accounting,
410M appendix results, and a negative 70M runtime evaluation.

This snapshot precedes the proposed pressure-allocation-by-nonlinearity study.

## Contents and provenance

- [main.pdf](main.pdf) and [main.tex](main.tex): archived paper and entry point.
- `sections/`, `figures/`, `tables/`, `references.bib`, and `iclr-style/`:
  bundled manuscript sources and assets, including existing unused asset copies.
- [rewrite-review.md](rewrite-review.md): argument and editorial decisions.
- [README.source.md](README.source.md): the original draft README, unchanged.
- [snapshot-manifest.json](snapshot-manifest.json): source paths, byte counts,
  SHA-256 hashes, archive timestamp, and repository HEAD for 61 copied files.
- [working-folder/](working-folder/): the complete former working directory,
  moved here to clear `draft/` for V2, including reviews, revisions, QA renders,
  and build files.
- [working-folder-manifest.json](working-folder-manifest.json): paths, byte
  counts, and verified SHA-256 hashes for all 102 moved files.
- [Analysis 013](../../../../analyses/013-2026-09-04-matched-intervention-manuscript/README.md):
  owning evidence reduction and observations.

The source files and PDF were copied byte-for-byte. The original README was
renamed to `README.source.md`; this README and the inventory are archive metadata.
The initial 61-file snapshot excluded earlier `revisions/`, `qa/` renders,
the redundant style ZIP, and temporary TeX build files. The complete
`working-folder/` archive now retains those materials too. The parent `draft/`
contains only its new README and `archive/`, ready for V2.

## Rebuilding

To rebuild the bundled paper, copy this version to a scratch directory and run
`latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` there. Keep this
archive unchanged. The preserved `01_build.ps1` and historical notes retain
their original working-draft path assumptions; that script regenerates assets
from the repository and should be used only from the original layout.

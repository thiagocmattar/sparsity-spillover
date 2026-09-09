# ICLR 2027 v0 submission draft

This ignored workspace consolidates the paper-facing draft without changing the
older tracked `manuscript/introduction.tex` and `manuscript/methodology.tex`.
Those files remain historical inputs; the sources here are the v0 target
submission.

## Build layout

- `main.tex` - anonymous ICLR 2027 parent document.
- `sections/` - hand-edited abstract, introduction, related work, measurement,
  interventions, experiments, results, discussion, conclusion, and appendix.
- `references.bib` - primary-paper bibliography.
- `figures/` and `tables/` - paper bundle copied from the owning analyses by
  `01_build.ps1`.
- `framing-review.md` - adversarial review of the 2026-09-04 framing note.
- `claim-ledger.md` - source and limitation ledger for result-bearing prose.
- `supplementary-results.md` - result inventory and 410M inclusion decision.
- `submission-readiness.md` - format checks and human closeout items.
- `handoff.md` - clean-room research bootstrap specification for a separate
  implementation repository.

The paper-facing figures and tables are owned by
`analyses/012-2026-09-04-paper-synthesis/`. This draft copies them only to make
the ignored submission bundle self-contained.

## Evidence boundary

The current results are one-seed, one-pass MiniPile evidence at Pythia-14M,
70M, and 410M. All endpoint losses and `R_model` values cover the 338 complete
validation blocks and use the paired eager logical-product pass. `R_model` is a
logical exact-zero opportunity, not a sparse-kernel or wall-clock measurement.

The Pythia-410M cohort is retained as a fixed-token boundary condition. It must
not be described as proof of undertraining. The trained A4-OL1/A7-OL1 contrast
changes both gate topology and pressure sites; only the no-pressure 14M A4/A7
comparison isolates topology expansion.

## Intended build

Run Analysis 012 first, then run `01_build.ps1`. The script copies the exact
PDF/table outputs, records their SHA-256 values, invokes `latexmk`, and leaves
`main.pdf` beside the TeX sources. The official ICLR 2027 style files are kept
locally in this ignored folder. Review `main.log` for layout/reference warnings
and render every PDF page before treating the build as publication-ready.

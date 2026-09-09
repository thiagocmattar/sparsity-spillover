# V2 working draft

The reviewed draft contains [introduction.tex](introduction.tex),
[related-work.tex](related-work.tex), and [methodology.tex](methodology.tex),
with exact definitions in [methodology-appendix.tex](methodology-appendix.tex)
and citations in [references.bib](references.bib).
The previous paper and working materials remain in [archive/](archive/README.md).

Framing: how activation pressure, gate nonlinearity, and site placement
interact to determine the quality--sparsity trade-off. The introduction develops
the question and a candidate explanation; it does not announce new findings.
Results remain intentionally deferred.

The user-requested [adversarial review](reviews/2026-09-05-adversarial/README.md)
contains three independent sub-agent assessments with explicit scoring rubrics,
initial and revised scores, and a response to their criticisms. The revised
text centers separately controllable choices whose effects need not be
separable. Related work has three paragraphs on sparsification interventions
and two on inference speedup, with gradient-projection context confined to the methodology appendix.

The subsequent [methodology review](reviews/2026-09-05-methodology/README.md)
uses the same three independent perspectives and five-criterion scoring
structure. Main methods use inline math and cover interventions, model-wide
sparsity, and selected-site reach; formal counters and optimizer mechanics
are in the appendix. [methodology-notes.md](methodology-notes.md) records
operational provenance and the paper's calligraphic-S notation crosswalk.

[positioning.md](positioning.md) records the argument, source checks, evidence
boundaries, and length budget. The user's [supplementary.md](supplementary.md)
is unchanged. [main.tex](main.tex) is a lightweight reading wrapper, including
the existing architecture/ladder PDF as a setup reference.

From this directory, build the reading copy with:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The draft retains the repository's local-only ignore policy; its existence and
scope are recorded in the tracked parent [README](../README.md).

Verified after revision on 5 September 2026: the LaTeX/BibTeX build completes
with no undefined citations/references or overfull/underfull boxes. All 15
cited sources resolve, and citation keys are unique. All eight rendered pages
were visually inspected. The introduction is approximately 630 words, related
work 570, and main methodology 560 (inline expressions counted as one word).
The three sections and setup figure occupy four pages in this reading layout;
references occupy page 5 and the methods appendix pages 6?8. This is not a
submission-template fit claim.

All twelve analytic architecture/topology numerator and denominator pairs
match the existing ceiling implementation. The setup figure's labels were
enlarged for legibility, with no changes to its values or site mapping.
No scientific code was modified and no experiment was launched. The draft
retains its local-only ignore policy; the tracked parent index and setup
artifacts record this manuscript revision.

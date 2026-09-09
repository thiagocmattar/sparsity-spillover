# V2 working draft

The reviewed argument pass contains [introduction.tex](introduction.tex) and
[related-work.tex](related-work.tex), with [references.bib](references.bib).
The previous paper and working materials remain in [archive/](archive/README.md).

Framing: how activation pressure, gate nonlinearity, and site placement
interact to determine the quality--sparsity trade-off. The introduction develops
the question and a candidate explanation; it does not announce new findings.
Results and methodology sections are intentionally deferred.

The user-requested [adversarial review](reviews/2026-09-05-adversarial/README.md)
contains three independent sub-agent assessments with explicit scoring rubrics,
initial and revised scores, and a response to their criticisms. The revised
text centers separately controllable choices whose effects need not be
separable. Related work has three paragraphs on sparsification interventions
and two on inference speedup, with the gradient-surgery strand deferred.

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
with no undefined citations/references or overfull/underfull boxes. All 11 cited
sources resolve; three additional bibliography entries are retained for later
methods writing. Citation keys are unique and active documentation links
resolve. All five rendered pages were visually inspected. The introduction
is approximately 630 words and related work 570 words, occupying roughly
2.2 pages in this reading layout; page 4 reproduces the setup artifact and page
5 contains references. No scientific code or experiment artifacts were changed,
and no training or evaluation was launched.

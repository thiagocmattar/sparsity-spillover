# V2 working draft

The reviewed draft contains [introduction.tex](introduction.tex),
[related-work.tex](related-work.tex), [methodology.tex](methodology.tex), and
the setup/comparison opening of [experimental-study.tex](experimental-study.tex).
General definitions are in [methodology-appendix.tex](methodology-appendix.tex);
Pythia-specific details are in [experimental-appendix.tex](experimental-appendix.tex).
Citations are in [references.bib](references.bib).
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

The [section-split review](reviews/2026-09-05-methods-experiments-split/README.md)
subsequently shortened Methodology to general definitions and moved Pythia,
MiniPile, A* recipes, comparison logic, and the figure into Experimental Study.
The experimental opening explains what the comparisons can establish;
empirical findings remain deferred. [experimental-notes.md](experimental-notes.md)
records its provenance and the one-seed, selected-recipe coverage limits.

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

Verified after the section split on 5 September 2026: the LaTeX/BibTeX build
completes without undefined citations/references or overfull/underfull boxes.
All 15 citation keys resolve. All nine final PDF pages were visually checked.
Main Methodology is approximately 380 words; the Experimental Study opening
is approximately 320 words excluding its caption. The reading copy puts
Methodology on page 3, the experimental opening and figure on page 4, and its
comparison paragraph on page 5. References occupy page 6 and appendices pages
7-9. The short page 5 leaves the continuation point for future results; no
submission-template fit claim is made.

The four numbered equations were preserved exactly across their relocation.
The existing figure artwork and all ceiling values remain unchanged in this
structural revision. No scientific code was modified, no numerical results
were regenerated, and no experiment was launched. Draft sources and review
records retain their local-only ignore policy; the tracked parent index and figure-provenance notes record
this manuscript revision.

# Manuscript results snapshot, 8 September 2026

User-authorized results adoption after the five-figure plan. This is a
self-contained TeX/PDF snapshot of the otherwise Git-ignored working draft.
It records the completed state, not a historical sequence of intermediate edits.

- [main.pdf](main.pdf): final 24-page reading copy.
- [results-argument-map.md](results-argument-map.md): claim/evidence/limitation
  chain written before the results prose.
- [training-results.tex](training-results.tex) and [kernel-autoresearch.tex](kernel-autoresearch.tex):
  five results subsections, five main figures, one compact contrast table.
- [results-appendix.tex](results-appendix.tex): complete results and diagnostics.
- [draft-README.md](draft-README.md): working-draft index at capture time.
- [SNAPSHOT.json](SNAPSHOT.json): byte-identical copied source/PDF identities.
- [figures/SOURCES.json](figures/SOURCES.json): original figure sources/hashes.
- [VERIFICATION.json](VERIFICATION.json): numerical, build and layout checks.

## Build

Run from this directory with a LaTeX installation providing the packages
listed in main.tex:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

All included TeX, bibliography, table and figure dependencies are present.
Markdown copies retain their original working-draft links; they are context
records, not build dependencies. The PDF is a 5.5-inch lightweight reading
wrapper, not an ICLR submission-template/page-budget assessment.

## Evidence

[Observation O012](../../observations/O012-manuscript-results.md) identifies
all source observations and claim boundaries. The unchanged figure PDFs
include main figures 01, 02, 05-v3, 03 and 07, plus the architecture ladder,
operation accounting and three full-range clipping figures in the appendix.
The [table formatter](../../03_manuscript_tables.py) uses retained evidence
and verifies count pooling and paired differences; no evaluation is run.
The appendix endpoint table deliberately recomputes U against the common A7
ceiling, rather than copying recipe-specific U from the raw records.

The complete 14-file, 13.68 MB measurement release is copied locally into
`manuscript/draft/supplementary-data/`. Its [manifest](data-release-manifest/SOURCES.json)
and [field dictionary](data-release-manifest/README.md) are retained here.
To reconstruct that directory, copy each manifest entry's repository-relative
`source` to its relative destination and verify its SHA-256. The raw data
already belong to Analysis 018 and Runs 030/031; this snapshot avoids a second
tracked copy. No token dataset, model weights, cache or credentials are included.

## Verification

The final build has 24 pages, 15 resolved citation keys, no unresolved
references, no box warnings and no compilation errors. All fonts are
embedded with no Type 3 fonts. All pages were visually inspected, with
full-size checks of the dense main figures and endpoint table. All page
renders after the final rebuild exactly match the inspected renders.
Ten figure copies and fourteen evidence copies match their sources by SHA-256.
All seven table files match the analysis outputs. The formatter verifies
54 count totals, 44 paired differences and 14 pooled histograms.
No scientific evaluation was launched; source run records are unchanged.

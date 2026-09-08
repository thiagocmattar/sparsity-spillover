# Reviewed manuscript snapshot - 8 September 2026

This analysis-owned snapshot preserves the user-authorized manuscript polish
while `manuscript/draft/` remains local-only. The earlier results snapshot is
unchanged. See [O014](../../observations/O014-manuscript-polish.md) for scope.

[main.pdf](main.pdf) is the verified 26-page reading copy. All inputs needed
to build it are local to this directory: twelve TeX files, bibliography,
figures and seven generated tables. Eleven figure assets are retained; ten
are embedded and the overview v2 remains an alternative. The original artwork
and numerical measurements are unchanged.

The [revision log](reviews/2026-09-08-polish/revision-log.md) explains each pass,
user steering and review resolution. The same folder retains technical-reader,
scientific and literature reviews, final visual checks, protected introduction
paragraphs and [verification](reviews/2026-09-08-polish/verification.json).
`SNAPSHOT.json` identifies every copied source, byte count and SHA-256.
`.gitattributes` preserves exact bytes rather than normalizing copied sources.

## Rebuild

From this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

This reproduces the existing reading wrapper; conference-template fitting
was excluded from the task. A clean build is verified separately in
`CLEAN-BUILD.json`. PDF metadata can differ; rendered pages are compared.

## Measurement data

The supplementary README, copy manifest and newly composed protocol are
retained here. Fourteen larger measurement copies are not duplicated in this
snapshot. Their `source` paths in `supplementary-data/SOURCES.json` are relative
to the repository root; copy each to its manifest key to reconstruct the local
measurement directory and verify its SHA-256. The original sources remain in
Analysis 018 and Runs 030/031. They include all 540 clipping evaluations and
seven signed histograms. `protocol.json` has its own thirty-source provenance
map and records the exact preparation, training and runtime scope.

No training, diagnostic evaluation, timing experiment or figure regeneration
was performed during the manuscript polish. Only three table formatter labels
changed; all seven numerical table bodies were checked against the prior version.

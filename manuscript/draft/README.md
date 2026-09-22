# Manuscript navigation

Compile `main.tex`; it is the only file that includes other TeX files.
Read and edit the numbered sections in this order:

- `01-abstract.tex`
- `02-introduction.tex`
- `03-related-work.tex`
- `04-methodology.tex`
- `05-experimental-study.tex`
- `06-discussion-conclusion.tex`
- `07-ai-use-statement.tex`
- `08-appendix-interventions.tex`
- `09-appendix-sparsity-accounting.tex`
- `10-appendix-experimental-setup.tex`
- `11-appendix-complete-results.tex`
- `12-appendix-kernel-validation.tex`

Tables are embedded in their owning section. The complete 14M/70M table uses
`longtable`, repeats its column headings, and keeps each recipe together.
Original table fragments are retained locally under
`.archive/draft-reorganization-tables/` and in Git history. Historical analysis
generators still produce those standalone fragments; apply regenerated table
contents to the owning numbered section when updating results.

`figures/` contains the PDF figure assets. `references.bib` and the bundled
conference style files are required for compilation. `.archive/` is ignored.

From this directory, run:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

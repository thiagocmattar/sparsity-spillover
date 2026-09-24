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

The scale comparison in Figure 5 now uses Analysis038 Figure 01, copied to
the stable asset path `figures/02-14m-70m-latency-quality.pdf`. Figure 6 uses
Analysis038 Figure 03, adding the 31M Base trajectory. The 31M result table
is embedded in `11-appendix-complete-results.tex` as `tab:31m-results`.
Evidence, generators and the approved change checklist are linked from
[Analysis038 O003](../../analyses/038-2026-09-24-base-t2-t7-scale-frontier/observations/003-manuscript-integration.md).
The Analysis038 Figure 02 companion is not used in this draft.

From this directory, run:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

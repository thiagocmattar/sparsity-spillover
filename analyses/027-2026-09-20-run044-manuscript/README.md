# Analysis027: surgical Run044 manuscript integration

The user requested that the completed 14M T2/Ph kappa=0.5 result be included
in Figure 1 and the full manuscript data. This is a local reduction and
manuscript update; no model, evaluation, timing or cloud execution is added.

## Scope and completed plan

1. Add Run044 to the existing 40-point Figure 1 without changing its ten recipe
   styles, bounds, clipping paths, historical measurements or layout.
2. Complete the T2/Ph grid in the appendix. The earlier 74-row complete-data
   export omitted all four Run041 T2 points; include those plus Run044. The
   complete inventory is now 79 endpoints (45/22/12 at 14M/70M/410M). The
   41-point main figure continues to exclude the four historical naive-L1
   ablations, as previously selected by the author.
3. Update only the relevant counts, kappa list, evidence links and missing
   post-hoc coverage statement. Preserve the author's other edits.
4. Rebuild, inspect and install the 21-page PDF, verify all source measurements
   and copied artifacts, and commit this scoped change.

The compact table adds T2/Ph as a third column pair alongside the existing
T4/P0 and T7/P0 kappa sweeps. This retains the original table height and both
complete-results tables on page 16. Figure 1 remains on page 2.

The build exposed an existing author reference to the undefined label app.
A label alias at the kernel appendix resolves it without rewriting the
author's experimental-study paragraph. Existing author edits are preserved;
only the targeted introduction replacements and this label are staged from
the previously modified TeX files.

## Outputs and provenance

- [Updated Figure 1](figures/01-14m-complete-t2-grid.pdf).
- [Observation, caption and limits](observations/001-run044-integration.md).
- [Exact Figure 1 coordinates](data/figure-data.json).
- [Complete 79-endpoint data](data/full-trained-results.json).
- [Generated compact table](tables/14m-only.tex).
- [Numerical and artifact verification](data/verification.json).
- [Build, layout and preservation checks](data/manuscript-verification.json).

The source observations are [Run044](../../runs/044-2026-09-20-pythia14m-hz-h-only-ol1-kappa05/observations/001-final-results.md)
and [Analysis026](../026-2026-09-20-14m-hz-quality-sparsity-latency/observations/001-quality-sparsity-latency.md).
Analysis024's earlier main-figure and complete-data exports are retained
unchanged. Manuscript copies retain their existing filenames and updated
SOURCES.json identities.

## Reproduction

From the repository root:

    .venv/Scripts/python.exe -X utf8 analyses/027-2026-09-20-run044-manuscript/01_collect.py
    .venv/Scripts/python.exe -X utf8 analyses/027-2026-09-20-run044-manuscript/02_plot.py
    .venv/Scripts/python.exe -X utf8 analyses/027-2026-09-20-run044-manuscript/03_table.py
    python -X utf8 analyses/027-2026-09-20-run044-manuscript/04_verify.py

The final verifier additionally checks the installed manuscript copies.
After regeneration, copy the figure, the two data files and the table to the
destinations recorded by manuscript/draft/figures/SOURCES.json and
supplementary-data/SOURCES.json; the table destination is
manuscript/draft/tables/compact-results/14m-only.tex.

For the checked manuscript build, top-level TeX/bibliography/style files and
the figure/table directories were copied into an isolated temporary directory.
Run pdflatex in nonstopmode with halt-on-error, then bibtex, and repeat
pdflatex until labels stabilize. Isolation prevents stale working directory
auxiliary files from shadowing the new references. The installed PDF is
byte-identical to the visually reviewed build. All 21 pages were reviewed
as a contact sheet; changed pages 2, 5, 16 and 17 were inspected at reading
resolution. All other page bodies match the author-source baseline.
References and citations resolve, with no overfull boxes and three pre-existing
underfull horizontal-box warnings.

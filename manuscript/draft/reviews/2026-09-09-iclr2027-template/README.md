# ICLR 2027 template conversion, 9 September 2026

The user requested conversion of the current main draft to the
[official ICLR 2027 template](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip).
The freshly downloaded ZIP is retained here. Its SHA-256 and the hashes of the
four installed style files are recorded in [verification.json](verification.json).
All four files match the download byte-for-byte.

Only the manuscript wrapper changed: `article` now loads
`iclr2027_conference,times`, uses the official title/anonymous author block,
headings, spacing, captions, review header and line numbers, and selects the
official bibliography style. Custom geometry and reading-copy formatting were
removed. References follow the conclusion and appendices start on a fresh page.
The title text, all section TeX and bibliography entries, figures, numerical
tables and supplementary measurements are unchanged.

The final [PDF](main.pdf) has 27 pages: main text 1-13, references 13-14 and
appendices 15-27. All pages were rendered and visually inspected. All 52 labels
and 15 citations resolve; all fonts are embedded, with no Type 3 fonts, overfull
boxes or text outside page bounds. The template produces one underfull vertical
box on page 3 (badness 1383); inspection found no clipping or overlap. The
official style and its standard page stretching were not modified to suppress it.

This preserves the full draft; it is not a nine-page submission edit. Before
initial submission, shorten the main text to the template's nine-page limit and
add its required AI use statement after author review. No author identity or
disclosure attestation was invented. Anonymous review mode remains enabled;
`iclrfinalcopy` is commented out.

Build from `manuscript/draft` with `pdflatex -interaction=nonstopmode
-halt-on-error main.tex`, then `bibtex main` and two further `pdflatex` passes.
The [wrapper](main.tex), [PDF](main.pdf) and [log](main.log) retain this build.
Draft assets remain Git-ignored/local-only; the tracked manuscript index records
the conversion. No scientific code, result or experiment was changed.

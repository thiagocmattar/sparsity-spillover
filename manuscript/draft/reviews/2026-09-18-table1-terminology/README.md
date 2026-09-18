# Table 1 terminology and simplification

The author requested a simpler Table 1 consistent with Figure 1, retaining
the 410M column, separating threshold families, and removing the naive-L1 row.
The table now has nine rows: T0/P0 (Base Model), T1/P0, T1/P1, and three
pressure scopes each for T4 and T7. Horizontal rules separate T0/T1/T4/T7.

P1 and Ph both target h. P_all means that the pressure and thresholding site
sets coincide (P = N), so it corresponds to P1, P4 or P7 under T1, T4 or T7.
The table uses P_all for both multisite all-pressure rows and writes N in
their pressure-site cells. The methodology defines these aliases; Figure 1's
caption names the controls T0/P0 and T1/P0. All 27 displayed ceiling values
match their source rows in the preceding table, including the 410M column.
The ceiling provenance remains linked in `experimental-study.tex`.

The caption and experimental setup text are shorter. Historical naive-L1
measurements remain in the appendix and in the 74-condition study count;
removing their Table 1 row does not erase measured results. This step does
not relabel historical figure artwork or measurement identifiers.

The canonical [main.pdf](../../main.pdf) has been rebuilt. Figure 1 remains
on page 2 and Table 1 on page 5. The author-requested cleanup removes
`main-14m-70m.pdf` and `main-readability.pdf`; their prior versions remain
in Git. Historical review records retain their hashes and explain the cleanup.

The 35-page PDF builds with resolved references and no overfull boxes. The
table, caption, definitions and reflowed pages were rendered and inspected.
Existing underfull-box and longtable glue diagnostics are recorded in
[verification.json](verification.json). No training, measurements, ceiling
calculations or source figure artwork changed. Unrelated ongoing analysis
work is excluded from this commit.

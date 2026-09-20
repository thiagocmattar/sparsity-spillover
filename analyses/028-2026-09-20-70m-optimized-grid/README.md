# Matched 70M implementation comparison

Approved manuscript integration of Run045's 26-checkpoint comparison of native
PyTorch, the original Run035 port and frozen Run042 opt073. Keep 14M as the main
study; use 70M to test implementation transfer. Canonical final-checkpoint quality
and integer-pooled logical sparsity remain distinct from BF16 numerical checks.

Complete: all 26 checkpoints / 78 fresh processes qualify. All 819 returned
artifacts pass local size/SHA256 checks, and local reduction exactly reproduces
the remote summary. Both owned Pods are deleted; estimated total cost is below
USD2.60 (not a settled invoice).

Scope: update the 70M figure, paired-pressure latency differences, compact tables,
cohort counts, and the kernel appendix; make surgical narrative changes. Preserve
the author's current wording and all unrelated edits. Run046 is outside this
approved timing cohort. Per the user's explicit presentation decision, its
additional T2/Ph kappa=.5 endpoint is retained separately in the appendix, with
original-port timing and optimized latency explicitly unmeasured. No new training, post-hoc clipping or kernel search is performed.

`work/author-baseline-before-edit/` holds an ignored snapshot of the working TeX
immediately before this task's first successful edit, allowing only this task's
changes to be staged at closeout. It supersedes the earlier initial snapshot.

The analysis uses `01_collect.py`, `02_figures.py`, `03_tables.py` and
`04_controls.py`; `05_copy_artifacts.py` installs only the approved generated
artifacts. See [the matched comparison](observations/001-matched-70m-grid.md)
and [retained 14M diagnostics](observations/002-retained-14m-diagnostics.md).

`06_verify.py` checks all 84 endpoints, preservation of 79 historical records,
26-point figure membership, integer-count arithmetic, backend joins, table values
and installed artifact hashes. The optimized 70M peak is 1.480544x the fixed
1.611048ms native Base reference; the original port peaks at 1.013479x.
Dense optimization and sparse execution both contribute. Low-loss T2 points
remain slower than native Base. See `data/verification.json` and
`data/build-verification.json` for the final manuscript checks.

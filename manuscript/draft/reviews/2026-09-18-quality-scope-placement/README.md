# Quality overview scope and manuscript placement

The author requested the quality overview directly after Table 1, followed
by the base-model speedup figure. The main quality figure should show only
14M/70M; the complete 410M curve belongs beside its existing appendix
trajectories and endpoint data. The main text must retain the three distinct
scale roles.

## Final placement

| Item | Number | Page |
| --- | --- | --- |
| Recipe and ceiling table, retaining 410M | Table 1 | 5 |
| All-recipe 14M/70M quality overview | Figure 3 | 5 |
| Unchanged base-model speedup plot | Figure 4 | 6 |
| Complete 410M quality panel | Figure 15 | 26 |
| Unchanged full 410M post-hoc trajectories | Figure 16 | 27 |

Protocol and calibration text now precede Table 1. The quality overview
follows the table without intervening prose, and the speedup figure follows
the quality overview. Its caption and agentic-development text are preserved
exactly. Detailed 410M quality/schedule discussion moves to the appendix;
the main setup keeps 14M's controlled-study role, 70M's targeted-replication
role and 410M's fixed-token sparsity stress-test role. The conclusion uses
the same 410M interpretation. Endpoint tables remain unchanged.

## Evidence and preservation

Analysis024 [observation 031](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/031-quality-main-appendix.md)
contains the two captions, source script and associated manuscript writing.
The new export partitions the existing records exactly: 62 trained / 40
control-clipping points in the main figure and 12 trained / 20
control-clipping points in the appendix. All 410M points are visible at
their complete loss range. The earlier three-panel PDF and original
74/60-point export are unchanged; their old manuscript label is retired.
The separate 410M historical figure remains byte-identical with all twelve
trained endpoints and 120 clipping evaluations.

No training, checkpoint evaluation, timing, source measurement or cloud
resource changed. Manuscript PDF copies and supplementary-data copies match
the new analysis outputs by SHA256.

## Checks

- Four existing all-model quality tests pass.
- Independent ID-based comparison confirms the exact 74/60-record partition.
- Source/output/copy hashes and preserved original figure hashes reconcile.
- `pdflatex`, `bibtex`, and two further `pdflatex` passes produce 33 pages
  with no unresolved references/citations and no overfull boxes.
- Six underfull vertical boxes and one underfull horizontal box remain;
  rendered layout has no clipping or overlap.
- Both standalone figures, high-resolution manuscript pages 5-7 and 26-27,
  and contact sheets covering all 33 pages were visually reviewed.

[verification.json](verification.json) records the final hashes, resolved
labels, pages and numeric checks. Build and visual-QA intermediates stay
under ignored `tmp/pdfs/quality-scope-placement/`.

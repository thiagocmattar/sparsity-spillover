# Status Report Number 2 notes

## Scope

Status Report Number 2 preserves Report 1's architecture/intervention diagram,
structure, completed 14M results, and historical Run 012 appendix. It advances
the cutoff through the verified Pythia-70M selected ladder in Run 018 and the
cross-scale synthesis in Analysis 010. The promoted 70M subset is A0, A1-H,
A4-OL1, and A7-OL1; unpromoted intermediate 70M ablations and all 410M rows
remain explicitly incomplete.

## Generated source

Run `python 01_build.py` from this folder or the repository environment to
regenerate `status-update.tex`. The focused builder starts from Report 1's TeX,
fails closed on expected section identities, and inserts only the updated scope,
RunPod execution record, analysis-status row, 14M/70M tables, and Analysis 010
frontier. It reads:

- `analyses/010-2026-09-01-pythia14m-vs-70m-selected-ladder/figure_data.json`;
- Run 018 accepted attempt manifests for scientific-process wall-clock;
- report-local `billing-refresh.json` for cost and resource provenance.

The report must be compiled with `pdflatex` twice so references and page numbers
settle. Run `python 02_verify.py` afterward to check deterministic source
generation, evidence coverage, billing reconciliation, expected report content,
and the final PDF signature and size. The tracked deliverables are
`status-update.tex`, `status-update.pdf`, `01_build.py`, `02_verify.py`,
`billing-refresh.json`, and this note; LaTeX auxiliary files are not
deliverables.

## Billing refresh

The 2026-09-01 RunPod REST v2 refresh queried hourly Pod billing from 29 August
through 2 September and reconciled the returned rows to exact Pod IDs recorded
by each run. Several earlier closeout snapshots were still missing late-posted
hourly buckets, including values described as settled. The refresh therefore
supersedes those snapshots for this report:

- seven main report runs: `$237.614005885212`;
- historical Run 012: `$14.849361371831037`;
- total including historical Run 012: `$252.46336725704305`.

These are Pod GPU plus temporary Pod-disk charges. They exclude the one retained
100 GB Standard network volume, which remains the only billable resource at
`$0.01/hour`. The same audit returned zero Pods.

## Interpretation limits

The 70M result is descriptive one-seed evidence across two model sizes, not a
scaling law. The selected promotion does not fill unrun 70M ladder cells and
does not substitute for the unobserved 410M promotion. Site percentages are
count-pooled exact-zero masses; `q` and `k` are post-RoPE. `R_model` is logical
zero-product opportunity, not measured runtime speedup. TEAL is evaluation-only
clipping and did not record downstream Q/K/V activation mass.

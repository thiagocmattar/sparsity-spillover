# Revision v2 handoff — 9 September 2026

The authorized EXISTING_EVIDENCE revision is complete. The [reading PDF](../main.pdf)
has 33 pages: main text 1–9, AI use statement/references 10–11, and appendices
12–33. All six main figures are retained at readable width; the expanded
cross-size comparison is Figure 5 on page 7 and the replacement runtime display
is Figure 6 on page 8. The full paper contains 15 figures and 14 numbered tables.
The official template and main wrapper are unchanged. The nine-page initial
submission limit was checked against the [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
on 9 September 2026.

## Deliverables and evidence status

Edited sources are the draft's `abstract.tex`, `introduction.tex`,
`related-work.tex`, `methodology.tex`, `methodology-appendix.tex`,
`experimental-study.tex`, `experimental-appendix.tex`, `training-results.tex`,
`kernel-autoresearch.tex`, `results-appendix.tex`, and `conclusion.tex`, plus
the new `training-dynamics-appendix.tex`. Existing BibTeX keys were verified;
`references.bib` required no change. [changes.md](changes.md) maps every task
to its final paragraph, figure or audit.

[Analysis 020](../../../analyses/020-2026-09-09-intervention-revision/README.md)
owns the reduction scripts, four new vector PDFs, generated training-dynamics
table, mathematical tests and five observations. Copies are under the draft's
`figures/` and `tables/`, with source hashes. [claims.csv](claims.csv) separates
observations, derivations, design rationale and hypotheses. The author-facing
[cross-size CSV](data/cross_size_interventions.csv) contains 396 records;
[training-curves.csv](data/training-curves.csv) contains 6,408 raw boundary records.
The 54 primary conditions, 540 clipping evaluations, 29 paired 14M contrasts
and 15 cross-size contrasts are unchanged. No original run record was modified.

| Topic | Verified | Conditional or unresolved |
|---|---|---|
| OL1 | All 24,208 boundaries from 34 OL1 conditions are present; 7,283 (30.09%) activate the relative cap. The ideal protected-direction, norm-bound and fixed-state saturation properties have proofs/tests. | Saturation varies sharply by recipe and size. The protected direction is the adaptive task direction, not the raw task gradient or actual task loss. This is not evidence of trajectory-level hyperparameter robustness. |
| ReLU followed by clipping | At target p = 0.5, ReLU A1-H has lower total loss and higher achieved sparsity than GELU A0 clipping at all three sizes. Incremental clipping damage is smaller at every positive tested target. | ReLU has an initial loss cost; total loss favors A0 at low targets and returns to favor A0 at extreme targets at 14M/410M. A matched quantile does not imply matched achieved sparsity. All policy rows and observed mismatches are retained. |
| Training dynamics | Nine complete 712-record trajectories identify task loss, AMP-unscaled pre-clipping task-gradient norm and learning rate. In fixed steps 570–712, 410M A0 continues improving; its median norm is 0.850 versus 0.317/0.295 at 14M/70M. | Both selected 410M sparse recipes also improve late. Norms, learning rates and tokens per parameter differ. There is no late validation trajectory or causal test that more training resolves the cross-size reversal. |
| Cross-size recurrence | At kappa = 0.5, A7-OL1 improves over A4-OL1 by 0.2086/0.1736/0.0703 loss and 14.769/5.006/9.024 sparsity pp at 14M/70M/410M. | This compares complete recipes, with one seed. Zero-threshold ordering changes at 410M. Absolute loss and same-size A0-relative penalty have different, explicitly reported patterns. |
| Runtime attribution | Native-relative geometric means are 1.2340 for K050, 1.1829 with all skipping disabled and 1.2506 with attention skipping disabled. The incremental sparse factor is 1.0432, helping 14/30 checkpoints; attention-dense is faster on all 30. | These are existing BF16 full-sequence RTX5090 results. The incremental factor divides separately native-normalized speedups. Scalar zero products do not guarantee whole-fragment bypass or profitable execution; no universal scaling law or equal-quality speedup is claimed. |

The corrupt first A1-H log line is retained as a documented source gap: 4,055
NUL bytes make its first boundary unreadable. It lies outside all OL1 records
and the nine selected trajectories; it was not imputed. Across the 54-condition
log inventory, 38,447 of 38,448 expected boundaries are readable.

No additional experiment is needed for the current bounded wording. Independent
matched seeds are the clearest next evidence for the central intervention
contrast. A few additional clipping targets would sharpen exact-sparsity matching;
a matched 410M continuation addresses the separate, higher-effort budget question.
[optional-evidence.md](optional-evidence.md) gives the triggers, minimum scope,
stopping/evaluation rules and how outcomes would change the paper. E02's missing-log
trigger is false. These options were not launched and are not remaining work in
this revision. Authors should review the completeness of the retained AI use
statement before submission.

## Main-text reader check (Q03)

This is an editorial self-check, not an independent reviewer or replication.
Each answer can be obtained from the main text and figures alone:

1. **Why fixed thresholds?** Section 3.1 replaces ranking or per-input threshold
   estimation with a fixed cutoff, letting training determine surviving counts
   and exposing distributional adaptation. No speed advantage over Spark is tested.
2. **Why fixed-budget OL1, and what is protected?** Section 3.1 uses conflict
   projection plus a relative auxiliary-norm cap as a controlled pressure mechanism.
   Once saturated, a larger weight cannot enlarge the ideal fixed-state correction.
   Protection is relative to the adaptive task direction; the appendix qualification
   is explicitly signposted rather than advertising task-loss preservation.
3. **What does reach predict?** Section 3.2 computes structurally affected
   multiplication share from placement, architecture and sequence length before
   training. It guides which operations to target, not attainable quality or latency.
4. **Which pressure effect is conditional?** Figure 3 and Section 4.2 show that
   A4 pressure improves both axes at low thresholds but costs quality at higher
   thresholds. At 0.5, A7 pressure buys 12.096 pp for +0.1265 loss, versus A4's
   2.498 pp for +0.3783; the target-gradient mixture also changes.
5. **How does ReLU change subsequent clipping?** Sections 4.1/4.4 and Figure 5
   distinguish its base cost, smaller incremental clipping loss and a region with
   lower total loss and higher sparsity. Low-target and extreme-target reversals
   prevent interpreting this as universal dominance.
6. **What recurs, and what changes at 410M?** High-threshold A7-OL1 beats A4-OL1
   on both axes at all sizes. At zero threshold, A4-OL1 leads at 14M/70M and
   A7-OL1 leads at 410M. A0 worsens from 70M to 410M while high-threshold A7-OL1
   improves; same-size penalties therefore need separate interpretation.
7. **What do logs support about stopping?** Section 4.4 reports continuing late
   410M A0 improvement and its larger pre-clipping norm. This motivates testing
   a finite-budget explanation, without establishing it as the cause.
8. **Why can more zero products fail to reduce latency?** Figure 6 and Section
   4.5 separate absolute latency from native-relative ratios. Whole fragments
   must be bypassed; zero detection/control costs can outweigh skipped work.

The requested phrase search is recorded in [verification.json](verification.json).
The only exact trigger occurrences in the final PDF are assessed there;
the premature-stopping occurrence explicitly rejects causal attribution, and
the retrospective maximum-sparsity table is in the appendix. There is no
“Cohort means” panel or replacement aggregate bar chart in the active PDF.
Unfavorable endpoints, clipping tails and runtime attribution remain visible.

## Reproduction and verification

The following commands run from the repository root unless noted. The checked
environment was Python 3.12.10, CPU PyTorch 2.11.0, NumPy 2.5.2, Matplotlib 3.11.1
and pytest 9.1.1; PyYAML is also required by evidence imports. Use the repository
layout. LaTeX/BibTeX are provided by MiKTeX; Poppler and global Python with
PyMuPDF provide PDF inspection. No checkpoint weights or GPU are needed for
the existing-record figure generation or mathematical tests.

Regenerate the new figures from the retained derived records:

```powershell
.venv/Scripts/python.exe analyses/020-2026-09-09-intervention-revision/03_figures.py
.venv/Scripts/python.exe analyses/020-2026-09-09-intervention-revision/04_verify_claims.py
```

To rederive those records and the generated dynamics table from original local
logs/manifests, run these first, in order:

```powershell
.venv/Scripts/python.exe analyses/020-2026-09-09-intervention-revision/01_audit_logs.py
.venv/Scripts/python.exe analyses/020-2026-09-09-intervention-revision/02_cross_size.py
```

These two commands additionally need the original attempt directories and
source inventories identified by Analysis 018's `figure_data.json`, plus the
retained Run 030 clipping release. Some raw inputs are ignored local evidence
and will not exist in a fresh Git-only checkout. Rebuilding an audit is distinct
from generating the PDF from checked-in derived data. No missing input is
silently substituted or remeasured. All 336 manifest source references and
36 selected final-checkpoint manifest identities passed reconciliation.

Existing tables/plots were regenerated with the following commands. The original
12 table fragments and eight numerical-display PDFs reproduced byte-for-byte;
source metadata was refreshed where it fingerprints the current manuscript.

```powershell
.venv/Scripts/python.exe analyses/019-2026-09-09-manuscript-rewrite-audit/01_audit_training.py
.venv/Scripts/python.exe analyses/019-2026-09-09-manuscript-rewrite-audit/04_make_tables.py
.venv/Scripts/python.exe analyses/019-2026-09-09-manuscript-rewrite-audit/05_make_figures.py
.venv/Scripts/python.exe analyses/019-2026-09-09-manuscript-rewrite-audit/07_make_condensed_figures.py
```

The first command needs original local evidence. The other three use retained
numerical inputs; see Analysis 019's README. Its prior runtime audit remains the
source for the 3,542-file reconciliation; this revision rechecks the derived
ratios and identities without claiming a new full raw-timing audit.

Focused verification:

```powershell
.venv/Scripts/python.exe -m pytest analyses/020-2026-09-09-intervention-revision/test_geometry.py analyses/018-2026-09-08-results-materials/test_evidence.py tests/test_pressure.py tests/test_ceilings.py tests/test_sites.py -q
```

Result: **67 passed**. This covers OL1 direction/cap/saturation/degeneracy/global
scaling, ideal-versus-stabilized limits, gate boundary derivatives, pooled counts,
site placement, reach/inventory identities and retained evidence consistency.
`04_verify_claims.py` passes **40 quoted-number checks**, plus complete-recipe
ordering, ReLU policy regions, cap counts, checkpoint identities, runtime ratios
and the 96-evaluation figure/omission checks. Original task bytes are unchanged.

Build from `manuscript/draft/`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

A clean-directory build copied only top-level TeX/BibTeX/style files and the
`figures/`/`tables/` directories into `tmp/revision-v2-final-clean/`, then ran the same
commands with no preexisting LaTeX intermediates. All 33 pages match the working
PDF in extracted text and PyMuPDF 1x rendered pixels. PDF bytes may differ due
to build metadata. All figures/tables resolve, all fonts are embedded, no Type 3
fonts occur, and no text lies outside page bounds. There are no undefined
references/citations, multiply defined labels or overfull boxes. Four underfull
vertical-box warnings reflect template page stretching; affected pages were
inspected with no clipping or overlap.

Every page was rendered with Poppler and visually inspected at 1,300-pixel page
height, including legends, threshold/reach guides, omissions, tables and
placement relative to the first reference:

```powershell
& 'C:/Users/thima/AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdftoppm.exe' -scale-to 1300 -png manuscript/draft/main.pdf tmp/revision-v2-render/page
& 'C:/Users/thima/AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdffonts.exe' manuscript/draft/main.pdf
```

Local logs/renders are ignored under `tmp/revision-v2-qa/`,
`tmp/revision-v2-render/` and `tmp/revision-v2-final-clean/`. The tracked
[verification.json](verification.json) records final hashes, label/page inventory,
phrase-search findings and source-copy checks. Historical `revision/` documents
remain records of the preceding rewrite; this directory owns the current task.

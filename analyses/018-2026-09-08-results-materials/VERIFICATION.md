# Verification record ? revised 8 September 2026

## Numerical and source verification

The revised local build completed successfully:

```text
54 trained conditions (30 / 12 / 12)
190 raw clipping evaluations (150 / 20 / 20)
25 matched intervention contrasts
12 analytic topology-by-architecture ceilings
30 included final-kernel checkpoints
8 single-page PDF figures
230 directly hashed source files
```

Raw logical counts reconcile with model/block denominators and recomputed
analytic per-sequence ceilings. All measured endpoints retain complete
validation coverage. Activation pooling reconstructs site totals, zero counts,
threshold counts and RMS from per-layer summaries. Same-size initialization,
order and cache identities, seed 1234, 712 updates and input-token budgets
are checked against actual manifests/configs. Larger clipping values also
match Analysis 011. Evaluation clipping sites are verified as a,m,h,z before
using their reach ceiling. No source-A0 zero denominator is substituted.

The historical h-only A4 cohort is absent from every current numerical result,
including runtime. All final runtime points link to an included trained
checkpoint's canonical sparsity. Candidate and ablation summaries and the
OLS fit are recomputed on the filtered cohort, not copied from the old report.

```powershell
.venv/Scripts/python.exe -m pytest -p no:cacheprovider analyses/018-2026-09-08-results-materials/test_evidence.py tests/test_ceilings.py tests/test_metrics.py -q
```

Result: **38 passed in 1.54 seconds**. Coverage includes count corruption
rejection, band partitioning, frontier dominance/ties, cohort/dose coverage,
matched differences and identities, natural-zero normalization, undefined
unmodified A0, clipping evaluation-site normalization, integer ceiling units,
operation weighting, common-site statistics, scoped transfer claims,
runtime subset arithmetic and saved data/source/artifact hashes.
No model execution or shared scientific-code change was made. There is no
experiment launch, so the full bootstrap launch suite is not applicable.

## Figure and insertion review

Every final PDF was rendered in color and grayscale at 115.2 dpi and visually
inspected. Every page is exactly 396 points (5.5 inches) wide. All text lies
inside its page bounds; ordinary labels are at least 8 pt. Only conventional
math subscript/superscript words (model, arch, max) are smaller, at 6.3 pt.
All figures use embedded vector lettering with selectable text.

The review fixed cropped right-edge ticks in Figures 03/07 and an initially
truncated activation-mass range in Figure 05. Final plotted measurement
ranges contain all selected values. Distinct recipe markers, clipping line
styles and operation hatches retain the intended distinctions in grayscale.
The low-threshold overview cluster remains naturally overlapping; exact values
are in the table. The complete clipping-loss range necessarily compresses
small trained differences in Figure 03.

[FIGURE-REVIEW.md](FIGURE-REVIEW.md) records a criticism, revision and proposed
paper role for all eight figures. The canonical site labels were checked
against the manuscript architecture ladder PDF. One shared activation legend
replaces repeated panel annotations; post-Wo output is not mislabeled as z.

The revised results.tex and both generated compact tables compile in a
temporary 5.5-inch-text-width article wrapper. The second pass produces an
8-page preview with **no LaTeX warnings, unresolved references, overfull or
underfull boxes**. All preview pages were rendered and inspected as a layout
proof. This validates syntax, captions and target-width appearance; it is not
a final conference page-budget or float-placement decision. No manuscript
source was edited or rebuilt.

Temporary wrappers, logs, PNG renders, grayscale renders and contact sheets
remain under ignored tmp/analysis018-revision. Publication outputs are PDF only.
artifact_inventory.json records every generated figure/table's SHA-256 and
byte size. The analysis .gitattributes preserves their inventoried bytes
across Git line-ending conversion.

## Closeout scope

The revision is limited to Analysis 018 and its summary in research/INDEX.md.
Original runs, prior analyses, manuscript, unrelated .gitignore/run-log changes,
and historical archive state remain outside the change. No dataset, weights,
credentials, temporary render or active-run file belongs in the commit.
The completed checklist includes the scoped review and version-control closeout.

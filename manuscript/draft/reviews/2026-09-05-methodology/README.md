# Methodology: adversarial review and revision

User-requested methodology work, 5 September 2026. The same three independent
sub-agents reviewed writing, scientific rigor, and literature/presentation.
Each first audited the old methodology against the operational sources,
then reviewed the new main section and appendix twice. The root agent wrote
and revised the manuscript. No reviewer edited manuscript sources.

## Explicit rubric

Each perspective received five equally weighted criteria scored 1–5:
**1** blocking weakness; **2** major revision; **3** competent but
uncompetitive; **4** strong; **5** exceptional. Reports had to justify scores,
quote precise issues, rank necessary fixes, and propose concrete revisions.
Reviewers were instructed to assess the actual draft independently, not to
assume improvement or pursue a target score. Results and condition-specific
experimental schedules are deliberately outside this drafting stage.

| Perspective | Five criteria | Initial | Second pass | Final |
| --- | --- | ---: | ---: | ---: |
| Writing | Clarity/precision; progression to results; main/appendix economy; notation usability; explanatory payoff | 20/25 | 20/25 | 20/25 |
| Science | Operational fidelity; metric/ceiling precision; intervention identifiability; main/appendix reproducibility; claim rigor | 22/25 | 23/25 | 23/25 |
| Literature/presentation | Citation/definition provenance; main/appendix progression; notation/figure consistency; workload interpretation; reader utility/economy | 21/25 | 21/25 | 22/25 |

These are subjective editorial scores for the methodology, not acceptance
probabilities or evidence for the paper's eventual empirical claims. The
unchanged second literature total reflects improved provenance but a newly
observed figure-legibility issue. Its final score includes a separate visual
follow-up after that issue was corrected. No pooled score is meaningful.

## Reports and source snapshots

- Source audits: [writing](writing-source-audit.md),
  [science](science-source-audit.md), [literature](literature-source-audit.md).
- Initial reviews: [writing](writing-initial.md), [science](science-initial.md),
  [literature](literature-initial.md).
- Second reviews: [writing](writing-revised.md), [science](science-revised.md),
  [literature](literature-revised.md).
- Final figure inspection: [visual follow-up](literature-visual-followup.md).
- [before/](before/) preserves the prior reading copy, active documents,
  original setup artifacts, and historical methodology source.
- [initial/](initial/) preserves the first new methodology/appendix and PDF.
- [after/](after/) preserves final active sources, documentation, and PDFs.
- [source-hashes.json](source-hashes.json) records snapshot SHA-256 hashes;
  [ceiling-verification.json](ceiling-verification.json) records the twelve
  exact numerator/denominator checks against the existing implementation.

## Revisions made

1. **Brief prerequisites for results.** Main text defines interventions,
   quality/sparsity, and architectural reach. Inline gate and metric equations
   remain there; exact site shapes, boundary derivatives, pressure averaging,
   OL1 procedure, attention counters, and ceiling derivation are in the appendix.
2. **Operational fidelity.** The old source's omitted task clipping and
   optional OL1 budget were corrected; its unsupported AMSGrad generalization
   was not copied. Post-gate, equal-tensor pressure averaging is explicit.
   The norm cap is relative to the adaptive task direction, and approximate
   projection does not imply preserved task loss.
3. **Precise measurement.** The new calligraphic S notation preserves the
   existing fractional units and integer-count estimand. The denominator is
   the declared matrix-multiplication workload including the dense head.
   Natural zeros can occur outside the selected-site reach; the opening reach
   sentence no longer suggests otherwise. The V-to-context closure is explained.
4. **Unambiguous scope.** Attention counters are explicitly per block and
   evaluation batch before model aggregation. Local tensor cardinality is
   defined, and its evaluation-batch index no longer conflicts with the
   within-batch index. Initialization matching is explicitly within model size.
5. **Source provenance.** The appendix links to immutable architecture
   configurations, checked against local run records. AdamW's original paper
   was added. No source citation is used as a substitute for local semantics.
6. **Readable setup figure.** L1N is decoded, historical wording removed,
   table caption placed above its table, and the figure placed with the reach
   explanation. Final rendered inspection prompted larger graph labels and
   a narrower ladder source with compact columns. The improved figure remains
   on page 4; values and site assignments are unchanged. A second ceiling chart
   was unnecessary because the ladder already compares all three sizes.

The final small prose pass removed a redundant opening roadmap and replaced
“gates … one-sidedly” with “applies one-sided gates to …”. It changed no method.
The remaining color-only gate markers are a minor accessibility consideration
for the eventual submission layout, not an unresolved scientific definition.

## Verification and scope

The LaTeX/BibTeX reading copy and all three setup PDFs build without undefined
references/citations or overfull/underfull boxes. All 15 citation keys resolve.
All eight final pages were inspected; unchanged page renders were also checked
by hash against the previously inspected pass. Main methodology is approximately
560 words, or about one text page; the entire current main text and setup figure
occupy four pages, references one, and the appendix three in the reading layout.
This is not a final ICLR template-fit claim.

No scientific code was changed, no measured results were regenerated, and no
experiment was launched. Draft sources/review records remain local under the
existing ignore policy. The tracked manuscript index and setup artifacts are
the version-controlled part of this change. The user's supplementary notes and
historical methodology remain unchanged.

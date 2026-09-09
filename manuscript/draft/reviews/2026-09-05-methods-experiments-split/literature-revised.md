# Second split review: revised boundaries and rendered reading path

Independent review, 2026-09-05. Re-read the revised Methodology, Experimental Study, both appendices, and wrapper. Inspected `tmp/pdfs/methods-experiments-revised/page-3.png`, pages 4–5, and pages 8–9. No manuscript edits, new experiments, or fresh empirical interpretation.

## Same rubric and final score

Five equal criteria; anchors remain 1 = blocking, 2 = major revision, 3 = competent, 4 = strong, 5 = exceptional. Maximum 25. The score concerns the requested organization and interpretability, not the acceptance prospects of a completed empirical paper.

| Criterion | First pass | Second pass | Reason |
|---|---:|---:|---|
| Citation/definition provenance | 5 | 5 | Algorithm sources remain with definitions, and architecture/data sources remain with the experimental instantiation. |
| Methodology/experiment boundary | 4 | 5 | Graph inventory and named placement details now belong to the experiment; independent gate/pressure sets are reflected consistently in the objective wording. |
| Notation/figure consistency | 4 | 4 | Relocated references resolve, S units remain consistent, and the readable figure follows the experimental heading. |
| Workload/coverage interpretation | 5 | 5 | Exact counting, structural reach, fixed-token scope, one-seed coverage, and complete validation remain clearly bounded. |
| Utility/economy | 4 | 4 | The compact methods and study introduction give complementary information without adding results or a redundant chart. |
| **Total** | **22/25** | **23/25** | **Requested split succeeds; no further material correction identified.** |

## First-pass issues resolved

The main methods now count over declared transformer-block matrix products. The six-operation Pythia inventory lives with the architecture-specific formula in Appendix C.2. That formula therefore cannot reasonably be mistaken for a universal formula for every Transformer family.

The methods appendix gives the reusable actual-operands rule, while the post-RoPE and context-z examples belong to the Pythia site description in Appendix C.1. The attention counters are explicitly scoped to the declared multi-head attention graph. The general reach definition specifies structural zero guarantees independent of checkpoint values, followed by the separately declared Pythia propagation rules.

Both main and appendix pressure definitions now use executed site activations after a gate when present. This matches the independence of the gate and pressure site sets established in [METHODS](../../../../research/METHODS.md). No new inclusion requirement or scientific implementation change has been introduced.

## Rendered reading path

- **Page 3:** Methodology is self-contained and contains no model names, dataset counts, A* recipes, or diagram-specific ports. Its appendix references resolve to A/A.2 and B/B.2 as intended.
- **Page 4:** Experimental Study begins with its model/data setting and paired-measurement statement, followed by the Pythia graph/ladder and its caption. The reader encounters the empirical heading before the figure. The site/recipe explanation then follows naturally below the caption.
- **Page 5:** The matched-comparison paragraph states fixed-token larger-model coverage, one seed per size, the recipe-comparison limitation, and operation-level interpretation. It is not a result claim.
- **Pages 8–9:** Appendix B's general normalization is visibly separated from Appendix C's experimental setup. The site table, architecture-specific equation, configuration footnote, and MiniPile coverage are readable. No overlapping text, clipped formulas, missing-reference markers, or table-width defects were visible.

Page 5 currently has substantial whitespace because a forced page break puts references after its single paragraph. This is a consequence of the intentionally results-deferred reading copy, not a scientific or section-boundary defect. Do not shrink the now-readable figure to eliminate that whitespace while the results section is still absent.

## Interpretation and provenance checks

The dense language-model head still adds denominator without zero credit; local zero fractions, model product sparsity, structural reach, and runtime remain different quantities. The all-zero ceiling does not become a bound on every naturally sparse observation. The operation-specific comparison language is more precise than trying to attribute a cross-size aggregate difference using the ceiling alone. These meanings remain consistent with [METRICS](../../../../research/METRICS.md).

The experimental source retains the verified configuration footnotes. MiniPile's 500 documents, 338 complete blocks, 1444-token excluded tail, and 2047 loss targets versus 2048 input positions remain in the experimental appendix, consistent with [DATA](../../../../research/DATA.md). Precisions, schedules, and final cohort tables are appropriately deferred rather than invented.

The existing figure is sufficient and remains readable. Its color-only gate encoding is an unchanged minor accessibility consideration; this organizational revision does not warrant a redraw or an additional chart.

No further material source, literature, section-boundary, or presentation correction is requested. Results remain a separate drafting stage.

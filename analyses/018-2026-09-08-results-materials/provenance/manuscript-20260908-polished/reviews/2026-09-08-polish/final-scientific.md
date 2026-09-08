# Final scientific content audit

Reviewed 8 September 2026 after the terminology and contribution-framing
revision. Scope: current abstract, introduction, methodology, experimental
setup, results, conclusion, relevant appendices, and generated table text.
No manuscript, figure, measurement, or protocol file was edited in this pass.

## Verdict

**No new must-fix scientific issues.** The stronger contribution language is
supported by the structured comparisons. The paper correctly derives its
strength from connected observations across interventions and settings,
without treating parameter sweeps as independent training replications or
claiming a general law. Reporting the exact seed and validation protocol once
in the appendix is sufficient; it does not need to dominate the abstract or
each caption.

## Findings

- **The terminology change preserves the estimands.** Activation sparsity
  remains an exact-zero fraction at a named site. Model-wide sparsity remains
  a pooled fraction of zero-activation-operand products in the declared
  workload, including the dense head only in the denominator. The sparsity
  ceiling counts products guaranteed to have a zero operand at 100%
  activation sparsity of the selected sites. The text retains the natural-zero
  and quality-cost distinctions; it does not redefine the ceiling as an
  observed maximum or a quality-preserving bound.
- **The mathematical site-set rename is consistent.** The main definition and
  appendix use the selected nonlinearity set N. The scalar functions G+ and
  G+- remain unchanged; changing their name would serve no mathematical
  purpose. Threshold boundary and derivative conventions remain explicit.
- **The common normalization is preserved.** Main text, scale caption and
  endpoint table use U_arch with the same-size A7 denominator. The equality
  with the block zero-product fraction is still stated correctly. Changing
  the table label from U_A7 does not change any value.
- **The contribution is strong but scoped.** The abstract and introduction
  emphasize 54 matched conditions, conditional pressure effects, different
  distribution responses, the cross-size high-threshold result, and measured
  kernel realization. They do not claim that OL1 is universally better, that
  the three sizes establish a scaling law, or that the internal attention
  response isolates a causal pressure mechanism.
- **The new confidence paragraph is supported.** Matched effects, activation
  distributions and product counts provide a connected interpretation, and
  the high-threshold persistence plus zero-threshold reversal constrain its
  scope. “Agreement across these comparisons” is appropriately descriptive;
  it is not presented as a seed-level confidence interval or an independent
  statistical replication claim.
- **The experimental disclosure remains available.** The protocol appendix
  states seed 1234, one training seed per condition, parameter sweeps as
  intervention variability, and reuse of the same validation split without
  separate confirmation. The main text still names matched initialization,
  order, budget, final-checkpoint evaluation, selected larger-model recipes,
  and the lower 410M learning rate.
- **The systems claim remains calibrated.** The reported 1.234x K050 result,
  1.043x sparse-path ratio, and detrimental attention skipping retain the
  declared workload and controls. “Best” is restricted to searched proposals;
  the faster attention-dense ablation remains visible. The conclusion does
  not claim cached-decoding or hardware transfer.
- **The overview attribution is repaired.** The comparison now says that both
  trained representation and affected computation change. It no longer
  implies that the A0-clipping versus A7-OL1 difference has been causally
  partitioned into learning and site-coverage contributions.

## Verification

The retained Analysis 018 evidence still has SHA256
`76a2b2d52944cb4aa652a1c4a2d2b4250fc4e478291c05a7851f40004a020144`.
The numeric bodies of all seven manuscript tables are identical to their
retained analysis-owned tables: 54 endpoints, 29 paired effects, compact and
full scale contrasts, density masses, operation contributions, and kernel
ablations. This confirms that the terminology changes did not alter the
tabulated results. Earlier source-linked numerical and protocol audits are
recorded in `scientific-reviewer.md`.

## Optional precision refinement

The pressure definition says “after its selected nonlinearity,” although the
general formulation permits pressure at sites outside the selected
nonlinearity set. “After the nonlinearity, where one is applied” would remove
that small ambiguity while retaining the executed post-transformation
activation objective. All reported pressure recipes use the selected sites,
so this does not change or invalidate any current result.

This review covers scientific content. Final PDF layout, protected-paragraph
byte identity and the complete build remain the root agent's verification
tasks; no new experiment is required by these findings.

# First split review: definition provenance and section boundaries

Independent adversarial source review, 2026-09-05. Read `methodology.tex`, `experimental-study.tex`, both new appendices, `main.tex`, and `experimental-notes.md`. No active manuscript edits. The current split PDF was compiling during this source review; the unchanged figure artwork had already passed a prior bounded visual check.

## Rubric and score

Five equal criteria. Anchors: 1 = blocking, 2 = major revision, 3 = competent, 4 = strong, 5 = exceptional. Maximum 25. Scores assess this section split, not acceptance prospects for the unfinished paper.

| Criterion | Score | Assessment |
|---|---:|---|
| Citation/definition provenance | 5/5 | Architecture/data citations and immutable configuration sources move with their instantiation; algorithm provenance is retained. |
| Methodology/experiment boundary | 4/5 | The main split is effective, but the methods retain a graph-specific operation inventory and two placement-specific sentences. |
| Notation/figure consistency | 4/5 | S notation, labels, and relocated figure source are consistent; current PDF placement remains to be checked. |
| Workload/coverage interpretation | 5/5 | The scalar unit, causal scope, head denominator, validation coverage, one-seed scope, and partial ladder coverage remain explicit. |
| Utility/economy | 4/5 | The new experimental introduction explains the instantiated question and comparisons without adding results or a long setup table. |
| **Total** | **22/25** | **Strong split with two focused boundary/wording fixes.** |

## What works

Pythia/MiniPile, A* labels, exact ports, matched comparisons, and the combined figure now belong to Experimental Study. The text distinguishes separate pretraining conditions from a curriculum, detailed 14M coverage from selected larger recipes, and thresholds from independent replicates. It does not claim transfer succeeded.

The Pythia denominator is now explicitly in an architecture-specific appendix. Its dimension tuples and immutable sources remain attached to that calculation. MiniPile packing and exact target coverage belong to the experimental appendix. The methodological equations retain their units and the dense-head/no-zero-credit convention; neither zero fractions nor selected-site reach become runtime claims.

The A4/A7 combined-change limitation is preserved, and the zero-threshold identity statement is correctly contextualized as a property of the instantiated recipes. The equal-tensor objective normalization remains visible. The source record in [Analysis 013](../../../../analyses/013-2026-09-04-matched-intervention-manuscript/README.md) supports these interpretation limits.

## Ranked concrete fixes

### 1. Finish removing graph-specific inventory from Methodology

The main definition still lists the blocks' “QKV, attention-score, probability--value, attention-output, and two FFN matrix products.” This inventory belongs to the selected graph; in particular, two FFN products are not a universal feature of Transformer implementations. The later phrase “declared multiplication workload” limits the claim but leaves unnecessary dependence on the experimental architecture inside the reusable definition.

Define N0 in the main methods as the zero-activation-operand count over the **declared block matrix products**, with Nmodel adding the corresponding dense counts and the dense head. Leave the six-family inventory in the architecture-specific appendix, where its formula already enumerates those products. This changes presentation, not the operational metric. See [METRICS](../../../../research/METRICS.md).

Likewise, the generic counter appendix retains the sentences about “actual post-RoPE operands” and “actual post-gate context z.” Move those placement-specific sentences beside the Pythia site table, or replace them in Methodology with the general rule to count the actual operands entering each multiplication after all preceding transformations. The attention OR counters can stay, explicitly scoped to the stated causal multi-head attention graph.

### 2. Make independent pressure targets explicit in the measurement wording

The methods say the pressure target set is independent of the gate set, then describe all pressure activations as “measured after gating.” The executed study's pressure-bearing rows use gated targets, but the reusable definition should not appear to require that inclusion relation.

Prefer “measured at the executed site, after its gate when one is present,” and make the same clarification in the methods appendix's “post-gate activations” sentence. This is consistent with independent fields in [METHODS](../../../../research/METHODS.md), including unselected ports retaining their existing transformations. It requires no new implementation or experiment.

## Cross-reference audit

The inspected references have valid targets in the new split:

- Methodology's `app:interventions`, `app:pressure`, `app:accounting`, and `app:ceiling` point to generic definitions/counters.
- Experimental Study's `app:experiment` and `fig:intervention-ladder` point to its setup appendix and contained figure.
- The generic appendix scopes its attention example with `app:sites` and links the specialization through `app:architecture-counts`.
- The moved site table retains `tab:sites`; recipe assignments point to `sec:experiments`, replacing the obsolete methodology reference.
- The moved denominator retains `eq:denominator` and refers back to `sec:measurement` for the established convention.

No stale reference to the removed main `sec:reach` subsection was present in the inspected active sources. The wrapper includes Methodology before Experimental Study and both appendices in the intended order. Confirm the compiled PDF puts the experimental heading before the relocated figure.

## Optional scope polish

The evidence notes record fixed token exposure and differing learning rates across sizes, while the main experimental section only states matching within each size. That is accurate, not a false claim of identical cross-size training. At the results stage, identify these differences before interpreting scale transfer; no new schedule is needed in this introductory split.

The general reach definition should continue to mean reach under **declared exact propagation rules**, not an unspecified process for discovering every possible graph simplification. The Pythia appendix already supplies the specific closure and exclusions. Preserve those when making the prose generic.

## Figure decision and next pass

The existing graph/ladder now has the correct section owner and should remain the sole architecture-ceiling display. No new chart or redraw is justified by this split. The second pass should verify the two wording/boundary fixes and actual heading/figure placement, without expanding scope into results or implementation changes.

# Presentation audit for the methods/experimental-study split

Independent source audit, 2026-09-05. Read current `methodology.tex`, `methodology-appendix.tex`, `main.tex`, and the introduction. No active manuscript edits. The recommendations below concern relocation and cross-reference repair, not a new experiment or a broader methodological claim.

Two later scored reviews will use five equal criteria: citation/definition provenance; methodology/experiment boundary; notation/figure consistency; workload/coverage interpretation; utility/economy. Anchors are 1 = blocking, 2 = major revision, 3 = competent, 4 = strong, 5 = exceptional; total 25. No score is assigned to a split that has not yet been written.

## Minimum main-text change

Keep Methodology about what the interventions and measurements mean. Introduce a site as a declared tensor port in a forward graph, with separately specified gate and pressure site sets. Retain the two fixed-threshold operators, ReLU, the equal-tensor L1 objective, naive L1/OL1 distinction, exact-zero aggregation, the scalar-multiplication unit, model sparsity, and selected-site reach. Preserve the distinction between natural zeros and the reach ceiling, and between logical counts and execution timing. These are interpretive definitions, not setup details.

Move these existing passages into a separate Experimental Study section:

| Current material | Destination/reason |
|---|---|
| Pythia-family random initialization and MiniPile opening | Experimental Study opening: specifies the instantiated model/data setting. |
| a/m/h/z/q/k/v list; post-RoPE placement; replacement of GELU at h | Experimental Study's Pythia implementation, with detailed ports in its appendix. |
| A0/A1-H/A4/A7 recipe descriptions and mixed gate assignments | Experimental Study: these are chosen conditions, not properties of the reusable gate operators. |
| Within-size matching and A4-OL1/A7-OL1 combined-change limitation | Experimental Study's comparison design. |
| h-only and v-to-output reach examples; figure-based comparison of model sizes | Experimental Study: applies the reach definition to this graph. |
| Combined Pythia graph/ladder figure | Experimental Study, immediately after its first substantive introduction. |

The new section needs only a concise setting paragraph, recipe paragraph, and matched-comparison interpretation paragraph, plus the existing figure. Results remain deferred. Do not turn the relocation into an expanded experiment inventory, a new schedule, or a fresh claim of transfer success.

The introduction may retain its Pythia/ladder roadmap. A roadmap can identify the empirical vehicle before the formal Methodology section without blurring the methods/experiments boundary.

## Appendix split with minimal duplication

Keep a methods appendix with reusable gate semantics, objective averaging, OL1's exact update, pooled local zeros, and linear/attention product counters. The reference clipping norm of one is a setting and should leave the generic algorithm description; the algorithm itself can take a clipping threshold as a parameter. Float32 evaluation of the pressure objective can remain as an implementation property of the defined method if that remains the intended contract.

Move the Pythia port table, A* identities, specific dimensional denominator, named-site reach mapping, A* ceiling numerators, pinned architecture tuples/configuration links, and complete MiniPile validation coverage to an experimental appendix. Preserve their exact semantics and existing provenance. In particular:

- The formula `T(4d^2+2*d*d_f)+d*T*(T+1)` assumes the current equal-width multi-head attention and two-projection FFN graph. Removing the word Pythia does not make it universal for arbitrary Transformers. Keep it explicitly scoped to this study's graph.
- The direct linear count `N=npq` and OR-based QK/PV counters are reusable mathematical rules. They can remain in the methods appendix, with their causal/full-sequence domain stated.
- The numerator/denominator definition, dense-head convention, integer pooling, and selected-site/natural-zero caveat remain in Methodology. The chosen architecture dimensions, exact evaluated T, number of validation blocks, and source-cache coverage belong to the experimental instantiation.
- Do not relocate the MiniPile counts into a generic “measurement contract” that makes them look inherent to the sparsity metric.

The scientific sources remain [METHODS](../../../../research/METHODS.md), [METRICS](../../../../research/METRICS.md), [DATA](../../../../research/DATA.md), and the run-linked architecture pins already checked. This split should not modify the underlying scientific contract.

## Labels and reading order

Keep labels attached to moved objects whenever their meaning is unchanged: `fig:intervention-ladder`, `tab:sites`, and the denominator/counter equation labels can remain stable. Add an Experimental Study section label and a distinct label for its recipe paragraph/subsection if the methods appendix needs a precise reference. Repair the current appendix's A4/A7 reference to `sec:interventions`: after the split, that reference should point to the experimental recipe definition.

Methodology should cite the methods appendix for gate derivatives, OL1, and generic counters. References promising “exact placements” or “coverage” should point to the experimental appendix. If `app:ceiling` remains attached to the Pythia specialization, the main methods text must identify that link as an application rather than as a general derivation.

The wrapper should read introduction → related work → methodology → experimental study → references → appendices. Put the figure declaration in the experimental-study source so future edits do not silently return it to the end of Methodology. Inspect the rendering to ensure the experimental heading appears before the figure; a float must not visually erase the newly requested boundary.

Update the reading-copy subtitle and provenance notes to name the Experimental Study section. This is ordinary document maintenance; historical drafts, raw R-field artifact keys, and prior review snapshots should remain unchanged.

## Figure and citation boundaries

Preserve the now-readable figure artwork. Its title already identifies Pythia, its caption distinguishes candidate recipes from empirical coverage, and its ceiling columns are explicitly analytic. No additional diagram or ceiling chart is needed. Introducing a second “generic” graph would add work and invite an unsupported claim of architectural generality.

Pythia and MiniPile citations belong with the experimental setting; AdamW and limited gradient-projection context remain with algorithm definitions. Architecture configuration footnotes should move with the instantiated ceiling calculation. Do not strip citation provenance merely to make methods appear more generic.

## Acceptance condition for the first split draft

A reader finishing Methodology should understand how to specify an intervention and how S/Smax are calculated for a declared forward workload, without yet needing Pythia, MiniPile, or A* labels. A reader finishing Experimental Study should understand how those definitions are instantiated here and which matched comparisons do or do not isolate an intervention. Neither section should imply that candidate recipes are a full factorial design or that the pending results have already established a mechanism.

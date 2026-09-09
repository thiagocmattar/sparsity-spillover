# Second scored writing review: methods / experimental-study split

Re-read the actual revised methodology, experimental study, and their appendices. Inspected rendered pages 3, 4, and 5 in `tmp/pdfs/methods-experiments-revised/`, covering the main definitions, the experimental opening and combined figure, and the comparison paragraph. No active manuscript files were edited.

**Overall: 20/25.** The requested separation is clear and the wording is more precise. The initial highest-value suggestion—making the explanatory target explicit—is resolved. The draft is ready for results development; no substantive writing issue requires another structural revision. Scores remain in the strong band rather than increasing automatically after local fixes.

Five equally weighted criteria; anchors: **1** blocking weakness; **2** major revision; **3** competent; **4** strong; **5** exceptional.

| Criterion | Initial | Revised | Evidence |
| --- | ---: | ---: | --- |
| Clarity and precision | 4/5 | 4/5 | The opening now names “pressure's effects on quality and model-wide sparsity.” Pressure can target ungated sites, and the ceiling's zeros are “guaranteed” by the structural intervention. The named selected cohort and one-seed scope remain explicit. |
| Section separation | 4/5 | 4/5 | The general Methodology no longer enumerates the Pythia operation inventory. Pythia site mapping, counts, gate recipes, and evaluation details belong to the experimental section or its appendix. General counting definitions refer to the declared graph without pretending to establish a universal transformer denominator. |
| Question-driven progression | 4/5 | 4/5 | The opening states the empirical question; the figure makes the alternatives visible; the comparison paragraph explains pressure contrasts, threshold dependence, placement, and the larger-model scope. The final sentence now explicitly invokes operation-level interpretation. |
| Economy | 4/5 | 4/5 | The main Methodology remains compact and the experimental body is roughly three hundred words. The combined figure does useful work and is placed immediately after the setup paragraph. There is no renewed optimization detour or premature schedule inventory. |
| Reader payoff | 4/5 | 4/5 | A reader can identify the intervention, understand the measurement, read the ladder as separate pretraining conditions, and judge what each matched contrast establishes. Architectural reach and per-operation responses are now clearly part of cross-size interpretation. |
| **Total** | **20/25** | **20/25** | Strong and ready for the next drafting stage. |

## Resolved issues and successful changes

- The vague phrase “resulting account” has been replaced by an explicit explanation of quality and model-wide sparsity effects.
- “Taken after its gate when one is present” preserves the independence of pressure and gate target sets.
- The structural-guarantee wording for selected-site reach prevents the ceiling from accidentally including checkpoint-specific or natural zeros.
- The operation inventory has moved out of the compact general definitions and into the concrete architectural specialization.
- The larger-model comparisons are explicitly described as having a fixed token budget, while initialization matching remains within size.
- The final paragraph explains why the aggregate sparsity metric should be read together with operation-level responses and architectural weighting.

## Outstanding substantive issues

None identified in this writing review. Results, exact schedules, and condition-level reproducibility details remain intentionally deferred. Their absence does not undermine the present separation of definitions and experimental setup.

## Optional final wording polish

1. **Unpack “fixed-token” if making a final prose pass.** “The fixed-token 70M and 410M extensions” is compact but slightly telegraphic. Stating that the extensions use a matched training-token budget would make the control easier to read. Keep the precise matching scope supported by the reported cohort; do not imply identical learning rates or a universal protocol across model sizes.

2. **The closing sentence could name the two quantities being separated more directly.** The current wording—“operation-level zero counts and architectural reach distinguish the measured response at each operation from changes in how those operations contribute to the aggregate sparsity metric”—is accurate but abstract. A reader-facing alternative, if it matches the later analysis, is: “Across sizes, we examine each operation's zero fraction alongside its share of the multiplication workload, separating changes within operations from changes in their contribution to aggregate sparsity.” The existing count-first methodology supplies these fractions. This is an optional clarity edit, not a request for a new analysis.

Neither edit is necessary before proceeding. Keep the existing short section structure and avoid filling the remaining setup space with generic details.

## Reading-copy observations

- Page 3 holds the main Methodology with clear subsection boundaries. The inline gates and sparsity ratios remain legible and unbroken by clipping or equation overlap.
- Page 4 starts Experimental Study with its question and setup, followed by the combined architecture/ladder figure and recipe explanation. The figure's placement supports immediate interpretation of the prose.
- The comparison paragraph continues on page 5. The substantial remaining space on that page reflects the intentional pause before results and the reading wrapper's bibliography break. It should not prompt artificial content or last-minute layout compression at this stage; later results will determine final page flow.
- These observations cover the inspected pages only and do not certify an ICLR template or all appendix pages.

## Recommendation

Preserve this structure and move to the evidence-backed result narrative. The main definitions and setup now provide the required vocabulary, controls, coverage boundaries, and interpretation rules without competing with the study's scientific argument.

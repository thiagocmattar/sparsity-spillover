# Initial scored writing review: methods / experimental-study split

Reviewed `methodology.tex`, `experimental-study.tex`, both corresponding appendices, `main.tex`, and `experimental-notes.md`. This is a source review; the compiled reading copy was still being prepared. No active manuscript files were edited. The absence of results and condition-specific schedules is intentional and is not penalized.

**Overall: 20/25.** The split is successful. Methodology now defines the controlled choices and measurements, while Experimental Study tells readers where those definitions are instantiated and what the comparisons can establish. No structural rewrite is necessary before drafting results. The remaining suggestions concern explanatory wording and small repetitions.

Five equally weighted criteria; anchors: **1** blocking weakness; **2** major revision; **3** competent; **4** strong; **5** exceptional.

| Criterion | Score | Evidence |
| --- | ---: | --- |
| Clarity and precision | 4/5 | “These are separately pretrained conditions, not stages of a training curriculum” resolves a plausible misreading of the ladder. Matching is explicitly within size, and A4-OL1/A7-OL1 is explicitly a joint change. The opening's “resulting account” remains slightly opaque before the specific quantities are named. |
| Section separation | 4/5 | Pythia/MiniPile, A* labels, named site placements, coverage, and the combined figure are now in Experimental Study. General gates, pressure, pooled zero counts, and selected-site reach remain in Methodology. Architecture-specific denominator and ceiling algebra have their own experimental appendix, with cross-references from the general accounting. |
| Question-driven progression | 4/5 | The experimental section opens with the pressure–threshold–placement question and later specifies exactly how repeated pressure contrasts and broader placement comparisons address it. The selected larger-model cohort is distinguished from detailed 14M work. This supports the intended sequence from individual/joint changes to explanatory transfer. |
| Economy | 4/5 | Approximately 373 methodology words and 300 experimental-body words are appropriate for the requested role. Two methods subsections and three experimental paragraphs are enough. The short caption carries legend and scope information; full schedules are sensibly deferred. Repeated reach/coverage wording could be compressed slightly. |
| Reader payoff | 4/5 | The reader can now interpret A0/A1-H/A4/A7, understand the separate training conditions, identify which comparisons isolate pressure, and avoid attributing the joint pressured contrast solely to gate placement. The final cross-size sentence ties the interpretation to architectural reach. |
| **Total** | **20/25** | Strong, compact prerequisites for the results stage. |

## Ranked revision opportunities

1. **Make the opening's explanatory target concrete.** “We examine how pressure's additional effect depends on thresholding and placement, and whether the resulting account transfers across model sizes.” The possessive “pressure's” is serviceable, but “resulting account” could describe any analysis. Readers should know that the account concerns quality and sparsity changes.

   Suggested rewrite: “We examine how the effect of pressure on quality and model-wide sparsity depends on thresholding and placement, and whether the same explanation remains informative across model sizes.” This does not claim that an explanation has already been established.

2. **Consider integrating the one-seed limitation into the comparison sentence.** The present sentence—“Comparisons use one initialization seed per size; thresholds are treatment levels, not statistical replicates”—is scientifically valuable and should remain in the main study. It slightly interrupts a paragraph otherwise organized around what each contrast tests.

   Optional rewrite: “These are paired comparisons for one initialization seed per size; the threshold sweep supplies treatment levels rather than independent replications.” Retain the qualification, regardless of the exact wording.

3. **Trim one repetition of selected coverage if page pressure develops.** Paragraph one says the larger models cover selected recipes; paragraph three names them; the caption says coverage is reported with results. The first statement usefully previews scope and the third-paragraph list is concrete. The caption can therefore say “Rows define candidate recipes” without another coverage sentence, or leave the caption as it is for standalone readability. This is a trade-off, not a mandatory cut.

4. **Keep the reach concept intuitive despite removing the earlier example.** The compact Methodology now defines the ceiling without the h-versus-v illustration. This is acceptable because the figure and experimental appendix supply the mapping. If future reader feedback finds “reach” abstract, restore one short example in Experimental Study near the figure rather than expanding the general methods or duplicating the table.

No item above is a blocking defect. The first is the highest-value improvement; the remainder are small editorial choices. There is no need to add a third methodology subsection, an experimental-design table, or a longer optimization discussion.

## Interpretation readiness

The section makes the limits of the expected results sufficiently clear at this stage:

- Pressure contrasts hold the gate configuration fixed.
- Repeating those contrasts across thresholds probes dependence on thresholding.
- A4/A7 without pressure probes the expanded gate set.
- A4-OL1/A7-OL1 changes both sets and compares recipes.
- The larger-model extensions cover a named subset and do not imply a factorial replication.
- One-seed comparisons and threshold sweeps do not provide independent statistical replications.
- Cross-size changes in aggregate sparsity must be interpreted with changes in architectural reach.

These points provide a useful basis for the results narrative without asserting findings. The later setup must state the exact cohort protocols, but adding them now would obscure the separation the user requested.

## Appendix organization

The new division is sensible. The methodological appendix defines gate boundaries, tensor weighting, OL1 geometry, generic zero counting, and normalization. The experimental appendix owns the concrete graph, site shapes, pinned Pythia configurations, architecture-specific reach, and validation packing. Cross-references make the dependence on a specified attention graph explicit. The improved e/b distinction in the local versus attention counts also resolves the previous review's optional notation point.

## Recommendation

Make the first sentence more explicit if convenient, then proceed to the results stage after the other independent reviews. The structure should now remain stable; repeated rearrangement would offer diminishing returns relative to developing the evidence-backed result argument.

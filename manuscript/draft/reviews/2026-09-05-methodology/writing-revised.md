# Revised methodology review: scientific writing

Re-read the revised methodology and appendix against the initial review. Inspected rendered pages 3, 4, 6, 7, and 8 in `tmp/pdfs/methodology-revised/`, covering the main methodology, ladder, and complete appendix. No manuscript files were edited.

**Overall: 20/25, with the initial mandatory writing fixes resolved.** The integer score remains unchanged because all five dimensions were already in the “strong” band; correcting local ambiguities does not by itself make the section exceptional. This version is ready to support drafting the results and condition-specific setup. No outstanding substantive writing issue requires another methodology rewrite.

Scoring anchors: **1** blocking weakness; **2** major revision; **3** competent but uncompetitive; **4** strong conference prose; **5** exceptional. All criteria are equally weighted.

| Criterion | Initial | Revised | Evidence in the revised text |
| --- | ---: | ---: | --- |
| Clarity and precision | 4/5 | 4/5 | “To characterize the computational reach of a gate-site set” correctly introduces the ceiling without claiming that only selected sites can produce zeros. The OL1 cap now explicitly references the adaptive task direction. “Within each model size” makes the matching scope unambiguous. |
| Progression toward results | 4/5 | 4/5 | Condition labels and gate behavior lead naturally to paired validation quality and sparsity, followed by the architectural ceiling needed for cross-size interpretation. The main text tells readers how to interpret A4/A7 recipe differences before they see results. |
| Main/appendix economy | 4/5 | 4/5 | The main methods remain compact and use three useful headings. Detailed pressure geometry, product multiplicities, config links, and validation coverage remain in the appendix. Adding the context-closure explanation does more explanatory work than its small word cost. |
| Notation usability | 4/5 | 4/5 | The meaning of tensor bars in the local-count denominator is now stated. The calligraphic sparsity quantities, gate and pressure sets, and shared denominator are legible in the rendering. Inline gate and ratio equations fit naturally into the prose. |
| Explanatory payoff | 4/5 | 4/5 | The value-gate example now explains its second reached operation: “since zero values produce a zero context.” The natural-zero and dense-head conventions support correct interpretation rather than functioning as detached caveats. |
| **Total** | **20/25** | **20/25** | Strong, usable main methodology with a sufficiently precise appendix. |

## Resolution of initial mandatory items

1. **Reach exclusivity:** resolved. The new opening defines intervention-conditioned reach rather than claiming universal control of zero operands.
2. **OL1 norm reference:** resolved. The reference direction is explicit in the main text and the appendix retains the precise ratio.
3. **Tensor element count:** resolved. The denominator's bars now explicitly denote the number of elements.

The additional within-size matching statement, per-block attention-count scope, immutable configuration links, and removal of historical “corrected” language also improve clarity and self-contained provenance.

## Remaining optional polish, ranked

1. **Disambiguate the two meanings of batch index b in the appendix if making another small edit.** The local-site fraction explicitly uses b for evaluation batches, while the attention-product equations use b for examples inside one evaluation batch. The newly stated local scope makes the equations understandable, but distinct indices would be cleaner. The smallest notation change is to use e for evaluation-batch indexing in the local-site fraction and reserve b for examples within a batch. This does not affect the main text or its interpretation.
2. **“One-sidedly” remains awkward.** “A4 applies one-sided gates to…” is more natural than “A4 gates … one-sidedly.” This is optional sentence polish, not a scientific ambiguity.
3. **The opening's organizing sentence is expendable if later page pressure requires compression.** “The methodology separates what an intervention changes…” repeats the introduction and is already demonstrated by the three headings. Removing it would save space without sacrificing a definition.

No additional optimization paragraph, general forward-graph derivation, or experimental-configuration inventory should be added to the main text. The remaining work belongs to the planned results/setup stage.

## Rendering observations

The reviewed pages show no clipped prose, equation overlaps, or unreadable inline formulas. The site table's caption appears above it and its rows are easy to scan. The main gate formulas remain compact; the longer projection and denominator definitions appropriately use displayed equations in the appendix. The ladder fits on the same page as the architectural-reach discussion and its caption explains L1N and the A4/A7 pressure coverage. These are observations on the inspected reading-copy pages, not a claim of ICLR template fit.

## Final writing judgment

The revised methodology gives the reader the prerequisites needed for the expected results: what changes between conditions, how zeros are weighted, what the ceiling means, and which comparisons isolate an intervention. It does so without turning OL1 or counting algebra into a competing narrative. Proceed to the next drafting stage after the independent scientific/implementation review is reconciled.

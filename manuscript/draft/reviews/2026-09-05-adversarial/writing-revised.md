# Independent writing re-review

Reviewed the revised `introduction.tex` and `related-work.tex` against the original sources preserved in `before/`, then re-read the latest science/literature refinements (separate learned sparsity from reach; explicitly locate internal attention operands in this study; expand the batching, Block Q-Sparse, and TwELL discussion). The scores below assess that latest version. This is an assessment of the argument-first stage, not a judgment that a results-free manuscript is ready for submission. No manuscript files were edited.

Rubric: **1** blocking weakness; **2** major revision; **3** competent but uncompetitive; **4** strong conference prose; **5** exceptional.

| Dimension | Original | Revised | Evidence for revised score |
| --- | ---: | ---: | --- |
| Clarity and precise wording | 3/5 | 4/5 | The central distinction now separates experimental control from separability of effects. The explanation distinguishes changes in magnitudes, exact zeros, and affected computation. Some abstract phrases remain, particularly “conditional value,” “additional settings,” and “computational value.” |
| Argument progression | 3/5 | 4/5 | Motivation now names actual decisions, the empirical gap leads to the intervention choices, and accounting answers an interpretive problem. The two related-work subsections form coherent cumulative chains. The ending still repeats rather than advances the argument. |
| Concrete reader payoff | 2/5 | 4/5 | Introduction 15–17 names recipe selection, training allocation, and sparse-kernel development. Lines 49–53 supply a useful diagnostic payoff. The prospective novel contribution is much clearer, although “interpretable and comparable” in the ending remains generic. |
| Economy | 2/5 | 3/5 | Detailed initialization, parameter counts, dataset, and denominator bookkeeping are removed. However, the introduction repeatedly restates the three interventions, their additional effects, and their decision value across paragraphs 1, 2, 3, 4, and 6. |
| Persuasive, calibrated voice | 3/5 | 3/5 | The draft sells concrete relevance without inventing findings. “The scientific value lies in” and “An explanation should also account for” still sound like an internal justification or proposal. Stronger claims are unnecessary; a firmer, less repetitive ending is needed. |
| **Total** | **13/25** | **18/25** | Substantive argument improvement, with a final compression and ending pass still worthwhile. |

## Highest-leverage remaining changes

1. **Tighten the final introduction paragraph.** Lines 71–80 restate the comparison, explanation, and payoff at length. In particular, “The scientific value lies in making sparsification recipes interpretable and comparable” is metacommentary, and “These distinctions provide a basis for adapting…” is a generic closing sentence. Keep the ladder and one-sentence Pythia setup, then end on a concrete transfer criterion: the explanation must remain informative when model size changes both the learned response and the computation reachable from a site. This is stronger than announcing that the study has scientific value and does not require inventing results.

2. **Remove one full repetition of the intervention question.** Introduction 31–39 already defines the choices and states how their individual and joint effects are studied. Lines 40–41 (“This makes the additional benefit or cost …”) add little. Delete that sentence or use the space only for a genuinely necessary distinction. Keep the user's controlling thesis at lines 35–36.

3. **Consider reducing the accounting paragraph by one sentence.** The final explanatory sentences overlap: the same local zero rate can affect different computation as architecture changes, and learned sparsity should be distinguished from the work reachable through the chosen sites. The latest refinement correctly avoids implying that observed sparsity is utilization of the reach ceiling. Preserve that precision if compressing; a small amount of repetition here is preferable to conflating the two quantities.

4. **Make the opening slightly more literal.** “Making computation depend on which features an input uses” (lines 7–8) is readable but imprecise about what “uses” means. The next sentence gives the actual mechanism. A cleaner opening is: “Activation sparsity can reduce language-model inference cost by letting specialized kernels skip products and weight accesses associated with zero activations.” The reduction in rhetoric is offset by immediate technical clarity.

5. **Keep the inference subsection's stronger detail, but trim its final generality.** The first paragraph now explains the systems trade-off with concrete overheads and workload distinctions; it is a major improvement. The final sentence, “Connecting intervention studies to these measurements can reveal…,” partly repeats the preceding requirement for complete-model timing. End with the specific test of whether useful intervention-induced zeros survive sparse-execution overhead, without adding another general promise about computational value.

## Assessment of the user's clarification

The new statement—separately controllable interventions need not have separable effects—is now the organizing idea, not an afterthought. This resolves the ambiguity of the original “three separable interventions.” The draft also distinguishes individual effects, interactions, and affected computation. Preserve that exact conceptual distinction during compression.

## Related-work structure

The requested structure is met: **Sparsification interventions has exactly three paragraphs; Inference speedup has exactly two.** The dedicated gradient paragraph is gone. The chain is now: existing sparse tolerance → learning sparse representations → broader placement and conditional effects; demonstrated acceleration → workload-dependent adaptation and measurement. The latest inference paragraph is stronger still: batching, Block Q-Sparse, and TwELL explain what adaptation depends on rather than merely asserting hardware dependence. Agentic kernels now occupy an appropriately brief supporting role rather than a competing contribution storyline. The added specificity is useful and should survive the final compression pass.

## Scope and calibration

Do not respond to the remaining proposal-like tone by asserting a discovered mechanism or transfer result. The appropriate editorial improvement is to state the explanatory test and its consequences more crisply. When the results stage is authorized, the final paragraph will need an evidence-backed result preview. Its current absence is intentional and is not itself a fault scored here.

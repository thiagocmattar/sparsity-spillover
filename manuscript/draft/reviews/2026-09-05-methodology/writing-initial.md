# Initial methodology review: scientific writing

Reviewed `methodology.tex`, `methodology-appendix.tex`, `main.tex`, `introduction.tex`, and `methodology-notes.md`. This is a source-based review of the reader prerequisites for the later results. Condition-specific experimental setup and empirical results are deliberately deferred and are not treated as missing methods content. No manuscript files were edited and no claim of visual PDF verification is made here.

**Overall: 20/25.** The new methodology is substantially more usable than the old source. It has a clear path from intervention labels to quality/sparsity axes to the meaning of the ceiling. The main text contains the right amount of formalism, while the appendix owns details needed to reproduce or audit the definitions. The remaining issues are concentrated, fixable wording and notation ambiguities rather than a structural problem.

Scoring anchors: **1** blocking weakness; **2** major revision; **3** competent but uncompetitive; **4** strong conference prose; **5** exceptional. All five criteria have equal weight.

| Criterion | Score | Evidence |
| --- | ---: | --- |
| Clarity and precision | 4/5 | “At $\kappa=0$, their forward maps are ReLU and the identity, respectively” makes the gate distinction intelligible without pushing derivatives into the main text. “The head receives no zero credit” is unusually clear accounting language. Two phrases need tightening: “relative norm” leaves its reference implicit, and “which operations can receive zero operands” overstates selected-site exclusivity. |
| Progression toward results | 4/5 | The order—interventions, quality/model-wide sparsity, architectural reach—lets a reader interpret both condition labels and plot axes. The A4-OL1/A7-OL1 qualification prevents a misleading interpretation of later contrasts. The appendix supplies a full explanation of why local value sparsity does not determine probability–value sparsity. |
| Main/appendix economy | 4/5 | Three short subsections keep the main section near the requested size. The full graph, repeated topology table, and AdamW algebra are absent from the main text. OL1 takes one compact paragraph. No additional subsection is warranted. The final sentence of the opening and some scope phrases could be trimmed, but the current density is defensible. |
| Notation usability | 4/5 | Calligraphic S is reserved for sparsity; gate and pressure sets use G/P; the two main ratios are inline and share an explicit denominator. The site notation matches the ladder. The appendix's $|A|$ denominator is ambiguous because bars already mean absolute values in the pressure definition. |
| Explanatory payoff | 4/5 | The method explains why the same local sparsity can mean different computational opportunity, why a value gate reaches more than one operation, and why recipe comparisons do not automatically isolate gate effects. The ceiling example would be even more intuitive if the V-to-context closure were stated directly in its main-text sentence. |
| **Total** | **20/25** | Strong draft with a short precision pass needed. |

## Ranked must-fix items

1. **Remove the suggestion that only selected sites can cause zero operands.** Main methodology line 68 says, “The selected sites determine which operations can receive zero operands.” Natural zeros can occur outside the selected reach, as the same paragraph correctly states later. The introductory sentence should describe the intervention-defined reach rather than all possible zero formation.

   Suggested replacement: “Gate placement determines which operations are reached when the selected activations are entirely zero.” This cleanly introduces the counterfactual used in the next sentence without implying that the gates control every zero in the model.

2. **State what OL1's relative norm is relative to.** Main lines 38–40: “A positive budget caps the correction's relative norm.” The appendix is precise, but the main paragraph should convey the idea without requiring it. The phrase also risks readers interpreting the budget as an absolute correction cap.

   Suggested replacement: “A positive budget caps its norm relative to the adaptive task direction.” Keep the existing no-loss-guarantee sentence; there is no need for additional optimization discussion.

3. **Disambiguate the local-sparsity denominator.** Appendix lines 96–99 define the denominator using $|A_s^{(\ell,b)}|$, while bars denote absolute activation magnitudes in the pressure section and nearby near-zero definition. A tensor's cardinality is intended, but is not explicitly stated.

   Smallest fix: append “where $|A_s^{(\ell,b)}|$ denotes the number of entries.” Alternatively introduce an element-count symbol, but adding a new symbol is less economical here.

## Optional high-value polishing

- **Explain the reach example at the moment it appears.** Main lines 72–73 state that zeroing v reaches probability–value and the attention-output projection. Add the short reason, “because $V=0$ also makes the context $PV$ zero.” The example then teaches the closure rule rather than merely reporting it. This is a better use of words than additional general claims about the methodology's purpose.
- **Replace “one-sidedly.”** Main line 29 can read: “A4 applies one-sided gates to $\{a,m,h,z\}$; A7 adds symmetric gates at $\{q,k,v\}$.” It is clearer and more natural at essentially the same length.
- **Name the prediction task once.** “We pair held-out next-token cross-entropy…” is marginally clearer than “validation cross-entropy” for a reader approaching the results. The complete target-count and coverage distinction can remain in the appendix.
- **If further compression is needed, trim metacommentary first.** “The methodology separates what an intervention changes…” repeats the introduction's organizing idea. The subsection sequence already does that work. Keep the concrete gate limits, matched-contrast qualification, metric denominator, and natural-zero caveat before keeping this opening sentence.

## What should remain

- The three headings are enough; subdividing gates, pressure, and comparisons would fragment a short section.
- Keep the gate formulas inline. They are readable, short, and central to the result interpretation. The appendix's case formula and denominator equations appropriately use displays.
- Keep the ReLU-versus-zero-threshold forward distinction in the main text and its derivative caveat in the appendix.
- Keep the A4-OL1/A7-OL1 comparison boundary in the main text. It is a prerequisite for interpreting the planned result, not incidental implementation detail.
- Keep the dense head in the denominator and the explicit multiplication-workload scope. These make the term “model-wide sparsity” precise.
- Keep the natural-zero caveat next to the ceiling definition. It prevents readers from treating the maximum label as a universal empirical upper bound.
- Keep detailed data coverage, AdamW clipping and moments, tensor weighting, and attention-product multiplicities in the appendix. Moving them into the main text would make it harder to read without materially improving first-pass understanding.

## Readiness judgment

After the three small precision fixes, this methodology can support the next drafting stage. The later results/setup must specify the actual conditions and training protocol, but this draft should not anticipate them with a large generic configuration inventory. No empirical claims need to be strengthened to make the methods read more confidently.

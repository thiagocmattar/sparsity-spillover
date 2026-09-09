# Scientific review — revised methods/experiments split

5 September 2026. Second pass on the actual revised Methodology, Experimental
Study, and both appendices. This assesses the requested structural manuscript
stage against the operational/source audit. It does not require or introduce
results, choose a new experiment, or assert complete submission readiness.

## Rubric and scores

Unchanged equal 1–5 criteria: **1 blocking**, **2 major revision**,
**3 competent**, **4 strong**, **5 exceptional**. Maximum 25.

| Criterion | Revised | Initial | Assessment |
| --- | ---: | ---: | --- |
| Operational fidelity | 4/5 | 4/5 | The split preserves the actual gates, pressure averaging, optimizer procedure, counting rules, and site placements. Saying pressure uses the executed site activation after a gate where present correctly permits independently chosen gate and pressure sets. |
| Metric/reach precision | 5/5 | 4/5 | Reach is now defined by structural zero guarantees independent of checkpoint values and natural zeros. The actual graph inventory is correctly located with the experiment. Exact counters, denominator-only head, causal validity, and selected-site versus observed sparsity remain precise. |
| Contrast identifiability | 5/5 | 5/5 | The recipe comparison boundary, same-size matching, separate pretraining conditions, and zero-threshold A4/A7 identity nuance are preserved. The text does not turn the ladder into a complete factorial or training curriculum. |
| Setup/coverage honesty | 4/5 | 4/5 | Fixed-token larger-size scope is now explicit in the main, together with selected recipe coverage and one-seed interpretation. Condition-level grids and optimization settings are transparently deferred to the later result cohort rather than invented. |
| Claim rigor | 5/5 | 4/5 | The revised cross-size statement concerns per-operation responses versus aggregation contributions; it no longer implies that an aggregate sparsity/ceiling pair identifies a learned mechanism. No empirical finding, speedup, or stronger transfer conclusion is asserted. |
| **Total** | **23/25** | **21/25** | **No substantive unresolved issue or mandatory scientific correction identified for this stage.** |

## Verification of the revisions

1. The general reach definition now counts products “guaranteed to have a zero
   activation operand as a structural consequence” of selected-site zeros,
   explicitly independent of checkpoint values and natural zeros. This agrees
   with the declared operation union in `ceilings.py` and preserves A0's zero
   selected-site reach. The Pythia-specific V-to-context closure remains in
   the experimental appendix.
2. The Methodology main refers to the declared block matrix inventory; the
   six Pythia families and their equal-width multihead denominator now appear
   with the actual architecture. No universal Transformer denominator is
   implied.
3. The Experimental Study main explicitly calls the 70M/410M extensions
   fixed-token and lists their selected families. It does not claim matching
   all optimizer settings across sizes, equal exposure per parameter, or
   convergence-matched transfer.
4. The final comparison paragraph now distinguishes per-operation measured
   response from the operations' contributions to aggregate sparsity. Given
   the defined operation numerators and denominators, this is a valid
   accounting distinction. It is not presented as a causal explanation of
   learning or a result already demonstrated by the paper.
5. Post-gate pressure wording now explicitly allows an ungated pressure target.
   The objective still uses actual executed activations and equal weighting
   across targeted tensors. Moving the site definitions did not change the
   formula or incorrectly force the pressure set to equal the gate set.
6. Actual post-RoPE QK operands and actual post-gate attention context are
   preserved in the Pythia appendix. The general counter discussion instead
   states the appropriate architecture-independent rule: count the operands
   consumed after intervening transformations.

## Optional wording only

The first methodological sentence says an intervention specifies “a gate
family and threshold.” The experimental A7 recipe deliberately uses different
families at different sites. Its explicit recipe definition prevents a real
ambiguity, but “a gate specification and threshold” or “per-site gate families
and a threshold” would make the general sentence naturally include mixed
mappings. This is optional wording, not a scientific blocker.

## What remains for the later manuscript stage

The split now puts general intervention/measurement definitions before the
concrete model, data, recipe map, and matched-comparison logic. The reader can
understand the method before encountering A* labels, and then assess which
questions the existing comparisons can answer. The remaining work is the
already-deferred study specification and evidence: actual condition grids,
optimizer values, precision, checkpoint choices, complete cohort coverage,
and the eventual empirical answer and transfer limits. None should be
fabricated to make this structural revision appear complete.

No active manuscript, scientific code, or result file was changed by this
reviewer; only this review file was added.

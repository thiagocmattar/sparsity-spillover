# Proposed results argument

The evidence supports a conditional design lesson: pressure's added value
depends on the gate threshold and where it acts. Broader reach is valuable only
when the learned representation supplies zeros in the additional operations.
Selected high-threshold recipe orderings persist across sizes, while the
zero-threshold ordering and dose response change. This answers the revised
introduction more directly than a single claim that more sparsification is better.

## 1. Establish the trade-off before explaining it

Use Figure 01 for the complete matched 14M study. All 35 trained endpoints are
visible, including the five historical h-only-pressure conditions under their
actual label. The 20 A0/A1-H uniform-clipping controls supply an evaluation-only
reference. Panel (b) retains the complete loss range, including the eight
clipping points outside the main panel. Rings mark nondominated trained points;
solid and dotted envelopes distinguish the main ladder from the pool that
includes historical A4-OL1[h]. Connecting lines order evaluated conditions;
they do not establish attainable intermediate mixtures.

Suggested text:

> The 14M study spans distinct quality-sparsity regimes. Local L1 pressure on
> the ReLU FFN hidden activation produces the lowest observed validation loss:
> at lambda=1, loss is 5.1023 with 3.9493% model-wide sparsity, compared with
> 5.2696 and 2.7141% for the unpressured ReLU control. Expanding gate placement
> reaches a different region of the trade-off. A7-OL1 at kappa=0.5 reaches
> 27.4827% at loss 5.8294. These endpoints illustrate why a recipe must be
> evaluated on quality and model-wide sparsity together.

The corrected A4-OL1 family contributes no point to the **global** trained
frontier across the main 30-condition ladder, despite improving or extending
parts of its **within-A4** comparison. Those are different selection pools;
do not describe a within-family gain as a global frontier gain.

Historical A4-OL1[h] is informative about placement: kappa=0.05 reaches
9.0356% at loss 5.1956. Expanding the pressure objective from h to four sites
increases loss at every matched threshold (0.2425-0.3196 nats/token).
The objective's equal-tensor normalization also changes, so this contrast
does not identify one harmful added site. Retain this result explicitly;
do not relabel it as the intended four-site recipe or restore discarded F001.

Figure 06 completes the post-hoc comparison with all 150 existing 14M clipping
points. Clipping the lambda=1 L1 model gives additional low-cost points, e.g.
4.7982% at loss 5.1070 and 5.6441% at loss 5.1441. The retained post-hoc
study did not clip A7 or the corrected A4-OL1 cohort. Thus the comparison
does not establish a universally optimal training-plus-clipping recipe, and
uniform TEAL-style clipping is not a reproduction of TEAL's greedy allocation.

## 2. Explain pressure's added value with blocked contrasts

Figure 02 has the requested two rows and shared x positions: change in loss
above and change in model-wide sparsity below. Each block names its own
parent and child. The sequence builds the explanation from local nonlinearity
to pressure, broader gates, and pressure at broader gates. It is not a
training curriculum or an additive decomposition of one final model.

Suggested text:

> Pressure's benefit depends on thresholding. At fixed A4 gates, adding OL1
> improves both outcomes at kappa=0 and 0.01. At kappa=0.5, it instead adds
> 2.4979 percentage points of model-wide sparsity at a 0.3783 loss cost.
> The response differs under A7 gates: at zero threshold, OL1 slightly
> worsens both outcomes, whereas at kappa=0.1 it adds 1.3717 points with a
> 0.000819 loss increase. At kappa=0.5 the gain rises to 12.0959 points
> at a 0.1265 loss cost. There is no uniform pressure benefit across doses.

The loss changes near 0.001 are measured differences, not a statistical
equivalence or significance claim. One seed cannot estimate between-seed
uncertainty. Within each fixed topology, the pressure/no-pressure contrast
isolates the addition of that topology's specified OL1 objective. Comparing
the two topologies' pressure responses changes objective sites and target
normalization; it is a recipe-dependent response, not a fixed-objective
gate-by-pressure interaction estimate.

The no-pressure A4-to-A7 block isolates the added Q/K/V symmetric gates.
At kappa=0 these gates are mathematical identities. The small measured
residual (-0.002123 loss; +0.005632 sparsity points) should be treated as
an implementation/numerical control discrepancy, not a benefit of the
identity transformation. At positive thresholds the contrast becomes active.
Likewise A1-H versus A4 at zero threshold changes gate support at additional
sites and the exact boundary derivative convention at h.

## 3. Test transfer while separating learned response and reach

Use Figure 03, with raw sparsity on top and the proposed normalized ratio
below. Both use loss relative to the same-size A0, retaining the A0 absolute
losses in the protocol table. All five trained doses remain visible.

Suggested text:

> The high-threshold recipe ordering persists across all three sizes. At
> kappa=0.5, A7-OL1 improves on A4-OL1 by 14.7692, 5.0057, and 9.0241
> model-sparsity points at 14M, 70M, and 410M, while reducing loss by
> 0.2086, 0.1736, and 0.0703, respectively. At zero threshold, however,
> A4-OL1 is better on both outcomes at 14M and 70M, whereas A7-OL1 is
> better at 410M. The larger-model evidence therefore supports partial
> recipe transfer, with a regime-dependent low-threshold response.

Do not write that the **pressure effect** replicates at larger scales: those
cohorts lack unpressured A4/A7 controls. They test complete A4-OL1/A7-OL1
recipes. There are no independent-seed repetitions, and three equal-token
cohorts do not define a scaling law. The budget is 1.493 billion input tokens
at every size, giving 106.1, 21.2, and 3.7 tokens/parameter. The 410M peak
learning rate is 3e-4 versus 1e-3 at 14M/70M. Its lower A0 quality than 70M
and reversed dose response constrain extrapolation; the earlier LR screen
does not identify undertraining as the cause.

### Recommendation on the ceiling idea

Keep it as a secondary explanatory normalization, **not a model-size-invariant
performance score**. At T=2048, the A7 all-block ceiling is 29.952%, 49.424%,
and 87.245%; its high-threshold endpoints realize 91.75%, 82.15%, and 92.40%
of those respective ceilings. Much of the increase in raw model-wide sparsity
therefore reflects changing architecture/workload shares, while realized
zero rates also vary. The ceiling is structural reach, not a budget of zeros
available without losing quality.

The main normalization is the user's proposed ratio
`U_arch = R_model / R_model_max = observed zero products / reachable products`.
It includes natural zeros outside selected reach and need not be bounded by
one. A0 has zero selected reach, so its ratio is undefined, not zero. We also
save `U_reach`, restricting the measured numerator to reachable operation
families. This is a useful sensitivity check, not a relabeling of R_model.
For high-threshold 14M A4-OL1, the two ratios are 99.07% and 98.63%; the
outside-reach contribution is 0.05657 model-sparsity points. A7 reaches all
counted block operations, so its ratios agree exactly.

Crucially, A4-OL1 realizes **more of its own narrower ceiling at every matched
dose and size**. That does not overturn A7-OL1's high-threshold raw frontier
advantage; dividing by different reachable workloads changes the estimand.
Figure 04 makes this explicit through operation contributions and zero rates.
For A7, the normalization equals the block-only zero-product fraction exactly.
Even that quantity still weights operations differently as width and sequence
length change; it is not fully architecture invariant.

## 4. A case study of exact zeros versus small activations

Figure 05 uses A0 and A4-OL1/A7-OL1 at kappa=0 and 0.5. These are the tested
threshold endpoints, chosen to expose the change in the above contrasts rather
than selected as a quality optimum. A0 is repeated as the same baseline in
both rows. The six common retained sites are m, h, post-RoPE q/k, v, and the
attention output **after W_o**. That last tensor is not pre-projection z.

Suggested text:

> At kappa=0.5, both recipes make more than 99.8% of FFN hidden activations
> exactly zero, yet their attention operands differ sharply. Under A4-OL1,
> 47.64% of query values and 56.12% of key values lie within magnitude 0.01,
> while their exact-zero fractions remain below 0.001%. Under A7-OL1,
> 93.54% of queries and 94.54% of keys are exactly zero; value zeros reach
> 98.71%. The corresponding QK/PV zero-product rates explain how A7 accesses
> additional attention work beyond A4's nearly exhausted linear reach.

Smaller RMS does not imply more exact zeros: at this dose A4-OL1 query RMS
is 0.146, below A7-OL1's 0.483, despite A7's far larger exact-zero fraction.
An explanation in terms of magnitude reduction alone misses gate placement.
The post-W_o output can be dense despite a sparse pre-W_o context; the
projection and bias break the identification between those two ports.

The retained data supports four empirical magnitude bands and pooled RMS,
not a continuous density, signed histogram, quantiles, or raw samples.
Lines in Figure 05 guide comparisons between categorical bands; the y axis
is logarithmic above 0.01% and linear near zero to retain exact empty bands.
No unseen distribution was reconstructed. The broader all-site table retains
a and z where available; A0 lacks their three-threshold diagnostic in this
source pass. No strong causal spillover claim follows from these marginals,
and no universal attention broadening pattern is supported by this selection.

## 5. Finish with the already established execution case

Retain the draft's current Run 029 kernel subsection after the training
results. Figure 07 packages its final figure with unchanged bytes; O007
provides the full caption and attribution boundary. The 42-proposal matched
retrospective achieves a 1.7830x qualified incumbent on the search checkpoint,
and K050 qualifies on 35/35 checkpoints with a 1.2502x geometric mean
speedup. The sparsity association has R-squared 0.7813. Sparse paths add
5.6% geometrically averaged acceleration over the fused no-skip ablation,
helping 19/35 checkpoints; attention skipping itself adds overhead.

This completes the argument from interventions to zeros to affected work to
measured implementation benefit. The runtime cohort changes weights, gates,
and quality across checkpoints; it cannot establish equal-quality gains or
causal sparsity-speedup proportionality. It is a BF16, batch-one, uncached
2048-token RTX5090 workload, not evidence for cached decoding or larger-scale
kernel transfer. Historical negative H100 transfer remains in the evidence
audit; it is not pooled into this common-denominator curve.

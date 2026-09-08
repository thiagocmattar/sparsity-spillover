# Argument map before drafting the results

Prepared 8 September 2026, before writing the results prose. Evidence comes
from Analysis 018 observations O001/O002/O003/O004/O007/O011 and Runs 030/031.
The organizing question is how pressure, thresholding and site placement
change the quality-sparsity trade-off, and whether the resulting zeros help
execution. Each step answers a different question; none substitutes for the next.

| Step | Evidence and inference | Boundary | Transition |
| --- | --- | --- | --- |
| 1. Establish the trade-off | Figure 01 places local OL1, broader trained recipes and A0 clipping on common loss/sparsity axes. Local pressure occupies a low-loss, limited-reach regime; A7-OL1 reaches much greater sparsity at a quality cost relative to A0. | The overview selects 16 trained endpoints, and curve connections do not interpolate attainable models. Broader site reach is part of the comparison. | Which additions account for the differences? |
| 2. Identify conditional effects | Figure 02 uses 29 explicit treatment-minus-reference contrasts. Pressure improves both axes in some settings and incurs loss in others; its large-threshold sparsity increment is much greater for A7 than A4. | Across-topology contrasts change the pressure objective as well as the gates. OL1 is not uniformly better than naive L1. | What does the corresponding activation mass look like? |
| 3. Inspect the representation | Figure 05-v3 and its exact-zero table separate a narrow nonzero attention peak under A4-OL1 from the large exact-zero mass under A7-OL1. A4 changes q/k/v despite not intervening directly there. Operation counts connect these distributions to model-wide opportunity. | These are pooled marginals of complete recipes, not a causal tracing experiment. A7 directly targets q/k/v. | Does the useful recipe ordering persist when model size and operation weights change? |
| 4. Test transfer | Figure 03 and the six-row contrast table show that A7-OL1 beats A4-OL1 on both axes at kappa=0.5 for all three sizes, while the kappa=0 ordering changes. Common-A7 normalization separates block sparsity from the dense-head workload share. | One seed per size, fixed tokens and a different 410M learning rate; larger cohorts do not isolate the pressure addition. No scale-invariant mechanism or scaling law is established. | Are logical opportunities useful on hardware? |
| 5. Test execution | Figure 07 shows qualified search progress and K050 on the corrected 30-checkpoint cohort. Matched ablations separate part of the sparse-path contribution from fusion and expose attention-skip overhead. | One GPU and full-sequence workload; checkpoint quality differs. Correlation is descriptive and acceleration is not an equal-quality comparison. | Evaluate pressure at its actual gate settings, count where zeros affect work, and time the implementation. |

## Prose decisions

Lead each subsection with its finding, then give selected numerical evidence
and explain its consequence. Use transitions between the five questions.
Put common protocol and one-seed scope in the setup, exact recipe tables and
all endpoints in the appendix, and only the six-row cross-scale contrast
table in the main results. Keep the figures' captions self-contained but
avoid repeating every caption detail in the prose.

The density caption must state that exact-zero mass is excluded and that the
symlog display is not a probability-area plot. The scaling caption must state
the common A7 denominator. The kernel caption and prose must consistently use
30 checkpoints, mean speedup 1.2340x and R-squared 0.8167; the original
35-checkpoint retrospective is historical evidence, not the plotted cohort.

## Structure to implement

Keep the existing general methods and setup. Put the full architecture/ladder
in the experimental appendix, then insert the four training results before
the revised kernel subsection. Number the five selected result figures by
appearance: overview, paired effects, distributions, scale, execution.
Appendix material supplies the training protocol, 54 endpoints, 29 paired
effects, 15 scale contrasts, exact-zero/tail coverage, operation accounting,
full-range clipping trajectories, and kernel qualification/ablations.
Copy the selected figures without alteration and retain their source hashes.

This map records the intended evidence chain; the TeX is the prose written
from it. It does not authorize additional experiments or claim unmeasured
causal effects.

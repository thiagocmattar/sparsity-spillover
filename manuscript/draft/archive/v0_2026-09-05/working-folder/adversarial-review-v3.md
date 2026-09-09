# Adversarial review v3: paired interventions and the path to a stronger paper

Status: reviewer simulation and evidence-planning note. This review reassesses
the current evidence as a representation and optimization paper. The TeX
manuscript and finding registry remain unchanged, and experiment authorization
remains separate.

## Revised verdict

Explicitly treating the 14M experiments as paired contrasts improves the
assessment, while the current evidence remains below the likely acceptance
threshold. I retain **15--20%** for the present narrative. A rewrite that fully
uses the paired design, corrects the related-work claim, and adds the
existing-artifact comparisons described below reaches **18--30%, centered near
24%**. This remains a weak-reject range because strong baselines, replication,
and novelty relative to [Q-Sparse](https://arxiv.org/abs/2407.10969) are still
open. A bespoke sparse kernel carries much less weight for this paper identity.

The representation framing improves the submission's prospects. Its claims
should concern how training and intervention placement shape sparse
representations, the quality--sparsity tradeoff, and the arithmetic reuse of
zero activations. Measured speedup can remain a motivating downstream
objective. A small amount of hardware-facing analysis would strengthen
relevance; representation analysis and replication offer greater value now.

Under this claim scope, the evidence establishes changes in sparse
representations and in the number of scalar products with an exactly zero
activation-derived operand. Inference speed remains a future empirical
question. Claims of reduced inference cost, latency, or executed computation
would require runtime measurement.

The strongest paper identity is:

> Where sparsity is imposed changes both how optimization creates zero
> activations and how often those zeros are reused by model computations. A
> matched intervention study shows that gate placement and sparsity pressure
> affect each other, so local activation sparsity gives an incomplete account
> of the quality--model-wide activation-sparsity-opportunity tradeoff.

`R_model` counts scalar multiplications with at least one exactly zero
activation-derived operand, accounting for overlap when both operands are zero
and for causal attention coverage. The controlled intervention study and its
candidate architecture--optimization hypothesis form the publishable
contribution; `R_model` supplies the common evaluation axis.

## What the paired design establishes

The 14M study holds initialization, data order, optimizer, schedule, training
budget, validation workload, and, where applicable, threshold fixed within each
comparison. This yields a controlled within-seed contrast. It is much stronger
than comparing unmatched recipes from separate papers or independently chosen
runs.

Pairing leaves two limits:

1. It estimates the effect for one initialization and data order. Fresh paired
   seeds are needed to establish that the direction and size of the effect are
   stable across training realizations.
2. A paired comparison isolates one factor only when the endpoints differ in
   one factor. A4 versus A7 isolates the addition of the three attention gates.
   A4-OL1 versus A7-OL1 changes both gate sites and pressure sites, so it is a
   comparison of two complete recipes.

The clearest central design is a 2-by-2 layout of topology-conditioned recipes,
repeated over five threshold doses:

| Gate placement | No sparsity pressure | Pressure at all gated sites |
| --- | --- | --- |
| A4: `a,m,h,z` | A4 | A4-OL1@`{a,m,h,z}` |
| A7: A4 plus `q_post,k_post,v` | A7 | A7-OL1@all seven sites |

This layout varies gate placement together with a topology-conditioned
objective. The pressure target expands when the topology expands. A factorial
interaction would require a fixed pressure policy across topologies. The
missing condition, A7 gates with pressure restricted to the original four A4
sites, would separate the effect of adding attention gates from the effect of
directly pressuring their activations.

The other useful paired contrasts should appear as supporting ablations:

- GeLU to ReLU tests the activation function.
- ReLU to ordinary L1 tests local sparsity pressure.
- ordinary L1 to orthogonal L1 compares complete pressure-update methods. OL1
  also changes moment construction, applies a separate displacement, and adds
  trust scaling. Its observed advantage is mixed, so it should remain an
  ablation.
- A4 with `h`-only pressure to A4 with four-site pressure tests pressure scope.
  For the realized seed and A4 recipe, expanding pressure from `h` to
  `{a,m,h,z}` raises validation loss at every tested threshold.
- A4 to A7 without pressure isolates the joint addition of the three post-RoPE
  Q/K/V gates.

## The strongest exploratory pattern already present

The underused pattern is the dose-dependent difference between the two
topology-conditioned pressure responses. The table below gives the effect of
adding each topology's corresponding pressure policy. The last column for each
outcome is the A7 pressure effect minus the A4 pressure effect. Positive
`R_model` change and negative loss change are favorable.

| Threshold | A4 pressure: change in `R_model` | A7 pressure: change in `R_model` | A7-minus-A4 pressure-effect difference | A4 pressure: change in loss | A7 pressure: change in loss | A7-minus-A4 pressure-effect difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.5980 pp | -0.1634 pp | -0.7614 pp | -0.0123 | +0.0118 | +0.0241 |
| 0.01 | +1.1730 pp | +0.1057 pp | -1.0673 pp | -0.0082 | +0.0170 | +0.0252 |
| 0.05 | +2.2506 pp | +0.7361 pp | -1.5144 pp | +0.0557 | +0.0249 | -0.0308 |
| 0.1 | +2.4413 pp | +1.3717 pp | -1.0695 pp | +0.1287 | +0.0008 | -0.1279 |
| 0.5 | +2.4979 pp | +12.0959 pp | +9.5980 pp | +0.3783 | +0.1265 | -0.2518 |

At thresholds 0.1 and 0.5, the seven-site objective has a markedly different
pressure response from the four-site objective. At threshold 0.5, seven-site
pressure creates 9.60 percentage points more incremental model-wide
opportunity than the corresponding A4 pressure effect, while its loss penalty
is 0.252 smaller. At threshold 0.1, pressure adds less opportunity under A7
than under A4, while the observed loss change is +0.0008. The numerical
sentinel below places that loss difference within the unresolved range. The
current design attributes these differences to the complete
topology-conditioned policies. Separate attribution requires a fixed pressure
scope.

This pattern has two adversarial qualifications. First, it was recognized
after examining the completed grid, so it is an exploratory hypothesis until a
fresh-seed experiment confirms it. Second, A7-OL1 pressures the added Q/K/V
sites, while A4-OL1 pressures the original four sites. The missing
fixed-pressure condition is the cleanest way to determine whether the
hypothesized architecture--optimization interaction comes from gate placement,
direct Q/K/V pressure, or their combination.

The A4-to-A7 threshold-zero contrast is also useful as an internal numerical
sentinel. The added symmetric Q/K/V gates are effectively identities at zero;
the observed difference is -0.0021 loss and +0.0056 percentage points of
`R_model`. The +0.0008 A7 pressure loss change at threshold 0.1 falls inside
this scale and should be described as unresolved. This sentinel measures
numerical divergence inside the matched setup, whose GPU jobs were scheduled
separately. Across-seed uncertainty requires fresh paired training runs.

## Acceptance-impact ranking

The changes below are estimated **absolute percentage-point changes in
acceptance probability**, conditional on clear, supportive results. Their
effects overlap substantially, so each range should be read against the stated
baseline. A strong comparator can also lower the probability by closing the
reported gap; that outcome would correctly falsify the current claim.

| Priority | Addition | Cost class | Conditional effect | Why reviewers would care | Main adverse outcome |
| ---: | --- | --- | ---: | --- | --- |
| 1 | Fresh paired seeds for the central 14M contrasts | New training | **+6 to +12 pp** | Converts a one-trajectory observation into replicated evidence and permits uncertainty on intervention effects. | Direction changes across seeds or the small improvements disappear. |
| 2 | Matched implementation of [Q-Sparse](https://arxiv.org/abs/2407.10969) under the study protocol | New training | **+5 to +9 pp** | Addresses the closest train-time, model-wide prior art and tests for insights beyond top-k plus a straight-through estimator. | The proposed recipes are uniformly weaker and offer little new mechanistic insight. |
| 3 | Faithful implementation of [TEAL's](https://arxiv.org/abs/2408.14690) published blockwise greedy threshold allocation | Checkpoint evaluation | **+4 to +8 pp** | Tests the central claim against a strong optimized train-free frontier and upgrades the uniform-clipping control. Calibration data must remain separate from final evaluation. | **-8 to -15 pp** if optimized clipping closes most of the gap. |
| 4 | Cross-evaluate A4-trained and A7-trained checkpoints under both A4 and A7 gate topologies | Checkpoint evaluation | **+2 to +5 pp** | Measures acute gate sensitivity and compatibility between each learned representation and the alternate topology at matched training steps. | An easier evaluation topology explains the apparent gain and weakens the adaptation account. |
| 5 | Add A7 gates while keeping pressure fixed at the four A4 sites | New training | **+4 to +7 pp** | Completes the most important missing contrast and identifies whether Q/K/V pressure causes the differential response. | The exploratory response difference collapses when pressure scope is controlled. |
| 6 | Replicate the controlled A4/A7 recipe layout at 70M | New training | **+4 to +8 pp** | Turns "the selected recipe persists" into a cross-size test of the same scientific mechanism. | The response difference is specific to 14M. |
| 7 | Re-evaluate the 12 retained checkpoints per 14M condition | Checkpoint evaluation | **+2 to +4 pp** | Shows when zeros emerge, when quality recovers, and whether the endpoint response difference reflects stable dynamics or a late fluctuation. The promoted 70M and 410M cohorts retain final checkpoints only. | Curves cross erratically or the claimed mechanism appears only at the endpoint. |
| 8 | Continued-training adaptation from a dense checkpoint | New training | **+5 to +9 pp** | Fills the conceptual gap between train-free clipping and training from initialization, and compares with a regime used by prior work. | Short adaptation recovers the full benefit, narrowing the contribution of training from initialization. |
| 9 | Evaluation on an untouched text corpus | Checkpoint evaluation | **+1 to +3 pp** | Tests whether quality--sparsity behavior transfers beyond the corpus used for training and model selection. | The frontier improvement is corpus-specific. |
| 10 | Check whether activation shrinkage is compensated by larger downstream weights | Existing diagnostics plus checkpoint evaluation | **+2 to +5 pp** | Tests whether absolute thresholds encourage activation shrinkage followed by weight growth instead of structurally useful sparsity. | The model exploits this rescaling loophole, requiring a narrower claim or normalized gates. |
| 11 | Q/K/V attribution | Checkpoint evaluation or new training | **+1 to +3 pp** for evaluation; **+3 to +5 pp** for trained ablations | Evaluation-time subsets provide checkpoint-conditional attribution; separately trained variants identify training effects. | Evaluation-only subsets fail because the checkpoint co-adapted to all three gates. |
| 12 | Mask organization and workload-sensitive accounting | Checkpoint evaluation | **+2 to +5 pp** | Connects learned representations to implementable structures: shared channels, blocks, heads, and full-sequence versus cached decoding. | Zeros are scattered and offer little reusable structure. |
| 13 | Train a shared channel/block-mask pressure rung | New training | **+3 to +6 pp** | Tests whether optimization can create reusable paths across related tokens or batches, giving a direct representation-to-implementation bridge. | Q-Sparse already studies block structure, so an unmatched demonstration adds little novelty. |
| 14 | Joint activation and weight sparsity rung | New training | **+2 to +4 pp** | Extends the intervention view to both operands of linear products and can expose complementary or redundant zero patterns. | It opens another literature and baseline burden, diluting the current question. |
| 15 | Paired block-level loss distributions and fixed-quality frontiers | Re-analysis or checkpoint evaluation | **0 to +2 pp** | Replaces selected endpoint anecdotes with uncertainty over validation examples and explicit fixed-loss comparisons. | Block resampling quantifies evaluation-set composition; training seeds quantify training variability. |
| 16 | End-to-end or kernel runtime | Systems experiment | **+2 to +5 pp** | Corroborates the inference motivation. Its marginal value is limited when the claims stay representation-focused. | Packing and dispatch costs erase speed gains. |
| 17 | Supervised fine-tuning robustness | New training | **0 to +2 pp** | Tests whether downstream adaptation preserves the learned sparse paths. | Fine-tuning destroys the structure. |
| 18 | Reinforcement-learning or preference-training study | New training | **0 to +2 pp** | Broadens scope while offering little leverage on the current paper's main uncertainty. | It diffuses the contribution and consumes budget while leaving the mechanism unclear. |
| 19 | Training-maturity-matched 410M replication | Large new training | **0 to +2 pp** under the proposed two-scale claim; **+2 to +6 pp** only if scale becomes central again | Adds a third mature scale through substantially greater token exposure. | Very high cost and lower marginal value than seeds, baselines, and the missing comparison cell. |

The existing operation decomposition should also move into the main argument.
It can add roughly **0 to +2 pp** through clearer evidence presentation: report
which attention and MLP multiplications contribute each change in
`R_model`, and show cases where ranking by local zero fraction disagrees with
ranking by operation-weighted opportunity. This is an analysis of the current
artifacts, so it belongs in the paired-framing revision as an existing-evidence
analysis.

Two additional re-analyses would sharpen the role of the metric and the gates:

- Compare every endpoint using unweighted local exact-zero mass, TEAL's
  matrix-weighted sparsity summary, and `R_model`; report rank disagreements and
  agreement rates. Systematic differences would show what operation-aware
  counting adds beyond established summaries (**+2 to +4 pp**).
- Separate topology-adoption cost from threshold-dose response. Report A0 to
  A4/A7 at threshold zero, then report each gated family's higher thresholds
  relative to its own zero-threshold endpoint (**+1 to +3 pp**). This keeps the
  cost of adopting one-sided A4 gates distinct from the effect of increasing
  the threshold.

Mask organization should serve as mechanism and implementation-relevance
evidence. Prior work already documents layer-, token-, and batch-dependent mask
structure in ReLU Transformers ([Mirzadeh et
al.](https://arxiv.org/abs/2407.07848)), so a descriptive mask figure alone
offers limited novelty.

Gate-topology cross-evaluation is diagnostic. The A4 and A7 weights followed
different optimization trajectories, so this analysis measures acute
sensitivity and co-adaptation compatibility. A unique decomposition of
"quality recovered through training" would require a stronger identification
design.

## Highest-value program centered on representation evidence

The best sequence for a representation and optimization paper is:

### 1. Rebuild the argument from existing artifacts

- Present the 14M study as a map of matched contrasts, with the cumulative
  recipe ladder serving as secondary organization.
- Make the differential topology-conditioned pressure response the central
  exploratory pattern and the architecture--optimization interaction the
  confirmatory hypothesis.
- Decompose model-wide opportunity by operation and report local/model-wide
  ranking disagreements.
- Treat ordinary versus orthogonal L1 as an ablation and the four-site pressure
  result as a seed-specific warning about expanding pressure scope.
- Keep the full 410M result in the appendix as an equal-token boundary case and
  disclose it in the main text: "An equal-token 410M promotion reversed the
  trained dose response under substantially lower tokens-per-parameter
  exposure, so we exclude it from scale inference and report it separately."

This revision alone moves the forecast from 15--20% to approximately
**18--30%**.

### 2. Use retained checkpoints for strong falsification and mechanism tests

- Implement a faithful optimized train-free comparator.
- Cross-evaluate A4-trained and A7-trained checkpoints with both gate
  topologies at matched checkpoint times.
- Evaluate the 14M learning trajectories retained at steps
  `0,1,2,4,8,16,32,64,128,256,512,712`.
- Audit activation scale, downstream weight norms, and their products.
- Add one untouched-corpus evaluation and measure channel/block occupancy and
  cross-token mask consistency as the primary bridge to implementable sparse
  computation.

If these results support the hypothesis, the forecast rises to roughly
**24--36%** using the completed pretraining runs and checkpoint evaluations.
The one-seed training design remains the main ceiling on this package.

### 3. Run a small confirmatory training package

- Pre-register fresh paired seeds at thresholds 0.1 and 0.5 for A4, A4+OL1@4,
  A7, A7+OL1@4, and A7+OL1@7. State that seed 1234 supplied the discovery
  evidence and the two selected thresholds.
- If budget permits, reproduce the same controlled contrast at 70M.

Supportive results would move the forecast to about **32--45%**. Adding a
credible Q-Sparse comparison and either the 70M controlled replication or a
continued-adaptation study would place a strong version near **38--52%**.
Runtime evidence could improve confidence inside these ranges as optional
corroboration for the inference motivation. These scenarios assume that the
paper confines empirical claims to representations, validation quality, and
logical zero-product opportunity; claims about realized inference efficiency
would change the evidence requirement.

## Adversarial reviewer summary

| Reviewer lens | Likely score now | Acceptance estimate | Central objection | Evidence most likely to change the score |
| --- | ---: | ---: | --- | --- |
| Novelty and significance | 4/10, weak reject | 12--22% | Q-Sparse already spans attention, MLPs, batching, and multiple training regimes. | Direct comparator plus a confirmed topology--pressure interaction. |
| Empirical rigor | 5/10, marginal reject | 20--30% | Excellent matching and validation coverage, but one training realization and a weak train-free baseline. | Fresh paired seeds and TEAL's optimized allocation. |
| Representation/optimization | 5/10, marginal reject | 18--32% | The intervention map is promising, while the mechanism is inferred from endpoints. | Gate-topology cross-evaluation, learning trajectories, and the rescaling audit. |
| Meta-review | weak reject | 18--30% | The paper contains a stronger experiment than its current narrative exposes, but still lacks decisive replication and baseline closure. | A compact package centered on baseline falsification, mechanism, and paired replication. |

## Evidence inspected

- [Analysis 008](../../analyses/008-2026-08-31-full-pass-frontier-with-a7/README.md):
  corrected 14M frontier with A4, A7, and their pressured variants.
- [Analysis 009](../../analyses/009-2026-08-31-run012-vs-run015-a4-ol1-pressure-sites/README.md):
  realized `h`-only versus four-site pressure comparison.
- [Analysis 012](../../analyses/012-2026-09-04-paper-synthesis/README.md):
  cross-size and operation-level synthesis.
- [Finding F002](../../research/findings/F002-a7-extends-a4-logical-opportunity.md):
  approved scope of the matched A4-to-A7 result.

The review standard follows the [ICLR 2027 reviewer
guidance](https://iclr.cc/Conferences/2027/ReviewerGuidelines), which emphasizes
clear claims, placement in the literature, empirical support, and significance.
The probability estimates are subjective forecasts from three adversarial
reviewer simulations. Treat them as comparative planning estimates.

## Bottom line

Yes, treating the 14M conditions as paired interventions materially improves
the paper. The right description is a set of matched contrasts containing a
2-by-2 layout of topology-conditioned recipes, because some comparisons change
more than one component. This supports a stronger contribution than the
`R_model` metric: an empirical account of how intervention placement changes
the optimization response and the arithmetic reuse of learned zeros.

The most valuable next work is fresh paired replication, a faithful optimized
train-free baseline, the missing fixed-pressure A7 condition, and checkpoint
analyses that diagnose gate sensitivity across distinct trained trajectories.
A runtime study, reinforcement learning, and a training-maturity-matched 410M
run have substantially lower expected value for this paper.

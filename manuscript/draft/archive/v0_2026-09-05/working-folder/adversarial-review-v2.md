# Adversarial review v2: scope, novelty, and ICLR prospects

Status: reviewer simulation for editorial planning. This document evaluates
[`narrative-review-v1.md`](narrative-review-v1.md); the TeX manuscript and
finding registry remain unchanged.

## Executive verdict

Move the 410M result out of the abstract and central result. Preserve its full
curves in the appendix and mention the reason for this scope choice in the main
discussion. The result belongs to a sharply different training-exposure
regime: every model received 1.493 billion tokens, which gives 106.1 tokens per
parameter at 14M, 21.2 at 70M, and 3.68 at 410M.

The compute argument supports this editorial decision. Causal attribution of
the 410M behavior requires a longer-horizon experiment. A common
Chinchilla-style reference point of about 20 tokens per parameter would allocate
approximately 8.1 billion tokens to the
405M-parameter model, 5.4 times its executed token budget. Matching the 70M
experiment's 21.2 tokens per parameter would require 8.59 billion tokens. With
training work proportional to parameters times tokens, that matched-exposure
410M experiment would cost about 33 times the executed 70M experiment.

The larger threat to the paper is novelty. Q-Sparse already trains activation
sparsity throughout attention and FFN projections, covers training from
initialization, continued training, and supervised fine-tuning, introduces
block sparsity for batches, and studies scaling from 300M to 7B. The paper
therefore needs a narrower contribution than an integrated
architecture--optimization view.

The three simulated reviewers converge on **weak reject**. For the current
evidence package, the panel estimates **10--20% ICLR acceptance, centered near
15%**. Transparent 410M appendixing and the full narrative repair move the
estimate toward **15--20%**. These are judgment ranges rather than statistically
calibrated probabilities. The 2026 ICLR base acceptance rate was 27.4%; the
present submission sits below that base rate because its inference motivation
extends beyond its measured evidence: model-specific runtime remains unmeasured,
and its central empirical comparisons use one seed.

## The 410M decision

### Recommended treatment

- Abstract and headline claim: 14M and 70M only.
- Main results: one sentence that identifies 410M as a separately reported
  equal-token extension.
- Appendix: complete 410M trained and clipping curves, baseline loss, learning
  rate screen, tokens per parameter, and interpretation limit.
- Scale language: "the result persists at a second model size."
- Future experiment: scale the token budget with model size before drawing a
  model-scale conclusion.

Complete omission would create two problems. First, the 410M cohort was a
planned extension and its behavior changed after the results at smaller sizes.
Removing it after observing that change resembles outcome-dependent selection.
Second, deleting it leaves one-seed evidence at two small models and removes the
only direct observation of where the current protocol stops transferring.

The appendix turns an adverse result into a clear scope boundary. It also lets
the main paper focus on the experiment that can be interpreted cleanly.

### What the compute argument establishes

| Model | Parameters | Training tokens | Tokens/parameter | A0 validation loss |
| --- | ---: | ---: | ---: | ---: |
| 14M | 14.1M | 1.493B | 106.1 | 5.209 |
| 70M | 70.4M | 1.493B | 21.2 | 4.100 |
| 410M | 405.3M | 1.493B | 3.68 | 4.547 |

[Hoffmann et al.](https://arxiv.org/abs/2203.15556) found that compute-optimal
parameter count and training tokens grow together under their setting.
[Sparsing Law](https://proceedings.mlr.press/v267/luo25i.html) is directly
relevant because it reports that activation sparsity changes with training data
and that larger models approach their sparsity limits more slowly.

Together, these results make a longer 410M study scientifically well motivated.
The present study holds the 410M token horizon fixed, leaving the causal
explanation open. The learning-rate screen tests learning rate within one
pass; it leaves training duration unresolved. The 14M--70M comparison also uses
very different tokens per parameter, so the current study should avoid the
label "compute-optimal scaling."

Suggested manuscript wording:

> Our main cross-size comparison covers Pythia-14M and Pythia-70M. Both models
> were trained on 1.493 billion tokens, and the trained interventions extend the
> uniform-clipping frontier at both sizes. The appendix reports an exploratory
> Pythia-410M extension trained on the same token budget, corresponding to 3.68
> tokens per parameter. A common compute-scaling reference would allocate about
> 8.1 billion tokens at this size. We therefore use the 410M experiment to
> characterize the equal-token regime and reserve model-scale inference for a
> study that increases training tokens with model capacity.

## The novelty problem exposed by the review

The proposed literature gap currently places training-time methods largely in
FFNs and local evaluation, positioning the integrated
architecture--optimization view as open territory. That claim will attract an
immediate objection.

[Q-Sparse](https://arxiv.org/abs/2407.10969) is the closest problem:

- it applies top-k sparsification with a straight-through estimator to every
  linear projection;
- it covers attention and FFN components;
- it evaluates training from initialization, continued training, and
  supervised fine-tuning;
- it introduces Block Q-Sparse for batched inference;
- it trains 300M--7B models and develops an inference-oriented scaling law.

The surrounding literature narrows the space further. [TEAL](https://arxiv.org/abs/2408.14690)
provides model-wide training-free sparsification, optimized per-layer
allocation, and measured decoding speed. [CHESS](https://arxiv.org/abs/2409.01366)
combines channel-wise FFN sparsity with selected attention sparsification.
[Spark Transformer](https://arxiv.org/abs/2506.06644) trains both attention and
FFN sparsity. [ProSparse](https://arxiv.org/abs/2402.13516) uses continued
training and progressive sparsity pressure.

A defensible gap is comparative:

> Existing methods combine different gate operators, intervention sites,
> training regimes, objectives, and evaluation measures. This leaves the
> contribution of each choice unresolved under matched conditions. We study a
> sequence of matched contrasts and count the
> exact scalar products receiving zero activations, including post-RoPE query
> and key coordinates and their reuse in causal attention products.

The resulting main contribution is:

> A controlled intervention study showing that, in Pythia-14M and Pythia-70M,
> training sparsity across attention and MLP computations contributes
> Pareto-optimal quality--activation-sparsity points beyond uniform post-hoc
> clipping, together with operation-level accounting that identifies where
> those opportunities occur.

This is an honest empirical contribution. In its present one-seed, small-scale
form, it remains weaker than the evidence expected for ICLR. `R_model` serves
as the comparison's supporting evaluation axis.

## Post-training adaptation and reinforcement learning

Post-training adaptation is the more consequential open regime. The current
ladder jumps from evaluation-only clipping to training
from random initialization. A short adaptation of an already trained dense
checkpoint would answer whether the frontier gain requires full pretraining or
only enough optimization for representations to adjust to the gates.

An expanded timing ladder would contain:

1. uniform evaluation-only clipping;
2. optimized train-free threshold allocation;
3. short sparsity-aware adaptation of a dense checkpoint;
4. longer continued pretraining with sparsity pressure;
5. training with sparsity interventions from initialization.

Q-Sparse and ProSparse already make adaptation a strong prior-art baseline. Its
absence should appear among the main empirical limitations rather than only as
a distant future direction.

Reinforcement learning becomes relevant when the intervention contains
discrete routing choices or directly optimizes a measured, non-differentiable
latency objective. The current gates and pressure objectives admit ordinary
gradient optimization, so an RL baseline adds little to the central causal
question. Post-pretraining robustness is a separate and valuable question:
supervised fine-tuning, preference optimization, or RL may reorganize the
learned sparse paths.

Suggested limitation:

> The present measurements characterize models immediately after pretraining.
> Future experiments will test whether short continued training can create the
> same frontier and whether supervised fine-tuning, preference optimization,
> and reinforcement learning preserve or reorganize the learned sparse paths.

## Reviewer rubric

The [ICLR 2027 reviewer guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines)
asks reviewers to decide whether the problem is specific and motivated, whether
the work is placed correctly in the literature, whether the evidence supports
the claims, and whether the work adds meaningful knowledge. This simulation
scores those questions from 1 (poor) to 4 (strong), then records an overall
10-point recommendation and reviewer confidence.

| Panel criterion | Score | Adversarial assessment |
| --- | ---: | --- |
| Question and motivation | 3/4 | Faster inference is important and the ladder asks a concrete comparative question. |
| Placement in literature | 1/4 | The current framing omits Q-Sparse and overstates the architecture-wide gap. |
| Support for claims | 2/4 | Full validation and exact accounting are strong; one seed, an incomplete 70M ladder, and a weak clipping baseline limit inference. |
| Significance | 2/4 | The matched study is useful, while runtime benefit and broad transfer remain open. |
| Presentation | 3/4 | The revised narrative is clear; several ladder edges change more than one factor. |
| Reproducibility | 4/4 | Provenance, integer aggregation, validation coverage, and operational definitions are unusually strong. |

The weighted panel result is **2.2/4, weak reject**. At a selective venue, weak
claim support and novelty outweigh the submission's reproducibility strength.

### Simulated independent reviews

| Reviewer | Focus | Overall | Confidence | Acceptance estimate | Primary reason |
| --- | --- | ---: | ---: | ---: | --- |
| R1 | Novelty and significance | 3/10, reject | 4/5 | about 10% | Q-Sparse occupies much of the claimed gap; evidence is smaller and narrower. |
| R2 | Empirical rigor | 4/10, weak reject | 4/5 | 12--25% | One seed, incomplete causal replication at 70M, weak post-hoc comparator, and missing adaptation. |
| R3 | Systems relevance | 4/10, weak reject | 4/5 | 10--20% | The evaluation stops at logical opportunity; model-specific latency remains unmeasured. |
| Meta-review | Combined | weak reject | high | 10--20% | Strong research hygiene, insufficient novelty and empirical closure. |

These estimates are correlated because all reviewers examined the same
evidence. They represent plausible reviewer behavior rather than independent
probability measurements. For context, ICLR reported a
[27.4% acceptance rate in 2026](https://blog.iclr.cc/2026/03/31/a-retrospective-on-the-iclr-2026-review-process/).

## Strongest rejection arguments

1. **The problem definition and baselines omit Q-Sparse.** It already
   spans attention, FFNs, batching, multiple training regimes, and larger
   models.
2. **The paper motivates speed and measures logical opportunities.** Sparse
   formats, mask construction, packing, dispatch, memory traffic, kernel
   coverage, and occupancy determine realized speed.
3. **The training-free comparator is uniform clipping.** TEAL's optimized
   block-wise allocation is a stronger baseline, so the present result supports
   a claim against uniform clipping only.
4. **Every headline training condition has one seed.** Full validation controls
   evaluation noise while leaving initialization, data-order, and optimization
   variation unmeasured.
5. **The timing ladder skips the transition from clipping to full pretraining.** Short post-training adaptation
   could recover much of the benefit for far less training cost.
6. **Several ladder contrasts change multiple factors.** At 70M, A4-OL1 versus
   A7-OL1 changes both gate sites and pressure sites. Only the 14M no-pressure
   A4/A7 contrast isolates the added attention operands.
7. **The workload is full-sequence and uncached.** Cached autoregressive
   decoding has different attention reuse, batching behavior, and bottlenecks.
8. **Dropping 410M leaves two small favorable scales.** Appendix disclosure
   preserves the boundary while keeping the principal claim coherent.

## What would materially change the score

Narrative repair alone improves clarity and reviewer trust:

- cite and distinguish Q-Sparse;
- narrow the contribution to matched contrasts plus exact causal-product
  accounting;
- call the baseline "uniform post-hoc clipping" everywhere;
- place 410M in the appendix with the token-budget calculation;
- replace "scales with model size" with "persists at 70M";
- make realized latency an open empirical question;
- describe OL1 as an ablation because its advantage over ordinary L1 is mixed.

These changes plausibly move the acceptance estimate to **15--20%**. They make
the paper defensible, but leave its decisive evidence gaps intact.

The minimum empirical package likely to reach the borderline range is:

1. multiple seeds for selected 14M and 70M contrasts, with uncertainty on the
   frontier;
2. faithful TEAL allocation and Q-Sparse-style baselines under the same data and
   evaluation;
3. a short continued-training adaptation rung;
4. a matched 70M contrast that separates gate placement from pressure sites;
5. sparse-kernel or end-to-end measurements that include packing and report
   cached decoding separately from full-sequence processing;
6. one post-training or downstream robustness test.

The panel estimates **30--45%** acceptance with stronger baselines, replication,
and adaptation evidence. Adding credible model-specific runtime results raises
the panel estimate to roughly **40--55%**. These ranges assume the experiments
support the present hypothesis; adverse or ambiguous results would require a
different claim.

## Recommended paper identity

The strongest current identity is:

> A controlled empirical study of where and when training-time sparsity
> interventions improve the quality--activation-sparsity frontier, evaluated
> with exact operation-level accounting across attention and MLP computation.

The stronger, still untested identity is:

> Training coherent sparse paths across attention and MLP operations improves
> quality at fixed measured inference cost relative to training-free and
> post-training alternatives.

The first statement fits the completed evidence. The second provides a useful
target for the next research phase.

## Bottom line

Dropping 410M from the central result is good editorial judgment. Keeping it in
the appendix is good scientific judgment. The compute-scaling argument explains
why the current 410M run belongs to a different regime, while a longer run would
test the proposed explanation.

This change improves focus and transparency. A competitive submission still
requires a direct comparison with Q-Sparse, a stronger train-free baseline,
replication of the central contrasts, a post-training adaptation rung, and a
connection between the logical metric and measured inference behavior. RL can
remain a future direction until the objective includes discrete routing or
measured deployment cost.

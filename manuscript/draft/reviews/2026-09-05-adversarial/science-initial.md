# Adversarial scientific review — initial draft

Reviewed 5 September 2026. Scope: `introduction.tex`, `related-work.tex`,
`positioning.md`, `supplementary.md`, operational definitions and metrics, and
Analysis 013's evidence boundaries. This evaluates the argument in an
introduction/related-work draft before the results stage; it is not a rating
of an unseen finished paper. No new experiments or result claims are proposed.

## Rubric and scores

Each dimension uses the same anchors: **1 = blocking**, **2 = major revision**,
**3 = competent but uncompetitive**, **4 = strong conference standard**,
**5 = exceptional**. Scores are reviewer judgments, not calibrated acceptance
probabilities.

| Dimension | Score | Evidence and assessment |
| --- | ---: | --- |
| Scientific relevance and practitioner utility | 3/5 | Inference cost and learning sparsity-aware representations are relevant. However, the opening explains why activation sparsity matters, not why this particular intervention study is necessary. The reader is left to infer which design decision the study will improve. The final list of empirical questions describes work rather than a valuable intellectual payoff. |
| Novelty relative to nearest work | 2/5 | Separating pressure, gates, and sites is a useful organizing lens but not sufficient novelty. Prior papers already ablate pressure schedules, gate functions, gradient estimators, and sparsity allocation. The draft must identify the narrower unresolved relationship and explain why existing results do not settle it. |
| Rigor, identifiability, and claim calibration | 3/5 | The distinction between zero counts and runtime, fixed-budget transfer limits, and topology reach versus realized sparsity is careful. But the central question advertises pressure effects conditional on gate nonlinearity, while Analysis 013 explicitly says A4/A7 pressure contrasts change both gates and the pressure objective. A clean isolated gate-by-pressure interaction is not identified by that comparison. |
| Explanation specificity and falsifiability | 2/5 | Pressure changes magnitudes, a threshold determines zeros, and operations determine product counts: these are useful definitions/accounting relationships, not yet a nontrivial explanation of learned behavior. The account gives no expected sign, boundary, or condition under which pressure should improve or worsen quality at a given opportunity. Optimization interference is introduced without stating what it would distinguish. |
| Transfer and execution argument | 2/5 | Transfer is posed broadly but currently means selected recipes at larger sizes in one family under a fixed token budget. The runtime paragraph names packing/fusion without explaining the workload dependence. Full-sequence zero-product fractions cannot by themselves explain memory-bound cached decoding. Agentic optimization is adjacent and presently lacks a distinct empirical role. |
| **Total** | **12/25** | **Major revision to the argument; several remaining scientific concerns depend on evidence rather than prose.** |

## Strongest plausible rejection rationale

The paper is positioned as an empirical explanation of sparse training, but the
introduction currently presents a taxonomy of familiar interventions, a
logical-product accounting metric, and a list of questions. It does not yet
identify a surprising empirical regularity or a predictive explanation that
existing sparsification papers lack. The key cross-topology comparison also
changes the regularization target, so it cannot isolate the advertised
gate-specific interaction. Selected one-seed size extensions and unsuccessful
runtime ports would delimit individual recipes, but would not on their own
establish a transferable mechanism. A reviewer could therefore see substantial
experimental work without a sufficiently distinct scientific conclusion.

This is the likely objection to the eventual paper if the argument remains
unchanged. The current user request intentionally precedes results, so the
editorial remedy now is a sharper, honest intellectual contract—not fabricated
findings or claims that the missing evidence already exists.

## Five ranked actions

1. **Sell the decision that the study can inform.** Put a concrete dilemma in
   the first two paragraphs: when sparsity is insufficient, should a researcher
   strengthen activation pressure, change the gate, or expand the targeted
   operations? A local FFN zero rate does not answer that decision. Explain why
   an intervention study can identify where pressure buys useful zeros, where
   a topology restricts accessible work, and why apparently similar recipes
   need different explanations. This is prose-fixable as a motivation and goal;
   any actual recommendation remains evidence-dependent.

2. **Narrow the novelty and align it with the comparisons.** Replace the broad
   implication that prior recipes never separate ingredients with the precise
   unresolved question: the marginal response to activation pressure under
   specified gates and pressure targets, connected to the operations reached.
   Do not describe A4-OL1 versus A7-OL1 as isolating gate nonlinearity. The ladder
   is a map of contrasts, not a factorial design or a proof that adding modules
   improves performance. Wording is fixable now; fixed-objective attribution is
   limited by the existing evidence.

3. **Separate an explanatory framework from a demonstrated explanation.** A
   gate formula explains which values are removed, not why training moves
   density near the threshold or preserves quality. Name the quantities whose
   joint behavior would support the account: the redistribution of activation
   mass relative to the threshold, its associated quality cost, and its
   contributions to affected operations. Do not imply that marginal histograms
   establish causal mediation. The present prose can state an explicit
   hypothesis and its evaluative purpose; a nontrivial predictive relationship
   and independent transfer check are evidence-dependent.

4. **Recover introduction space from methods.** Move gate semantics, the dense
   head and causal-product denominator, exact architecture sizes, the detailed
   ladder sequence, and the OL1 optimizer update to methods. Retain a short
   plain-language description of a shared operation-weighted measure and the
   reason local zero fractions are insufficient. Consider calling the measure
   “logical-product sparsity” or “zero-product opportunity” rather than
   redefining the heavily overloaded term “model-wide sparsity.” This is
   prose-fixable and would let the introduction emphasize the intellectual
   stakes without increasing length.

5. **Give inference speedup a causal chain of its own.** In the requested two
   paragraphs, first explain what prior systems actually make possible:
   skipping weight reads or arithmetic through sparse selection, layout,
   packing, and fusion. Then explain portability: token-dependent masks,
   batching/reuse, shape, dtype, and accelerator behavior change the balance
   between avoided work and overhead. End with the study's role: testing whether
   a learned sparsity pattern can be exploited in a specified workload. Agentic
   coding can be an engineering mechanism for adaptation, not evidence that
   this translation will succeed. This structure is prose-fixable; claims of
   speedup, general portability, or correlation with `R_model` require results.

## Strongest defensible thesis and reader payoff

Suggested thesis for this argument-development stage:

> Designing activation sparsity requires understanding how a training
> intervention changes the useful zero pattern, not only how many activations
> it suppresses. We study the conditional value of activation pressure across
> gate and site choices, connect local responses to the operations they affect,
> and examine which parts of that account remain informative at larger model
> sizes and under sparse execution.

The promised payoff should be concrete: a reader learns how to distinguish
three possible bottlenecks in a sparsification recipe—insufficient movement
toward the gate's zero region, limited operations reachable through the
chosen sites, and an execution path unable to exploit the pattern. The
decomposition is useful even if a more aggressive recipe fails. Its scientific
value ultimately depends on evidence that it explains a non-obvious
interaction or predicts a boundary of transfer; the decomposition alone is
not the result.

Avoid claiming a universal recipe, a first study of all ingredients, a complete
factorial interaction, a cross-family mechanism, or a new speedup. Avoid making
optimization-gradient handling an independent related-work strand: it is an
implementation detail at the present level of the paper.

## Literature corrections and useful primary evidence

- **Q-Sparse already performs component ablations.** Section 4.1 compares
  top-k against ReLU, removes the straight-through estimator, and reports
  componentwise sparsity trajectories, including QKV and FFN projections.
  Thus “joint sparse training leaves all ingredient effects unexplored” would
  be inaccurate. The narrower pressure-by-placement question is safer.
  [Q-Sparse, Section 4.1](https://arxiv.org/html/2407.10969v1#S4.SS1).
- **ProSparse does examine regularization and gating alternatives.** Section
  4.4 compares its pressure schedule with fixed L1 and regularization-free
  vanilla/shifted ReLU references. Acknowledge that mechanism-oriented
  precedent before distinguishing the present question's scope.
  [ProSparse, Section 4.4](https://arxiv.org/html/2402.13516v4#S4.SS4).
- **TEAL studies allocation and execution together.** Its blockwise greedy
  allocation explicitly weights the participating matrices, while its sparse
  GEMV path uses selectively loaded weights, a suitable layout, fusion, and
  work decomposition. This supports a substantial inference argument: zeros
  are useful when they change the operations or transfers that dominate the
  target workload. [TEAL, Sections 4.3–4.4](https://arxiv.org/html/2408.14690v2#S4.SS3).

The absence of a particular ablation in the reviewed versions is not proof
that the entire literature lacks it. State the positive distinction in the
present study instead of an unrestricted priority/absence claim.

## Requested related-work chain

The user's exact structure can support a clear argument:

**Sparsification interventions, three paragraphs:** (1) pretrained
representations tolerate post-hoc feature removal, but tolerance and placement
are heterogeneous; (2) training and pressure can alter that tolerance, with
established schedule/gate ablations and mostly FFN-local pressure; (3)
architecture-wide top-k training establishes feasibility beyond FFNs and
provides gate/STE ablations, motivating the narrower pressure response and
operation-level question addressed here.

**Inference speedup, two paragraphs:** (1) published sparse systems show how
specific patterns reduce dominant memory traffic or computation; (2) transfer
to another architecture/workload requires correct, profiled adaptation because
sparsity percentage omits pattern, shape, batching, and overhead. This makes
execution a test of practical usefulness rather than a numerical consequence
of the logical count.

## Re-review conditions

For the revised draft, a strong score requires an explicit useful dilemma, a
precise positive distinction from nearest work, an evidence-calibrated
explanatory contract, and a developed execution argument. Results-dependent
issues will remain separate from the editorial score. No files other than this
review were modified by this reviewer.

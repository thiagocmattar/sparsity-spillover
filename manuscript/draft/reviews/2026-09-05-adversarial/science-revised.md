# Adversarial scientific review — revised draft

Reviewed 5 September 2026 by the same independent scientific reviewer as
`science-initial.md`. Scope: the revised introduction and related work. The
rubric is unchanged. Scores concern argument quality and scientific positioning
at the explicitly requested pre-results drafting stage, not acceptance odds or
the adequacy of an unseen final results section.

## Rubric and scores

Shared anchors: **1 = blocking**, **2 = major revision**, **3 = competent but
uncompetitive**, **4 = strong conference standard**, **5 = exceptional**.

| Dimension | Revised score | Initial score | Assessment |
| --- | ---: | ---: | --- |
| Scientific relevance and practitioner utility | 4/5 | 3/5 | The opening now identifies concrete decisions: combining interventions, investing training, and prioritizing operations for kernel development. The pressure/placement dilemma is meaningful to a sparsification researcher. The closing gives an intellectual payoff rather than merely enumerating planned experiments. |
| Novelty relative to nearest work | 3/5 | 2/5 | The draft now acknowledges prior component studies and positively distinguishes marginal pressure responses under thresholding and placement. That is a credible study direction. The eventual novelty still hinges on which conditional empirical regularity is discovered, because the conceptual factorization itself is familiar. |
| Rigor, identifiability, and claim calibration | 4/5 | 3/5 | Separately controllable choices are correctly distinguished from separable effects. The text asks about individual/joint effects without claiming a completed factorial identification. It does not invent transfer or positive runtime findings. The explanatory obligation is stated as an obligation. Some wording about topology reach versus realized zeros could be more exact, as described below. |
| Explanation specificity and falsifiability | 3/5 | 2/5 | The draft distinguishes magnitude change, zero production, operation reach, and quality cost, and identifies useful competing reasons for limited gains. This is clearer and more actionable than the initial gate-semantics paragraph. It remains an explanatory framework rather than a predictive explanation. That limit is now honestly visible and is partly deliberate because the results stage is deferred. |
| Transfer and execution argument | 4/5 | 2/5 | The two inference paragraphs explain memory traffic, packing/fusion, overhead, dense residual work, and workload dependence. They distinguish batch-one decoding from full-sequence forward throughput and keep agentic coding as an adaptation tool. The introduction frames transfer as something an explanation must account for, without claiming that this has been established. |
| **Total** | **18/25** | **12/25** | **Substantially stronger and scientifically coherent positioning. Remaining limits to a high-impact paper are mainly the specificity and evidence for the eventual empirical insight.** |

## Is the payoff meaningful?

Yes. The revised argument promises to help researchers distinguish whether a
recipe is limited by the learned response to pressure, by which operations its
selected sites can affect, or by the execution costs of exploiting the resulting
zeros. That distinction is useful for deciding what to change next. The
combination of matched intervention effects and explicit operation accounting
could support a strong empirical paper if it identifies a non-obvious pattern
that would be obscured by headline FFN zero fractions.

However, no score should suggest that the existing decomposition already is
such a scientific result. It becomes an explanatory contribution when the
paper establishes a conditional regularity—for example, a boundary at which
an intervention's marginal benefit changes—and shows that the account predicts
or explains behavior beyond the contrast used to formulate it. The exact
regularity should come from the approved evidence and later results writing,
not from additional promotional language now.

## Strongest remaining rejection rationale

The finished paper could still be rejected if the promised explanation reduces
to three known facts: pressure changes magnitudes, gates create zeros, and
different sites affect different amounts of computation. A reader needs to
learn a substantive dependency that cannot be obtained from those definitions.
Likewise, a carefully measured case study is not automatically a transfer claim
if larger-model comparisons change pressure targets or if increased aggregate
sparsity follows architecture weights. The revised introduction no longer
claims those issues are solved; it sets up the correct obligations.

## Remaining issues, ranked

1. **The empirical insight is still to be supplied.** The motivation and
   conditional question are strong enough for this drafting stage. The
   contribution paragraph of the finished paper will need the actual
   evidence-backed answer and its practical implication. This is
   evidence-dependent and should not be repaired by asserting a result now.

2. **Preserve the attribution boundary during results drafting.** The ladder
   maps separately specified choices, but A4/A7 pressure comparisons still
   condition on different pressure objectives. “Individual and joint
   intervention effects” is acceptable as a study description when tied to
   matched contrasts, but later text must not relabel a recipe interaction as
   fixed-objective gate-by-pressure identification. This is an evidence
   boundary, not a reason to add experimental minutiae to the introduction.

3. **Make reach-versus-observation language slightly more exact.** The sentence
   “It also separates architectural opportunity from the extent to which
   training realizes it” can imply that observed `R_model` measures utilization
   of `R_model_max`. Operationally, the observed count includes natural zeros
   outside selected-site reach. A compact alternative is: “It distinguishes
   the operations a topology can reach from the zero products observed in the
   trained model.” This is a prose precision fix, not a material scientific
   overclaim in the current context.

4. **Retain a narrow literature gap.** “What remains to understand is the
   conditional value of combining them” is acceptable as a motivating question,
   but can be read as a broad literature absence claim if amplified. Prefer a
   positive formulation such as “These findings motivate a closer examination
   of the conditional value of combining them.” The related-work section now
   correctly acknowledges Q-Sparse's nonlinearity/STE ablations and ProSparse's
   schedule studies. Do not erase that credit in subsequent shortening.

5. **Treat `model-wide sparsity` as a declared quantity, not a universal unit.**
   The text defines it, and the TEAL number is properly labeled with different
   projection-based accounting. “Zero-product opportunity” remains the less
   ambiguous term when comparing models, architectures, and execution
   workloads. This terminology issue is minor in the current introduction but
   will matter in figure axes, cross-paper comparisons, and results claims.

## Exact user-requested structure

The revised related work satisfies the requested two-section chain:

- **Sparsification interventions:** three paragraphs progressing from post-hoc
  tolerance, through learned FFN sparsity, to broader placement and the
  conditional pressure question.
- **Inference speedup:** two paragraphs progressing from demonstrated
  execution mechanisms to workload-specific adaptation and decisive runtime
  validation.

The gradient-surgery strand is removed. Experimental minutiae are substantially
reduced. The inference discussion is now substantial enough to explain why
execution belongs in the argument, without claiming new acceleration.

## Material overclaim check

No new empirical conclusion or material unsupported speedup/transfer claim was
introduced by the revision. The existing distinction between selected-site
reach and all observed zeros should be preserved with the precision edit above.
As before, the strongest novelty or mechanism claim cannot be established by
the introduction alone. The current text is an appropriate scientific
contract for the next stage rather than an assertion that the contract has
already been fulfilled.

Only this review file was modified by this reviewer.

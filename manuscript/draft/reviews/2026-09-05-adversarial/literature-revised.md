# Revised adversarial literature and systems review

Reviewed on 2026-09-05 by the same independent sub-agent. Compared the revised `introduction.tex` and `related-work.tex` against the preserved `before/` snapshot. No manuscript files edited. The score evaluates this argument-first drafting stage; it does not certify a publishable empirical paper without results. The initial revision scored 19/25. A further passage check after the root agent addressed the systems and category issues gives the final assessment below.

## Rubric and score

Same anchors and weights: 1 = blocking; 2 = major revision; 3 = competent but uncompetitive; 4 = strong; 5 = exceptional. Equal weighting, maximum 25. These are editorial judgments, not acceptance probabilities.

| Criterion | Initial | Revised | Assessment |
|---|---:|---:|---|
| Citation fidelity | 4/5 | 4/5 | Q-Sparse and Spark ablations are now acknowledged. TEAL accounting and workload are qualified. The category-list ambiguity was resolved. |
| Logical progression | 2/5 | 4/5 | Post-hoc tolerance leads to learned adaptation, broader reach, then execution. The requested structure is satisfied. |
| Novelty fairness | 3/5 | 4/5 | The pressure-dependent question is now situated after existing component studies. One introductory gap sentence is broader than necessary. |
| Systems depth and workload specificity | 2/5 | 4/5 | The final passages distinguish decoding from full-sequence throughput and explain batching, mask variation, Block Q-Sparse, TwELL, and overheads. Agentic tooling occupies one bounded sentence. |
| Reader utility and economy | 2/5 | 4/5 | The first paragraph names decisions the study could improve; protocol specifics and the gradient detour are removed. |
| **Total** | **13/25** | **20/25** | **A substantially stronger position; one scope tightening remains.** |

## What the revision resolves

- Exactly three prose paragraphs appear under “Sparsification interventions” and exactly two under “Inference speedup.”
- The central framing now distinguishes separately controllable choices from potentially inseparable effects. The study concerns individual effects, interactions, and computation reached; this correctly reflects the user's clarification.
- The opening gives the reader a reason to care: which ingredients to retain, where training effort helps, and which operations warrant kernel development.
- The literature no longer implies that prior work ignored components. Q-Sparse's nonlinearity and STE ablations and Spark's predictor study are specifically credited.
- TEAL's 50% is explicitly described as projection-based accounting. It is not equated with this paper's `R_model`.
- TEAL decoding latency and TwELL full-sequence forward throughput are distinguished. The introduction also reserves actual exploitation of counted opportunity for direct timing.
- No transfer success, measured speedup for this study, or proved explanation has been invented.

## Issues identified during revised review and final disposition

### 1. Replace the agentic detour with one more systems mechanism

**Resolved in the subsequent passage check.** The final paragraph explains batching and differing token masks, contrasts Block Q-Sparse with TwELL, and limits agentic generation to one sentence. The reasoning below records the initial revision feedback.

The second inference paragraph allocates three sentences to generated-kernel benchmarks and the limits of their relevance. That is cautious but inefficient: the reader needs to understand why activation-sparse kernels behave differently across workloads. The first paragraph currently contains nearly all substantive systems content.

A clearer two-paragraph division would be:

1. **Decoding and weight movement:** TEAL's batch-one sparse GEMV path, optional Spark decoding evidence, and why saved weight traffic depends on mask patterns. Different tokens can activate different weight channels; batching can therefore reduce the traffic saved by any one token's zeros. Spark's appendix B explicitly discusses the union of per-token masks for memory loading. [TEAL](https://arxiv.org/html/2408.14690v2); [Spark](https://arxiv.org/html/2506.06644v1).
2. **Batched products and kernel fit:** TwELL's tile-aligned packing and FFN fusion address a different execution regime; retain its full-sequence throughput qualifier. Block Q-Sparse is a useful example of changing selection structure for batched execution. Conclude that accounting locates opportunity while correctness and complete-model timing establish its realized value. An agentic workflow, if retained, needs at most one sentence as the means of adaptation. [TwELL](https://arxiv.org/html/2603.23198v2); [Q-Sparse, section 2.1](https://arxiv.org/html/2407.10969v2).

This would make the second paragraph add a technical reason rather than a miniature survey of tools. Do not describe Block Q-Sparse as having demonstrated end-to-end latency gains; its paper primarily reports activated-parameter/FLOP efficiency and quality.

### 2. Disambiguate the computational objects in intervention paragraph three

**Resolved in the subsequent passage check.** The manuscript now explicitly assigns internal attention-operand feature coordinates to this study's ladder.

Current sentence: “These mechanisms act on different computational objects: sparse projection inputs, sparse coordinates in attention operands, and sparse attention positions expose different work to skipping.” Its antecedents are Q-Sparse and Spark, but the middle object belongs to this study's intervention ladder. A reader could misattribute operand-coordinate gates to those predecessors.

Suggested replacement: “Q-Sparse's projection-input masks and Spark's token-position selection affect different operations; both also differ from zeroing feature coordinates of query, key, or value operands inside attention.” Then continue with the study's pressure question. The exact distinction is supported by Q-Sparse section 2.1 and Spark section 2, cited above.

### 3. Narrow the introductory gap sentence

“What remains to understand is the conditional value of combining them” can still sound like a claim that conditional effects have not been studied anywhere. The next paragraph is substantially more precise. Prefer “We ask how the additional value of activation pressure changes with the gate rule and the sites where pressure and gates act.” This is an editorial scope correction: it states this study's question without asserting a literature-wide absence.

**Only remaining material wording fix at the final passage check.** A less repetitive replacement is: “This evidence motivates examining how the value of activation pressure changes with thresholding and placement.” No further material literature or systems issues were identified. The final one-sentence reference to iterative generation and profiling is supported by the [agentic optimization workflow](https://arxiv.org/html/2608.14560), and does not assert acceleration of this study's activation patterns.

## Optional wording polish

- “Post-hoc thresholding establishes how much sparsity an already learned representation can tolerate” is broader than any particular thresholding method can establish. “Tests how much sparsity…” or “reveals sparsity that…” avoids implying the method identifies an absolute tolerance limit.
- “Their local FFN zero fractions leave a further accounting question” should clearly refer to the reported statistic, not imply those papers never analyze computation or execution. “A local FFN zero fraction alone leaves…” makes that distinction economical.
- The payoff is now clear enough that the final introduction paragraph can lose one of its two closing payoff sentences if space is needed later for the result preview.

## Remaining evidence boundary

The revised introduction communicates why the experiment is worth reading. It still cannot state the novel empirical insight because the result stage has deliberately been deferred. At that stage, replace some prospective language with one precise interaction, an explanation that discriminates alternatives, and a named transfer result or failure. The present revision should not be scored as a completed ICLR contribution simply because its motivation is now strong.

# Adversarial literature and systems review

Reviewed on 2026-09-05 by an independent sub-agent. Scope: the initial `introduction.tex`, `related-work.tex`, and `supplementary.md`; primary-source checks below. This reviews positioning and attribution, not the validity of unreported experimental results.

## Rubric and score

Common anchors: 1 = blocking; 2 = major revision; 3 = competent but uncompetitive; 4 = strong; 5 = exceptional. Equal weighting, maximum 25. Scores are editorial judgments, not estimates of acceptance probability.

| Criterion | Score | Reason |
|---|---:|---|
| Citation fidelity | 4/5 | Most statements are careful and accurate; source distinctions are substantially better than the supplementary notes. Existing component ablations need fuller acknowledgment. |
| Logical progression | 2/5 | Five topical buckets dilute the intended progression. The gradient paragraph interrupts the argument, and several paragraphs repeat the study's purpose. |
| Novelty fairness | 3/5 | The narrow pressure-by-gate-by-site question is plausible; framing bundled recipes as insufficiently explained risks understating existing analyses. |
| Systems depth and workload specificity | 2/5 | Packing and fusion are named, but the different bottlenecks of decoding and full-sequence forward execution remain unexplained. |
| Reader utility and economy | 2/5 | The reader gets a careful taxonomy before learning what decision the study can improve. Technical caveats occupy space needed for the scientific payoff. |
| **Total** | **13/25** | **Major revision of argument, not merely sentence editing.** |

## Main adversarial judgment

A knowledgeable reviewer could ask: “All these ingredients already exist, several have been ablated, and the manuscript has replaced a method contribution with a list of comparisons. What will I understand afterward that I do not understand now?” The introduction must answer in terms of a decision: when does adding pressure buy useful sparsity at a given quality cost, and when should a researcher instead change the gate or its placement? This is an editorial inference about a credible payoff, not a result established by the draft.

The manuscript should sell an explanation of conditional effectiveness. A controlled interaction matters if it explains why a recipe succeeds in one configuration and fails in another, and if the explanation predicts an additional setting. Merely separating factors or introducing another aggregate statistic is insufficient for the promised empirical-science positioning.

## Source-supported corrections and boundaries

1. **Acknowledge prior component ablations explicitly.** Q-Sparse section 4.1 compares top-k with ReLU and STE with ordinary differentiation; figures 8–9 track both quality and component sparsity. It also reports overall sparsity and activated parameters. Thus, neither “first decomposition” nor “earlier methods only report local zeros” is fair. Its mask is on inputs to learned linear projections, not a mask on feature coordinates of Q/K/V within attention products. The narrower open question is pressure's marginal effect conditional on gate and pressure placement. [Q-Sparse, sections 2 and 4.1](https://arxiv.org/html/2407.10969v2).

2. **Treat TEAL as evidence for allocation as well as feasibility.** It studies input versus output sparsity and optimizes projection allocation within blocks. Its end-to-end result is single-batch autoregressive decoding, using sparse matrix-vector kernels to avoid loading unused weight channels. Its whole-model label is not the same denominator as this draft's full-sequence product accounting. The released inference path documents restricted support; this is evidence of a particular implementation scope, not proof that its method fundamentally cannot transfer. [TEAL, sections 4–5](https://arxiv.org/html/2408.14690v2); [official implementation](https://github.com/FasterDecoding/TEAL).

3. **Give ProSparse's analyses their due.** Its progressive regularization is an intentional response to quality loss from abrupt distribution changes. It combines ReLU replacement, continued training, a changing L1 coefficient, and threshold shifting; the headline sparsity is FFN-local. Calling its coefficient dependence merely a limitation misses a relevant precedent: training dynamics matter. Its PowerInfer speedup and its separate exact GPU operator results have different scopes; do not combine their numbers. [ProSparse](https://arxiv.org/pdf/2402.13516).

4. **Distinguish TwELL's workload from decoding.** Cetin et al. report full-sequence forward throughput and training throughput, keeping sequence length 2048 and varying microbatch size; the main environment is H100 PCIe. TwELL aligns sparse packing with matrix-multiplication tiles and fuses the gated FFN operations. This is particularly relevant to larger batched products, not interchangeable with TEAL's batch-one latency. Do not claim the kernels fail on Pythia without local, verified evidence. [Sparser, Faster, Lighter Transformer Language Models, sections 3–4](https://arxiv.org/html/2603.23198v2).

5. **Spark is also architecture and systems co-design with ablations.** Its attention mechanism selects token positions; this differs from zero feature coordinates. Appendix C.6 separates statistical top-k from the low-cost predictor and shows the combined architecture affects training quality. Appendix B explains why larger batches lose some weight-loading savings as masks differ across inputs. These are useful precedents for interaction and workload dependence. Keep only the aspect needed by the chain. [Spark Transformer](https://arxiv.org/html/2506.06644v1).

6. **Agentic kernel development is supporting methodology.** A generic benchmark citation cannot establish that an agent will accelerate these activation patterns, nor that higher product sparsity monotonically predicts end-to-end speedup. Until there are relevant results, one bounded sentence about execution-feedback-guided kernel adaptation is enough, if it belongs at all. Avoid a generic agent survey or a promised performance contribution.

## Proposed chain: exactly three intervention paragraphs and two inference paragraphs

### Sparsification interventions

1. **The representation already permits some sparsification.** Introduce post-hoc magnitude thresholding through CATS and TEAL. Explain that allocation across components matters. End with the next question: can learning change this quality–sparsity trade-off?
2. **Learning changes that trade-off, and the intervention matters.** Move through ReLU adaptation, ProSparse's progressive pressure, and the FFN regularization of Cetin et al. Explain what these establish about adaptation and training dynamics. End with why a high local FFN zero fraction leaves the computational reach of the intervention unresolved.
3. **Broader reach exists; the remaining question is conditional effectiveness.** Credit Q-Sparse's projection-wide training and component ablations, optionally one clause distinguishing Spark's token selection. State the precise gap: how pressure's effect changes with gate and placement, evaluated under matched training and common whole-forward accounting. Do not claim that nobody has examined components.

### Inference speedup

1. **Decoding: sparse execution can avoid weight movement.** Explain why TEAL's sparse GEMV path accelerates batch-one decoding. Use one workload-qualified result if a number genuinely helps. Explain that token-dependent masks reduce shared weight reuse as batching increases; local zeros alone do not identify the latency gain.
2. **Full-sequence execution: sparse work must fit efficient GPU execution.** Explain TwELL's tile-aligned packing and fusion in batched FFNs. Connect tensor shape, sparsity distribution, layout, and integration overhead to portability. Close on the paper's complementary measurements: product accounting locates logical opportunity; benchmarks determine which opportunity the target execution actually realizes. Mention kernel adaptation only in service of this question.

## Introduction structure lessons from established ICLR papers

I inspected LoRA and FlashAttention-2 as structural examples, without claiming a citation ranking.

- **LoRA:** a concrete deployment burden leads to the costs left by existing solutions, then to a compact hypothesis and its practical consequences. Reusable move: connect the proposed scientific object to a decision the reader already faces before introducing its notation. For this paper, that is the choice between stronger pressure, a different gate, and broader placement. [LoRA introduction](https://arxiv.org/html/2106.09685v2).
- **FlashAttention-2:** acknowledge an effective predecessor, identify a residual measurable gap, diagnose why that gap remains, and align the proposed changes with the diagnosis before showing broader impact. Reusable move: make each contribution answer a named cause rather than listing activities. This draft can follow that logic without copying an engineering paper's emphasis on headline speed. [FlashAttention-2 introduction](https://arxiv.org/pdf/2307.08691).

Recommended first-principles sequence: consequential choice → what existing evidence already resolves → remaining conditional question → explanatory lens → how matched comparisons discriminate explanations → reader payoff and transfer test. The current draft starts plausibly but spends too much of its second half explaining the protocol and defending measurement boundaries. Move model sizes, data names, detailed denominator coverage, and optimization implementation to methods.

## Acceptance conditions for revision

- A reader can state the study's practical scientific payoff after the first two paragraphs.
- Existing allocation studies and component ablations are acknowledged before narrowing the gap.
- Related work has the requested 3+2 structure, with each paragraph advancing the same argument.
- Decoding latency and full-sequence throughput are separated explicitly.
- No paper-derived sparsity percentage is equated with this manuscript's product statistic; no unmeasured speedup is promised.
- No result, transfer success, or causal mechanism is invented to strengthen the pitch.

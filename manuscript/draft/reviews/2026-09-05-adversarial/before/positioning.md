# Argument and source notes

Working interpretation of the user's 5 September 2026 request. This is an
editorial record, not an approved experiment design or a results section.
`supplementary.md` is preserved unchanged.

## Position

The paper asks how the effect of activation pressure depends on gate
nonlinearity and pressure placement. Its intended empirical contribution is
an isolated interaction, an explanation connecting optimization to activation
changes and affected operations, and evidence that the explanation predicts
behavior in additional settings. The intervention ladder organizes this
question; it is not itself evidence of a monotone improvement.

The introduction follows the older draft's progression: computational
motivation, successful prior approaches, unresolved question, explanatory
account, model-wide measurement, setup, and the paper's three questions.
It ends with questions rather than completed-result contributions because the
user requested argument development before results. Replace this ending with
precise evidence-backed contributions when that stage is approved.

The candidate explanation is that pressure changes activation magnitudes,
gate semantics determine conversion into exact zeros, and the affected
operations determine their product-weighted value. Optimization interference
may help explain the associated quality cost. These are interpretive
hypotheses, not established mechanisms. Gate formulas alone explain mask
semantics; they do not establish the cause of a learned quality difference.

## Evidence boundaries to preserve in the next stage

| Argument | Current boundary / evidence needed |
| --- | --- |
| A clean pressure-by-gate interaction | [Analysis 013](../../analyses/013-2026-09-04-matched-intervention-manuscript/README.md) explicitly records the missing A7+OL1@4 contrast. A4-OL1 versus A7-OL1 changes the gate set and the pressure objective. Within-topology pressure effects are interpretable, but differences between those effects do not isolate gate nonlinearity at a fixed objective. |
| An explanation of the interaction | Link pressure updates, activation exact/near-zero mass and scale, and operation contributions for matched interventions. Activation marginals and endpoint loss alone do not establish a causal route. This draft does not select or authorize new experiments. |
| Transfer of the explanation | [Analysis 011](../../analyses/011-2026-09-03-pythia14m-70m-410m-selected-ladder/README.md) supplies selected one-seed, fixed-token extensions through 410M. They are not a complete factorial replication or evidence of a scaling law. The denominator changes with architecture; compare operation-level behavior as well as the aggregate metric. No longer-training or cross-family transfer is claimed. |
| OL1 as a controlled pressure variant | The [operational method](../../research/METHODS.md) projects relative to AdamW's adaptive direction and caps the correction. It does not guarantee unchanged loss, maximal loss-preserving regularization, or elimination of the pressure-weight degree of freedom. Trust saturation can limit the effective correction only at boundaries where the cap binds. |
| Model-wide sparsity | Use the existing count-first `R_model` contract. The numerator counts zero-operand products in all six block families, including natural zeros; the denominator retains the dense LM head. The full-sequence workload is not cached decoding. |
| Architecture ceiling | `R_model_max` is the all-zero selected-site reach ceiling, not a universal upper bound on all observed zeros. Natural zeros outside selected reach can contribute to `R_model`; A0 can have a positive measured value despite zero selected-site reach. |
| Runtime realization | [Analysis 014](../../analyses/014-2026-09-04-pythia70m-vs-410m-sparse-kernel-sentinels/README.md) records negative matched systems evidence. The notes' proposed agentic adaptation is pending; the manuscript claims no new kernel, speedup, or correlation between speedup and `R_model`. |

Spillover is omitted from the central claim, following the user's instruction
that it did not persist in larger/longer settings. The old introduction's
F001-backed statement is not reused. Some passages in `research/MANUSCRIPT.md`
and historical manuscript records still describe 410M as unobserved; completed
Run 019 and Analysis 011 supersede that cutoff. No historical record or finding
registry was rewritten for this task.

## Literature checks

Sources checked on 5 September 2026. Bibliography entries use primary paper
records; preprint years are retained where conference metadata was not checked.

- [TEAL, Sections 4.3 and 5.4](https://arxiv.org/html/2408.14690v2): acknowledge
  optimized allocation and input/output placement studies. Our uniform
  TEAL-style reference is not the full method.
- [ProSparse, Sections 3 and 4.4](https://arxiv.org/html/2402.13516v4): acknowledge
  its schedule ablations rather than claiming that it never separates components.
- [Q-Sparse, Sections 2 and 3](https://arxiv.org/html/2407.10969v1): distinguish
  attention/FFN projection inputs from internal attention operands; acknowledge
  overall sparsity and activated-parameter reporting.
- [Sparser, Faster, Lighter, Sections 2--4](https://arxiv.org/html/2603.23198v1):
  cite both the FFN training intervention and its integrated execution design.
- [Spark Transformer, Section 2 and Appendix C](https://arxiv.org/html/2506.06644v2):
  distinguish sparse attention positions from sparse operand coordinates.
- [CATS](https://arxiv.org/abs/2404.08763): retain both its training-free and
  fine-tuning settings.
- [ReLU Strikes Back](https://arxiv.org/abs/2310.04564) and
  [Sparsing Law](https://proceedings.mlr.press/v267/luo25i.html): acknowledge
  prior work on activation choice and training/scale dependence.
- [PCGrad](https://arxiv.org/abs/2001.06782) and
  [Bloop](https://arxiv.org/abs/2402.02998): credit both conflict projection
  and the closer primary/auxiliary-objective precedent.
- [KernelBench](https://arxiv.org/abs/2502.10517) and
  [Agentic Kernel Optimization](https://arxiv.org/abs/2608.14560): keep the
  agentic discussion brief and tied to correctness and measured performance.
  The latter's landing page has an inconsistent month between its identifier
  and displayed submission date; the bibliography records only the verified
  title, authors, identifier, and year.
- [GEAK v4](https://www.amd.com/en/developer/resources/technical-articles/2026/geak-v4.html)
  was consulted but is omitted from the compact section. A broader agent
  survey would distract from the current empirical question. The unlinked
  CudaForge/CUDA Agent examples in the notes are not asserted or cited.
- [Pythia](https://arxiv.org/abs/2304.01373) and
  [MiniPile](https://arxiv.org/abs/2304.08442) identify the architecture family
  and corpus; our random initialization and protocol come from repository
  records, not the released Pythia training recipe.

## Size and preview

The section text is approximately 760 words of introduction and 560 words of
related work, excluding bibliography and the setup caption. The aim is roughly
2.5--3 ICLR text pages, reserving the bulk of the paper for methods, explanation,
and transfer. This is a planning allowance, not a template-fit guarantee.
The [2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
state nine pages for the initial main text and exclude references; later
FAQ/camera-ready language on the same page inconsistently calls the submission
limit ten pages. We use the explicit initial-submission rule conservatively.

`main.tex` is a simple reading wrapper, not a submission template. It places
the existing architecture/ladder artifact on a separate review page after the
two sections and then lists references. Its caption makes clear that the
ladder is not an execution-coverage table. The final paper's figure placement,
title, contribution paragraph, and result preview remain for later drafting.

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

After the user's adversarial-review request, the introduction now begins with
the decision a researcher faces: how to combine and place interventions to
obtain useful sparsity at an acceptable quality cost. It then credits existing
success and component studies, narrows to pressure's conditional effect,
connects learned responses to affected work, introduces accounting for that
purpose, and closes with the scientific payoff. The compact methodology now defines interventions and measurement. The
separate experimental section introduces concrete recipes; its appendix holds
architecture specialization and validation coverage.
The eventual result preview and evidence-backed contributions remain deferred.

The user's additional external feedback supplies the central distinction:
the choices are separately controllable, but their effects need not be
separable. A study must distinguish individual effects, interactions, and the
computation the resulting zeros can affect. This is an explanatory question,
not a claim to have identified every factor interaction in the existing ladder.

The reader payoff is diagnosing whether limited gains arise from the response
to pressure, the reach of the chosen sites, or the cost of expanding the
intervention, then assessing whether sparse execution makes that opportunity
useful. This gives a reason to study interactions beyond completing an ablation
table. The scientific insight still needs an observed conditional relationship
and an explanation that is informative outside its original comparison.

Related work now has two subsections: **Sparsification interventions**, exactly
three paragraphs, and **Inference speedup**, exactly two. The former progresses
from post-hoc tolerance to learned FFN sparsity to broader placement and the
specific pressure question. The latter explains execution mechanisms and then
workload-dependent adaptation. Gradient-projection context is confined to the methods appendix.

Three sub-agents independently reviewed the original and revised text using
explicit five-criterion, 1--5 rubrics. The reports, scored rubric, source
checks, response to criticisms, and original source snapshot are under
[reviews/2026-09-05-adversarial/](reviews/2026-09-05-adversarial/).

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
- [Q-Sparse, Sections 2 and 4.1](https://arxiv.org/html/2407.10969v2): distinguish
  attention/FFN projection inputs from internal attention operands; acknowledge
  overall sparsity, activated-parameter reporting, and nonlinearity/STE
  ablations. The batch-execution paragraph also credits Block Q-Sparse.
- [Sparser, Faster, Lighter, Sections 2--4](https://arxiv.org/html/2603.23198v2):
  cite both the FFN training intervention and its integrated execution design.
  Full-sequence forward throughput is distinct from TEAL's batch-one decoding
  latency. TEAL's quoted sparsity uses its own projection-based accounting.
- [Spark Transformer, Section 2 and Appendix C](https://arxiv.org/html/2506.06644v2):
  distinguish sparse attention positions from this study's operand-coordinate
  gates; credit its predictor ablation and batching analysis.
- [CATS](https://arxiv.org/abs/2404.08763): retain both its training-free and
  fine-tuning settings.
- [ReLU Strikes Back](https://arxiv.org/abs/2310.04564) and
  [Sparsing Law](https://proceedings.mlr.press/v267/luo25i.html): acknowledge
  prior work on activation choice and training/scale dependence.
- [PCGrad](https://arxiv.org/abs/2001.06782) and
  [Bloop](https://arxiv.org/abs/2402.02998): retained as sources for the later
  methods section, with the dedicated related-work paragraph removed as requested.
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

The revised section text is approximately 630 words of introduction and 570
words of related work, excluding bibliography and the setup caption. The aim
is to stay within roughly 2.5 ICLR text pages, reserving the bulk of the paper for methods, explanation,
and transfer. This is a planning allowance, not a template-fit guarantee.
The [2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
state nine pages for the initial main text and exclude references; later
FAQ/camera-ready language on the same page inconsistently calls the submission
limit ten pages. We use the explicit initial-submission rule conservatively.

`main.tex` is a simple reading wrapper, not a submission template. It places
the architecture/ladder artifact at the experimental opening on page 4, then
lists references and general-method/experimental-detail appendices. Its caption makes clear that the
ladder is not an execution-coverage table. The final paper's figure placement,
title, contribution paragraph, and result preview remain for later drafting.

## Methodology stage

The new [methodology](methodology.tex) adds approximately 560 words of main
text, mostly inline math, and [an appendix](methodology-appendix.tex) for
exact semantics. [Methodology notes](methodology-notes.md) map the paper's
calligraphic S to the unchanged operational R fields and identify all sources.
The ceiling describes selected-site reach, including V-to-context closure;
it does not bound natural zeros outside that reach. The main section keeps
OL1 brief and preserves the joint gate/pressure-change interpretation.

The existing three-size ladder supplies the architecture comparison. Its
lettering and table layout were adjusted after rendered review; its values,
rows, and gate placement are unchanged. No redundant ceiling chart or new
cross-family study was added. The three-reviewer methodology record is
[here](reviews/2026-09-05-methodology/README.md). Results, evidence-backed
contributions, and the final submission layout remain the next writing stages.

## Methodology / experimental-study split

The user approved separate sections. Main Methodology now has two subsections:
sparsification interventions; sparsity and architectural reach. It contains no
A* recipe IDs, Pythia sizes, dataset, or execution-coverage claims. The roughly
320-word Experimental Study opening defines those choices and explains the
controlled comparisons, followed later by question-driven findings. Its setup
figure appears after the opening paragraph.

The [new review record](reviews/2026-09-05-methods-experiments-split/README.md)
contains an independent writing proposal and two scored rounds from all three
reviewers. General reach is explicitly structural and independent of natural
zeros or checkpoint values. Pressure uses executed site tensors, after a gate
when present, preserving the independence of gate and pressure target sets.
Across-size interpretation uses operation-level zero fractions and workload
shares alongside aggregate sparsity and selected-site reach.

# Adversarial review and revision

Requested by the user on 5 September 2026. Three sub-agents independently
reviewed the introduction and related work, then re-read the revised text.
The root agent synthesized the critiques, revised the manuscript, and applied
the remaining wording corrections. No results, experiment design, or launch
was part of this task.

## Rubric given to the reviewers

Each reviewer received five equally weighted criteria and explicit anchors:
**1** blocking weakness; **2** major revision; **3** competent but
uncompetitive; **4** strong conference standard; **5** exceptional.
Each report must justify scores, identify high-priority issues, and propose
actionable changes. Reviewers were told to distinguish prose-fixable issues
from evidence-dependent limitations, to respect the argument-first stage,
and not to invent findings. Re-review requests explicitly prohibited assuming
an improvement or aiming for a target score.

| Review perspective | Five scored criteria | Initial | Re-review |
| --- | --- | ---: | ---: |
| Writing | Clarity/precision; argument progression; reader payoff; economy; calibrated persuasion | 13/25 | 18/25 |
| Science | Relevance/utility; novelty; rigor/identifiability; explanation specificity; transfer/execution | 12/25 | 18/25 |
| Literature and systems | Citation fidelity; progression; novelty fairness; systems depth; utility/economy | 13/25 | 20/25 |

These are subjective editorial assessments of these sections at the current
drafting stage. They are not acceptance probabilities or independent evidence
that the eventual scientific claims are established. The criteria differ by
reviewer; no pooled score is reported.

Full reports: [writing initial](writing-initial.md),
[writing revised](writing-revised.md), [science initial](science-initial.md),
[science revised](science-revised.md), [literature initial](literature-initial.md),
and [literature revised](literature-revised.md). The [before/](before/) snapshot
preserves the first draft's sections, bibliography, wrapper, positioning notes,
and compiled PDF. Final sources are in [after/](after/).

## Convergent criticisms and the revision

1. **The first draft described machinery before establishing significance.**
   The opening now names the decisions at stake: which recipe components to
   keep, where to invest training, and which operations warrant kernel work.
   The argument explains why individual benefits cannot determine the benefit
   of a combination.
2. **Separability was ambiguous.** The user's external feedback now supplies
   the core sentence: “The interventions are separately controllable, but
   their effects need not be separable.” The study concerns individual effects,
   interactions, and the computation affected by the resulting zeros.
3. **The explanation was close to a statement of gate definitions.** The
   revised text identifies empirical distinctions: a magnitude change without
   more zeros, zeros at sites with limited reach, and the quality cost of
   expanding the intervention. This makes the explanatory question concrete
   without presenting those possibilities as discovered mechanisms.
4. **The literature gap understated prior analysis.** Q-Sparse's nonlinearity
   and STE ablations, ProSparse's schedule studies, and Spark's predictor study
   are now credited. The gap sentence asks specifically about the additional
   value of pressure as thresholding and placement change; it asserts no
   literature-wide absence of component or interaction studies.
5. **Related work was a catalogue with a gradient detour.** It now has exactly
   three paragraphs under *Sparsification interventions* and two under
   *Inference speedup*. Optimizer details and the dedicated gradient strand
   are deferred to methods.
6. **The execution discussion was shallow.** It now distinguishes TEAL's
   batch-one decoding and projection-based sparsity measure from TwELL's
   full-sequence throughput. It explains weight movement, token-mask variation
   under batching, Block Q-Sparse, packing/fusion, overheads, and remaining
   dense work. Agentic generation occupies one supporting sentence.

The literature reviewer inspected the introductions of
[LoRA](https://arxiv.org/html/2106.09685v2) and
[FlashAttention-2](https://arxiv.org/pdf/2307.08691) as established ICLR examples.
The structural lesson applied was: concrete burden or decision → what prior
work resolves → a precise remaining dependency → explanatory approach →
evidence and practical consequence. No wording was copied and no influence
ranking is asserted.

## Final edits after the scored reads

The writing and science scores above precede the root agent's final small
wording pass. They were not raised on the root agent's authority. The literature
reviewer checked the expanded batching paragraph and computational-object
correction before assigning 20/25.

The final pass removed the duplicated intervention-question sentence and
“The scientific value lies in...” metacommentary, made the opening literal,
and ended on a concrete transfer criterion. It also avoided implying that
observed `R_model` is utilization of `R_model_max`, narrowed the remaining broad
gap wording, described a local FFN fraction as insufficient *by itself*, and
avoided suggesting post-hoc thresholding determines an absolute tolerance
limit. Typography changes removed an overfull line in the inference paragraph.

## Remaining scientific obligation

All three reviewers agree that stronger prose cannot supply the eventual novel
empirical insight. The results stage must identify a non-obvious conditional
relationship, give an explanation that discriminates alternatives, and show
what transfers or fails to transfer. The existing combined A4/A7 gate-and-pressure
change still cannot identify a fixed-objective gate interaction. Source and
scope boundaries remain in [positioning.md](../../positioning.md).

The draft makes these the study's obligations. It claims neither a completed
factorial identification nor a new speedup, and the discarded spillover claim
has not returned.

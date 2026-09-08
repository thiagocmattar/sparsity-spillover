# Independent literature and positioning audit

Reviewed 8 September 2026. Scope: the live draft's introduction, related work,
methodology, methodological appendix, experimental opening, kernel results,
results appendix, bibliography, and positioning notes. No manuscript or
bibliography was edited. The first two introduction paragraphs remain protected.
Primary paper records were checked online; detailed claims were checked against
paper full text or official repositories. This is a citation/positioning review,
not an independent reanalysis of the experimental artifacts.

## Overall assessment

The literature supports the paper's central framing: intervention placement,
training adaptation, and execution overhead jointly determine the usefulness of
activation sparsity. The draft correctly distinguishes local zero fractions,
model-wide product counts, and measured speedup. It also gives unusually useful
boundaries around its uniform clipping reference, OL1 geometry, and runtime
comparison. No missing or mismatched paper was found among the 15 references.

The strongest contribution is the matched analysis of how pressure's additional
effect depends on gates and site placement, followed by the relationship between
learned zeros, affected operations, and measured execution. The ladder, projection
idea, activation regularization, and use of coding agents should remain tools for
that study, rather than separate novelty claims. The current wording mostly does
this already.

Reviewer rubric, 1 = poor and 5 = strong:

| Criterion | Score | Reason |
| --- | ---: | --- |
| Source accuracy | 4.5 | All cited works resolve; one description of Sparsing Law needs narrowing. |
| Fairness to prior work | 4.5 | Existing allocation, schedule, and component ablations are acknowledged. |
| Clarity of contribution | 4.0 | Component comparison should be made explicit immediately after the protected introduction. |
| Novelty boundaries | 4.5 | No general optimizer or agent-capability claim; move OL1 attribution to its first main-text mention. |
| Readability | 4.0 | Several related-work endings repeat the same motivation; a few short replacements suffice. |

## Prioritized paragraph-level changes

1. **Narrow the comparison promised by the introduction.** The protected second
   paragraph naturally creates an expectation of a common-setup comparison of
   TEAL, ProSparse, and Q-Sparse themselves. The experiments instead compare
   selected components, using fixed gates rather than Q-Sparse top-k/STE and
   uniform clipping rather than TEAL's optimized allocation. Add a short sentence
   at the opening of the third paragraph:

   > We compare the underlying intervention choices in matched pretraining runs.

   Then continue with pressure, thresholding, and site placement. The experimental
   section's existing explicit uniform-TEAL-control limitation should remain. Do
   not change either protected paragraph. The facts in those paragraphs are
   supported; the issue is the expectation they create about the later experiment.

2. **Correct the Sparsing Law description.** Its core factors are training data,
   activation function, width/depth ratio, and parameter scale. The current
   "regularization, data, and model size" phrase overstates regularization as a
   studied axis. Suggested replacement:

   > Sparsing Law studies how activation sparsity changes with training data,
   > activation function, and model architecture.

   Evidence: [ICML paper, Sections 1 and 5](https://arxiv.org/html/2411.02335v4).
   Its sparsity measure is quality-constrained near-zero removal, so avoid treating
   its reported percentages as exact-zero measurements using this draft's metric.

3. **Credit OL1's antecedents at first mention.** The appendix correctly relates
   the update to PCGrad and Bloop. In the main methodology, introduce it as:

   > Orthogonal L1 (OL1), a pressure update based on gradient projection
   > [PCGrad; Bloop], makes a task-only AdamW update ...

   Use the existing two citation keys. Retain the precise appendix distinction:
   conflict-conditioned projection in AdamW-preconditioned coordinates, task-only
   moments, and a correction cap. Bloop projects an auxiliary gradient against a
   smoothed main direction; PCGrad removes conflicting components. Neither source
   establishes this particular implementation's novelty or loss preservation.
   [PCGrad](https://arxiv.org/abs/2001.06782),
   [Bloop, Section 2](https://arxiv.org/html/2402.02998v2).

4. **Name P0 and attribute its origin.** The results appendix currently introduces
   P0 as an unexplained ID. Suggested first mention:

   > P0, an early adapter derived from the released kernels of Cetin et al.,
   > qualifies on only four checkpoints.

   Cite `cetin2026sparser`. Retain the subset restriction and describe P0 as an
   adapter; its result is not an unchanged upstream benchmark. The official
   repository confirms the relevant code source, while the local run records own
   the adapter's identity and qualification outcome.
   [Official release](https://github.com/SakanaAI/sparser-faster-llms).

5. **Keep the agent discussion subordinate to the systems question.** KernelBench
   supports iterative execution/profiling feedback; Agentic Kernel Optimization
   supports human-orchestrated coding-agent search with correctness gates. Neither
   supports a claim that the present search measures a general agent capability.
   The live kernel section's "human-guided Codex search" is appropriate. Use
   "implementation" for K050 where possible, since it combines several kernels
   and framework components. [KernelBench](https://proceedings.mlr.press/v267/ouyang25a.html),
   [Agentic Kernel Optimization](https://arxiv.org/abs/2608.14560).

6. **Shorten repeated related-work transitions.** Each of the five paragraphs
   currently returns to quality, useful sparsity, or overhead. Preserve the
   three-intervention/two-speedup structure, but finish each paragraph with its
   distinct point: post-hoc tolerance; learned sparsity; site placement; workload;
   correctness and timing. This strengthens the current chain without adding a
   broader survey.

## Verification of all 15 bibliography entries

Titles and author lists match the linked primary records, allowing ordinary name
abbreviation variants. Existing preprint entries are valid citations, but formal
conference metadata should replace preprint metadata where verified below.
Keeping citation keys unchanged avoids unnecessary TeX edits. Updating the year
field changes the rendered citation year even in protected paragraphs; decide
explicitly whether their protection extends to that rendering.

| Key | Verified source and claim boundary | Bibliographic action |
| --- | --- | --- |
| `liu2024teal` | [Paper](https://arxiv.org/html/2408.14690v3), Sections 4.2-4.3 and 5.3-5.4: magnitude thresholding across attention/FFN projections; optimized allocation; batch-one decoding; reported 1.8x at 50% under its accounting. | Verified ICLR 2025 in the [published PDF](https://openreview.net/pdf?id=dGVZwyq5tV). Prefer `@inproceedings`, ICLR, 2025. Current 2024 preprint is not a false citation. |
| `biderman2023pythia` | [ICML record](https://proceedings.mlr.press/v202/biderman23a.html): architecture family; does not supply this study's random-initialization or MiniPile training protocol. | Change `@article`/`journal` to `@inproceedings`/`booktitle`; PMLR 202, pp. 2397--2430. Official author form is `Anthony, Quentin Gregory`; current omission of the middle name is minor. |
| `kaddour2023minipile` | [Primary record](https://arxiv.org/abs/2304.08442): Jean Kaddour, 2023; dataset motivation and identity. The draft's exact token/block coverage is local evidence. | Current title, author, year, and ID verified; retain. |
| `song2024prosparse` | [Paper](https://arxiv.org/html/2402.13516v7), Sections 3 and 4.4 / Appendix I: ReLU, progressive L1, threshold shifting, and schedule ablations. Describing continuation is valid for LLaMA experiments but does not exhaust its MiniCPM setting. | [Published record](https://aclanthology.org/2025.coling-main.180/): COLING 2025, pp. 2626--2644, ACL publisher. Prefer conference entry; 2024 preprint remains valid. |
| `mirzadeh2023relu` | [Primary record](https://arxiv.org/abs/2310.04564): title and eight authors verified; supports reintroducing ReLU sparsity. | Current preprint metadata verified. Search surfaced ICLR 2024, but the primary conference page/PDF triggered a browser challenge; keep current metadata rather than claim an independently checked conference record. |
| `luo2025sparsing` | [ICML record](https://proceedings.mlr.press/v267/luo25i.html): ten authors; training-data/activation/architecture/scale factors and quality-aware sparsity. | Change to `@inproceedings`/`booktitle`; PMLR 267, pp. 41311--41330. All current author identities match. |
| `cetin2026sparser` | [Paper](https://arxiv.org/html/2603.23198v2), Sections 2.2, 3.2-3.3, 4.1: ReLU plus L1 pretraining; TwELL packing in projection epilogues; fused inference; fixed 2048-token sequence throughput. | Six authors, title, ID, and 2026 verified. Retain preprint metadata; release is the linked [official repository](https://github.com/SakanaAI/sparser-faster-llms). |
| `lee2024cats` | [Primary record](https://arxiv.org/abs/2404.08763): thresholding in gated FFNs, both no-fine-tuning and fine-tuning settings, custom inference kernels. | [Published PDF](https://openreview.net/pdf?id=v3w2a7EInO) identifies COLM 2024. Prefer conference entry. |
| `yu2020pcgrad` | [Primary record](https://arxiv.org/abs/2001.06782): six authors, title, NeurIPS 2020, conflict-conditioned projection. | Current metadata verified; retain. |
| `hsieh2024bloop` | [ICML record](https://proceedings.mlr.press/v235/hsieh24a.html): six authors, EMA primary direction and auxiliary projection. | Prefer ICML 2024, PMLR 235, pp. 19085--19100, `@inproceedings`. |
| `wang2024qsparse` | [Paper](https://arxiv.org/html/2407.10969v3), Sections 2 and 4: top-k on projection inputs, STE, activated parameters, Block Q-Sparse. Attention projection inputs differ from internal Q/K/V feature gates. | Four authors, title, year, and ID verified; retain. |
| `you2025spark` | [Paper](https://arxiv.org/html/2506.06644v2), Section 2 and Appendix C: sparse FFNs and attention positions, low-cost predictor, statistical top-k, batching analysis. | All 19 authors, title, year, and ID verified; retain. Its batching discussion includes CPU measurements; avoid presenting them as GPU results. |
| `ouyang2025kernelbench` | [ICML record](https://proceedings.mlr.press/v267/ouyang25a.html): seven authors; execution/profiling feedback and correctness-plus-speed evaluation. | Prefer ICML 2025, PMLR 267, pp. 47356--47415. Preserve the accented `R{\'e}` from the arXiv author record if desired. |
| `luo2026agentic` | [Primary record](https://arxiv.org/abs/2608.14560): title, five authors, and 2026 match. Human orchestration and execution/profiling support the limited related-work sentence. | Retain. Landing-page submission month conflicts with the identifier's month, as already recorded in positioning.md. Do not add a precise submission date or infer an unverified venue. |
| `loshchilov2019adamw` | [Primary record](https://arxiv.org/abs/1711.05101): two authors, decoupled weight decay, ICLR 2019. | Current conference metadata verified; retain. |

## Scientific boundaries that should survive polishing

- The study tests components and selected recipe comparisons, not a complete
  factorial design or a benchmark ranking of TEAL, ProSparse, and Q-Sparse.
- A4-OL1 to A7-OL1 expands both gate and pressure sets, changing the pressure
  objective and the weights assigned to existing targets.
- The density overlays show responses to complete interventions. They do not
  isolate pressure as the cause of changes at ungated attention sites.
- OL1 is an implementation of a gradient-projection idea with explicit limits;
  the matched results do not establish its general superiority over naive L1.
- Published sparsity percentages use different denominators and sometimes count
  tolerated near-zero removal. Do not numerically compare them to this paper's
  exact zero-product fraction without recomputing a common metric.
- The direct systems result is specific to the declared full-sequence workload
  and hardware. Preserving the main FFN/attention workload distinctions in related
  work prevents readers from comparing its speedup directly with decoding claims.

## Submission-readiness judgment

The literature does not require a new direction or an enlarged survey. Resolve
the six small items above, retain the existing experimental limitations, and use
formal bibliography records where convenient. No scientific correction is needed
inside the protected introduction paragraphs. Citation accuracy is strong enough
for submission after the Sparsing Law phrase is corrected and P0 is identified.
This judgment concerns literature fidelity and framing only; it is not a claim
that this review verified every result, figure, or reproducibility requirement.

## Resolution after paragraph revisions

The revised introduction now explicitly describes a component comparison under
matched pretraining conditions. Related work corrects the Sparsing Law factors,
distinguishes the attention sparsity sites, and gives each paragraph a clearer
role. The first OL1 mention now credits PCGrad and Bloop. Those recommendations
are resolved. P0 was still an unexplained ID in the results appendix at this
follow-up read; root was notified and owns that prose change.

At root's request, `references.bib` now uses verified formal conference metadata
for Pythia (ICML 2023), Sparsing Law (ICML 2025), CATS (COLM 2024), Bloop (ICML
2024), and KernelBench (ICML 2025). Citation keys and years are unchanged.
Pythia's author entry now includes Quentin Gregory Anthony's full given name.
TEAL and ProSparse retain their valid 2024 preprint records so citations in the
protected introduction paragraphs keep their existing years. Other entries are
unchanged. No manuscript build or commit was performed by this reviewer.

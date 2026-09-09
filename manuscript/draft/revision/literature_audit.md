# Primary-source and venue checks, 9 September 2026

Retained claims were checked against primary papers and conference records.
No new citation or external speedup leaderboard was added. Existing preprint
years remain valid bibliographic identities even where later conference versions
exist. Publication titles were not subject to terminology replacement.

| Citation | Primary source | Check used in the revision |
|---|---|---|
| TEAL | https://arxiv.org/html/2408.14690v3 | Sections 4.3/5.4.2 distinguish greedy allocation from uniform allocation; decoding/batching sections support workload boundaries. |
| ProSparse | https://arxiv.org/html/2402.13516v7 | ReLU, progressive pressure and threshold shifting; Section 4.4 Q3 compares fixed/progressive pressure with comparable sparsity and different quality. |
| Q-Sparse | https://arxiv.org/html/2407.10969v3 | Projection-input top-k and STE; Section 4.1 ablations; Block Q-Sparse supports structured batched execution. |
| Spark | https://arxiv.org/html/2506.06644v2 | Statistical top-k, query/key predictor and attention-position selection; Appendix C separates predictor and sparsifying function. |
| TwELL | https://arxiv.org/html/2603.23198 | Sections 3.2/3.3 describe tiled packing and fused FFNs; regularization study supports the retained pretraining statement. Pythia compatibility failures come from our run records, not this paper. |
| CATS | https://arxiv.org/abs/2404.08763 | Thresholding, training-free and fine-tuning settings verified. OpenReview was challenge-protected; retained COLM citation was not changed. |
| ReLU Strikes Back | https://arxiv.org/abs/2310.04564 | Sparse-activation restoration framing; no numerical comparison imported. |
| Pythia | https://proceedings.mlr.press/v202/biderman23a.html | Architecture-family context and bibliographic metadata; our random initialization is verified by original manifests. |
| Sparsing Law | https://proceedings.mlr.press/v267/luo25i.html | Training data, activation and architecture study; ICML2025 bibliographic record. |
| KernelBench | https://proceedings.mlr.press/v267/ouyang25a.html | Execution-feedback kernel generation as context, not proof of our agent's productivity. |
| Agentic Kernel Optimization | https://arxiv.org/html/2608.14560 | Section 2 documents profiling/iteration with human orchestration. No published speed factors or author-agency claims transferred to our study. |
| Bloop | https://proceedings.mlr.press/v235/hsieh24a.html | Gradient-surgery/EMA context; no guarantee transferred to the exact OL1 implementation. |
| PCGrad | https://arxiv.org/abs/2001.06782 | Conflicting-gradient projection context and NeurIPS 2020 metadata; no guarantee transferred to this OL1 implementation. |
| AdamW | https://arxiv.org/abs/1711.05101 | Decoupled weight decay and ICLR 2019 metadata; actual update order checked in repository code. |
| MiniPile | https://arxiv.org/abs/2304.08442 | Dataset identity and title; realized splits, token counts and validation scope verified from repository records. |

The [ICLR2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
require at most 9 main-text pages at submission; references and appendices are
excluded. The current style ZIP remains official and unmodified. The
[AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors) requires an AI-use
section and a corresponding submission-form disclosure. That section is outside
the page budget. A factual statement of recorded assistance has been prepared;
the authors must review its completeness and their submission-form answers.
No assertion of completed personal author review was invented. The incorrect
`/AuthorGuide` URL returned an error; `/AuthorGuidelines` is the verified page.

This is an editing/attribution audit, not a new systematic literature review.

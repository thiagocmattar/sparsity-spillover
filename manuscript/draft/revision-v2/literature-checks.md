# Primary-source checks (2026-09-09)

- Q-Sparse, https://arxiv.org/html/2407.10969v1, Sections 2.1--2.2, equations 2--3
  and 12--13: absolute-value top-k at projection inputs, post-selection L2
  rescaling, and STE. Our retained values and rejected-entry gradients differ.
- Spark, https://arxiv.org/html/2506.06644v2, Sections 2.2 and 3, equations 9--11:
  attention-position selection; a per-input sample-mean/std Gaussian threshold
  followed by soft thresholding for FFNs. Its cost comparison is against naive
  sorting in its implementation context, not a theorem that all top-k sorts.
  Fixed cutoffs avoid estimation; no comparative runtime claim is made.
- Pythia, https://proceedings.mlr.press/v202/biderman23a/biderman23a.pdf, Table 6:
  gradient clipping 1.0 supports the norm-limiting lineage. The OL1 update-norm
  cap acts after preconditioning and is not identical to task-gradient clipping.

Existing bibliography keys are retained. These citations motivate choices;
our CPU proofs and retained observations supply our result claims.

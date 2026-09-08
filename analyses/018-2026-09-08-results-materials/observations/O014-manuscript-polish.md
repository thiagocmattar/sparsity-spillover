# O014 — Reviewed manuscript and standardized terminology

## Question

Can the complete draft communicate the evidence clearly to technical readers
and reviewers while preserving the user's first two introduction paragraphs
and the existing chain of arguments?

## Method and coverage

Reviewed every included section, ten figures, ten tables and all reported
results with independent technical-reader, scientific-reviewer and literature
agents. Revised the paper paragraph by paragraph and subsection by subsection,
then performed follow-up scientific and visual reviews. The original five-part
result sequence and all figure artwork are preserved. The all-variant overview
remains an alternative rather than replacing the selected original overview.

Added a concise abstract, compact recipe key, discussion, and a sourced
reproducibility protocol. Standardized threshold/nonlinearity terminology,
activation sparsity, model-wide sparsity, sparsity ceiling, and common-A7
`U_arch`. The ceiling still counts guaranteed zero-operand products at 100%
activation sparsity of the selected sites; its scientific definition is unchanged.
Following user feedback, seed details appear in the appendix protocol and the
main argument emphasizes the structured evidence across interventions,
parameters, distributions, operation counts, model sizes and runtime ablations.

## Captions and figure goals

The current captions are retained in the new snapshot's `training-results.tex`,
`kernel-autoresearch.tex`, `experimental-appendix.tex` and `results-appendix.tex`.
They supersede the earlier planned wording for this draft. Their main goals are:

1. Overview: expose different quality costs and computational reach.
2. Paired effects: show when pressure adds to a thresholded intervention.
3. Distributions: distinguish activation reshaping and omitted sparsity at zero.
4. Scale comparison: show the specific recipe ordering that persists.
5. Kernels: distinguish measured acceleration from product counts, including
   fusion and sparse-path ablations.

The appendix retains all trained endpoints, comparisons, density mass and
tails, operation decomposition, complete clipping paths and runtime controls.

## Result and verification

- The first two introduction paragraphs and their cited bibliography entries
  are unchanged; paragraph hashes and comparisons are retained.
- All 15 references were checked against primary records. Five same-year
  entries now use verified conference metadata; the protected paragraphs keep
  their existing preprint citations.
- Source audits reconcile 54 endpoints, 29 paired contrasts, 15 scale pairs,
  all 540 clipping evaluations and 14 pooled histogram groups. Numerical table
  rows are unchanged. Only terminology in two generated tables changed.
- Eleven figure copies and fourteen measurement copies match their originals.
  The new protocol's thirty source hashes resolve and match.
- The 26-page PDF has ten figures and ten tables, 53 resolved labels, no
  LaTeX/box warnings, and only embedded non-Type-3 fonts. All pages were visually
  inspected. Main PDF SHA-256:
  `c6b07a23495ca90fe237a5332d6d20cee52be9b8b17b704c55b826455f3b0d5d`.
- Final independent scientific review reports no remaining required scientific
  correction. The stronger framing stays within the measured scope.

## Caveats

This is an editorial and retained-evidence revision, not new experimental
evidence or a finding promotion. Treatment sweeps are not independent training
seeds. Density comparisons do not isolate a causal pressure route. Larger
models test complete recipes, and runtime evidence concerns the declared 14M
workload and device. Exact protocol limits remain available without dominating
the abstract or captions. Template fit, author metadata and submission itself
were outside the request. The review does not predict conference acceptance.

## Sources and reproducibility

- Live draft: `manuscript/draft/` (intentionally Git-ignored).
- [New source/PDF snapshot](../provenance/manuscript-20260908-polished/README.md),
  including review reports, final audit and protected-paragraph hashes.
- [O012 original results insertion](O012-manuscript-results.md) and O001–O004,
  O007 and O011; Runs 029, 030 and 031 own the underlying evidence.
- `03_manuscript_tables.py`: only three terminology replacements; all retained
  numerical checks pass. Plotting and scientific computation are unchanged.
- `provenance/manuscript-20260908-polished/SNAPSHOT.json`: file hashes and
  source mappings. Raw measurement copies remain local and reproducible from
  the retained supplementary source manifest.

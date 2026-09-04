# Verification of the manuscript evidence rewrite

Completed on 2026-09-04 for the user-authorized full draft rewrite.

## Evidence checks

- The reduction builds 59 main/appendix trained-checkpoint rows, 60 uniform
  clipping rows, six runtime sentinels, and a separate three-arm LR screen.
- Ten focused tests pass. They cover count aggregation, paired difference
  direction and units, Pareto directions/ties, pressure-scope identities,
  architecture decomposition, full clipping grids, all 20 common-loss
  selections, 410M reversal, training-only LR selection, negative runtime
  evidence, high-dose branch suppression, and generated assets.
- Every retained counting endpoint covers 338 sequences / 692,224 input
  tokens, with a 1,444-token excluded tail. Raw counts are pooled before
  division. Counting-pass and terminal losses are retained separately.
- Upstream identities are checked against recorded hashes; the selected raw
  logical and activation artifacts receive explicit SHA-256 entries.
- All five PDFs and twelve generated table fragments match the draft's
  build-input-hashes.json and the owning analysis files (17 matches).

## Manuscript checks

- main.tex compiles successfully with latexmk and the local official
  anonymous ICLR 2027 style. The final PDF is 17 US-Letter pages.
- Main text and disclosures end on page 8, within the nine-page ceiling.
  References occupy the end of page 8 and page 9; appendix pages are 10-17.
- All pages were rendered with pdftoppm and visually inspected. A final pass
  checked pages affected by the last prose/layout edits. All five figures
  were inspected at their embedded publication size. The frontier legend
  was moved outside the axes to keep every clipping point visible.
- There are no unresolved references/citations, missing assets, overfull
  boxes, or LaTeX warnings. One underfull vertical-box diagnostic around a
  float page and one underfull bibliography line remain; their rendered
  pages have readable, unclipped content.
- Corrected/checked current primary metadata for Q-Sparse, Spark Transformer,
  and Sparsing Law. The closest-prior-work discussion explicitly limits
  novelty and describes the unexecuted optimized baselines.
- No credentials, datasets, weights, checkpoints, transfer archives, active
  run files, or raster QA outputs are included in the scoped commit.

Final local manuscript PDF SHA-256:

`d5ceeadc0a24df9bb5f6a99ba038a6044472a0622e26269aecb76189b40b1f3d`

The manuscript remains in the repository's existing ignored draft directory.
Its prior source/PDF snapshot is retained under
manuscript/draft/revisions/before-evidence-rewrite/. The numbered analysis and
manuscript evidence crosswalk are the versioned part of this change.

## Scope of verification

This is artifact reanalysis and document verification. No new model training,
checkpoint evaluation, cloud launch, or inference benchmark was executed.
The complete bootstrap suite is a pre-launch requirement and was not needed
for this manuscript-only workflow. The work leaves research findings and
immutable source runs unchanged.

Independent training replication, fixed-pressure A7 controls, stronger matched
baselines, and downstream evaluation remain scientific work, as stated in
the manuscript and draft submission-readiness note.

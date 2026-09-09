# Rewrite verification and submission readiness

This is an evidence-grounded manuscript for author review. The rewrite improves
claim selection and presentation; acceptance remains dependent on the evidence
and reviewers' assessment.

## Build verification

- Official anonymous ICLR 2027 style, unchanged margins and typography.
- Main text, references, and appendix compile into main.pdf.
- Analysis 013's focused tests check pooled counts, pairing and units, full
  clipping grids, operating-point selections, architecture decomposition,
  410M direction, training-only LR selection, runtime limits, and output assets.
- The build regenerates five figure PDFs and twelve table fragments, and
  records their SHA-256 hashes before LaTeX compilation.
- Final layout, page counts, warnings, and test results are recorded in
  Analysis 013/VERIFICATION.md after rendering the completed document.

The official [ICLR 2027 author guide](https://iclr.cc/Conferences/2027/AuthorGuidelines)
allows nine main-text pages, with references and appendices outside that count.

## Scientific issues that rewriting leaves open

1. Independent paired training seeds.
2. A7 gates with the pressure set fixed to the four A4 sites.
3. Matched Q-Sparse-style training and optimized TEAL allocation.
4. Representation development, rescaling, mask structure, and a
   capacity-matched comparison for nearly suppressed branches.
5. Untouched-corpus/downstream validation and adaptation robustness.

The main text states the relevant limits. These are candidate follow-ups,
with separate design and launch decisions; none is represented as completed.

## Human closeout

Confirm the contribution and proposed claims, inspect the abstract and main
figures, and decide whether the present evidence warrants submission.
Verify the author-approved AI-use disclosure and final bibliography. Prepare
an anonymous, independently reproducible code/artifact release and validate
its contents from a clean environment. Confirm conference policies and forms
again at submission time.

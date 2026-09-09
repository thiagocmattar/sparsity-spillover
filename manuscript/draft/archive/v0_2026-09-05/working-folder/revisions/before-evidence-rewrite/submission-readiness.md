# ICLR 2027 v0 readiness

Checked on 2026-09-04 against the official ICLR 2027 author template and author
guidelines.

## Build status

- Anonymous official-style PDF builds successfully with `latexmk`.
- 13 pages total on US Letter: 7 pages of main text plus reproducibility and AI
  disclosures, 1 page of references, and 5 pages of appendix.
- The 9-page main-text ceiling is therefore respected with two pages of
  headroom. References and appendices are outside that ceiling under the 2027
  guidance.
- All citations and cross-references resolve. The final log has no overfull
  boxes or LaTeX warnings.
- Every PDF page and every source figure has been rendered and visually
  inspected.
- `build-input-hashes.json` pins the copied figure and generated-table inputs.

Official sources:

- Author guide: <https://iclr.cc/Conferences/2027/AuthorGuidelines>
- AI policy: <https://iclr.cc/Conferences/2027/AIPolicyForAuthors>
- Style archive retained locally as `iclr-2027-style-files.zip`.

## Scientific status

The draft is submission-shaped, not submission-ready evidence. It deliberately
states the following limitations in the abstract or main text:

- one seed, one corpus, one matched-token pass per scale;
- A4-OL1 versus A7-OL1 is a complete-recipe comparison;
- the 410M result is a fixed-token boundary with unresolved horizon, not a
  scaling law or proof of undertraining;
- OL1 has a geometric guarantee but mixed endpoint evidence;
- `R_model` is logical exact-zero opportunity for full-sequence uncached
  attention, not removed FLOPs or measured speedup.

The strongest reviewer-facing novelty claim is operand-level, causal-graph
accounting beyond TEAL's per-matrix footprint weighting. The draft does not
claim to introduce the first model-wide sparsity metric.

## Required human closeout before submission

1. Replace the anonymous author block only in the camera-ready or as permitted
   by the submission system; preserve double-blind supplementary files.
2. Decide whether one-seed evidence is sufficient for submission or approve a
   separately designed confirmation study. Do not present a future run as
   already supported.
3. Verify every author agrees with the AI-use disclosure and that it matches
   the final tool use.
4. Replace the promised code release with an anonymous artifact link and test
   it from a clean machine.
5. Audit bibliography metadata and contemporary related work immediately before
   submission.
6. Confirm the final ICLR dates, forms, subject areas, and any policy updates on
   the official conference site.


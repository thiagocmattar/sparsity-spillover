# Manuscript review and revision, 8 September 2026

User request: preserve the first two introduction paragraphs exactly; use their
direct style; keep the current argument; review every figure, table and result;
revise incrementally with technical-reader and reviewer feedback. Template work
and new experiments are outside this revision.

## Argument to preserve

1. Placement and combinations matter beyond a local sparsity percentage.
2. Pressure, gates and their sites specify the intervention; product accounting
   connects its zeros to model-wide work.
3. The overview motivates paired comparisons, which show conditional effects.
4. Distributions and operation counts explain why FFN sparsity alone misleads.
5. The high-threshold recipe ordering persists across three model sizes.
6. Hardware measurements and ablations establish where zeros repay their cost.

## Incremental passes

- Preserve and hash the user's opening; back up the current sources and PDF.
- Introduction, one paragraph at a time: remove repeated motivation and retain
  the question, accounting rationale, comparison design and result preview.
- Related work, then methods: verify primary citations and definitions; keep
  enough detail to understand the comparisons without appendix navigation.
- Setup, then each result subsection: lead with one insight, check each number,
  explain the comparison, state its material limit once.
- Review each caption and table against the actual exhibit and source data.
  Keep approved figure artwork; separate display conventions from shared protocol.
- Appendix: preserve complete results and reproducibility; clarify any missing
  implementation or data details and remove archival workflow language.
- Write a concise abstract and conclusion from the verified final claims.
- Independent rereview, build, visual inspection of every page, citation/label
  checks, numerical reconciliation, protected-paragraph and figure-hash checks.
- Record decisions and remaining scientific scope, save a new analysis-owned
  source/PDF snapshot, review the scoped diff, and commit without pushing.

## Completion evidence

The final audit must cover every input TeX file, cited reference, main and
appendix figure/table, and claimed result. A successful build alone is insufficient.
Scientific scope remains one training seed per size, matched fixed budgets,
complete MiniPile validation, logical products distinct from measured speedup,
and one adaptive 14M runtime case study. No editorial revision can establish
unmeasured seed robustness, causal spillover, or runtime transfer.

Progress and review resolutions are recorded in `revision-log.md`.

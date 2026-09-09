# Readiness assessment: existing-results rewrite

The reading draft is clearer and better supported, but is **not ready for
submission**. The current 30-page PDF has main text through page 14; the official
ICLR 2027 submission limit is nine main-text pages. The author must decide the
framing and figure budget and review the completeness of the AI use statement.
The unchanged definition now uses the approved architectural reach / R_arch
terminology. No new experiment is approved or implied by this assessment.

The scores below are planning judgments under task.md Section 1.3, not acceptance
probabilities. They are separate assessments and must not be averaged to obscure
a blocking issue.

| Dimension | Score (0?3) | Assessment and remaining boundary |
|---|---:|---|
| Contribution clarity | 2 | Two specific findings connect paired intervention effects to the local-zero/operation/runtime distinction. Author approval of this empirical framing remains open. |
| Comparison validity | 2 | Fixed-recipe pressure contrasts, unpressured placement changes and complete-recipe comparisons are distinguished. Expansion of the pressure objective also changes old coefficients; the missing fixed-weight crossed control remains unresolved. |
| Repeatability | 1 | Each training condition has one seed. Shared initialization supports pairing, not independent replication. Three timing processes do not establish training variability; tiny loss changes are not equivalence. |
| Quality and baseline fairness | 2 | High sparsity is paired with its A0 quality cost; retrospective loss budgets disclose clipped winners and absent latency measurements. Uniform four-site clipping is not optimized TEAL or a matched final-transformation adaptation control. A practical-efficiency headline would require stronger evidence. |
| Runtime attribution | 2 | Native, fusion-only and sparse-path comparisons use verified records; absolute latency and all sensitivity subsets are retained. Attention-dense is faster on every tested checkpoint. One 14M workload and a canonical-FP16/BF16-timing mismatch limit inference. |
| Scope and transfer | 2 | Larger-size results concern selected complete recipes, common token budgets and a lower 410M learning rate. No cross-size kernel speedup, scaling law or cached-decoding result is inferred. |
| Insight and readability | 2 | Operation sums explain the ranking reversal; zero masses are visible and runtime attribution replaces search chronology in the main text. All thirty pages were inspected. Main-text length still blocks venue compliance. |

## Verified current-results work

All 54 endpoints, 540 clipping evaluations, 29 paired deltas and 15 scale pairs
remain available. Integer operation sums and the common-A7 block-only identity
were checked. Retained timing samples support the 30-checkpoint attribution
analysis, while five historical h-only controls remain a separate cohort.
Full-range clipping, qualification failures and unfavorable outcomes remain
visible. Seven signed histograms and separate BF16 h/z row statistics retain
their actual precision and coverage.

The draft builds in a separate directory without prior LaTeX intermediates.
All thirty pages match the working PDF in extracted text and rendered pixels.
Twelve tables and five data PDFs regenerate byte-for-byte; two recipe PDFs
match in text and pixels despite LaTeX metadata differences. This used the
installed environment, not a new environment or a training reproduction.
All 61 focused scientific/evidence tests pass. Exact hashes and limitations
are recorded in verification.json and numerical_checks.json.

## Remaining decisions and recommendation

1. Confirm the empirical-design framing and keep the bounded uniform-clipping
   result as supporting evidence, or request a different headline. Broad
   post-hoc superiority is not supported by the present comparison.
2. Choose between retaining this fuller reading draft and condensing the main
   text to nine pages. The current six main figures are retained pending that
   choice. The separate optional ladder-to-matrix rename was not approved;
   the figure explicitly states that its rows are separately pretrained recipes.
3. Before submission, the authors must verify the AI use statement's completeness
   and their submission-form disclosure, and arrange any intended anonymous
   artifact access. No public anonymous release or complete checkpoint bundle
   is claimed by the current supplement.

For a stronger empirical case, the first additional investment should be the
fixed-coefficient controls (M1) and independent seed pairs (M2), with outcomes
allowed to narrow or reverse the claim. M3 becomes necessary for a stronger
post-hoc superiority claim. Longer training or larger kernels should follow a
specific unresolved question rather than replace these controls. Full experiment
design, cost and launch approval remain separate; all M/H tasks are deferred.
The proposed outcomes and wording consequences are in experiment_decisions.md.

The rewrite cannot remove the single-seed, objective-identification or comparator
limitations. Existing-results completion supports a bounded empirical reading
copy, not a universal sparsification method or a proven quality?latency advantage.

Venue facts were checked on 9 September 2026 against the
[official author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
and [AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).

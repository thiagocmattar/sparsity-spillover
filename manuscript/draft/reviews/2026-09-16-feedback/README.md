# Review corrections, 16 September 2026

The author requested application of `../../feedback-review-task.md` to the
current draft. Its SHA-256 is recorded in [verification.json](verification.json).
The input review and the author's separate `notes.txt` were not edited or
staged. This revision changes the manuscript and retained-evidence figures;
it launches no experiment and changes no training or kernel implementation.

The paper now asks why activation sparsity, zero-product opportunity and
useful acceleration can disagree. The title is **When Sparsity Becomes Useful
Computation: Pressure, Thresholding, and Site Placement**. Main text occupies
pages 1–9; AI-use and reproducibility statements start on page 10. The complete
PDF has 32 pages, including references and appendices.

## Requirement-by-requirement audit

| Review request | Correction and authoritative evidence |
|---|---|
| Central framing: controlled disagreements rather than uniformly better sparse models | Abstract, Introduction and Discussion organize the contributions around conditional pressure effects, site-specific versus model-wide zeros, and absolute execution cost. Figures 5 and 7 directly show the differing rankings. |
| Nine-page initial main text; separate AI-use statement | The rendered conclusion ends on page 9. `ai-use.tex` starts on page 10 and discloses discussion, code, analysis, figures and writing assistance. Official style files are unchanged; no margin or font adjustments were used to meet the limit. |
| §2.1: absolute latency versus validation loss as principal runtime comparison | Figure 7a plots all 30 checkpoints using projection-only K050 absolute latency and canonical validation loss. Native dense execution is identified separately. Neither interpolation to matched quality nor a causal sparsity effect is claimed. |
| Explain the seven-site native-ratio reversal | §4.6 reports projection-only and full latencies, both native denominators, the 1.609/1.783 ratios, and the fact that four-site is faster while seven-site has lower loss in the matched kappa = 0.5 pair. Appendix Figure 18 and Table 11 retain all matched evidence. |
| Distinguish native implementation improvement, projection skipping and attention skipping | §4.6 defines tN, t0, tP, tK and gives the three-factor decomposition 1.783 ≈ 1.360 × 1.325 × 0.989. Dense kernels/layouts are included in the first factor; it is not labeled a pure fusion effect. |
| Recommend attention-dense/projection-sparse execution | §4.6 recommends projection-only K050 for the measured cohort, checked against all 30 retained latency pairs. Full K050 remains an informative negative attention-skipping ablation. |
| Promote instruction-level evidence | Figure 7b uses actual projection MMA bypass versus matched t0/tP. Text and Appendix Figure 17 report R² = 0.946 versus scalar projection R² = 0.413. Padding, scalar substitution, counter coverage, and descriptive interpretation are explicit. |
| Preserve native-relative results without using them to rank recipes | Appendix Figures 15–16 identify each checkpoint's own native denominator. §4.6 and Appendix D.5 give the common dense-reference ratios 1.427/1.385 and their cross-model timing limitations. |
| §2.2: put quality cost beside sparsity maxima | Abstract and §4.5 pair 27.48/40.60/80.62% sparsity with dense-relative loss penalties 0.621/1.116/0.573. §4.5 additionally reports precise differences and perplexity factors 1.86/3.05/1.77. The 70M difference rounds to 1.1162 from full-precision data, rather than 1.1161 from the review's rounded endpoints. |
| Distinguish marginal pressure cost from total dense-relative cost | Introduction, Figure 3 caption and §4.5 distinguish seven-site pressure's +0.1265 loss from the complete 14M recipe's +0.6208. |
| Narrow cross-size claim; no convergence mechanism | §4.5 is “Cross-size recipe evaluation.” It reports the kappa = 0.5 ordering, the changed kappa = 0 ordering, missing larger-model pressure-free controls, fixed tokens, different tokens/parameter and lower 410M learning rate. Appendix Figure 8 explicitly says global norms are not normalized by parameter count and does not use them as a convergence criterion. |
| Analytic architecture-and-workload dependence | Appendix C.2, Equation 5 derives seven-site reach as L(12d+T+1)/[L(12d+T+1)+V]. It states the full-vocabulary, every-position assumption and the changed denominator for final-position logits. A focused test independently checks the simplification for all three architectures. |
| §2.3: conditional pressure benefit and correct four-site increment | §4.2 and Figure 3 use +2.498 pp, displayed +2.50, and report the accompanying +0.3783 loss. The seven-site contrast is +12.096 pp/+0.1265. |
| Complete-recipe interaction, not isolated Q/K/V causality | §4.2 defines the difference of matched pressure effects and reports +9.598 pp and −0.2518 loss. Pressure targets and equal-tensor normalization change together. |
| Do not infer a transition from the last threshold | §4.2 and Discussion state that no sampled thresholds lie between 0.1 and 0.5; the result is described at the largest tested threshold. |
| §2.4: OL1 geometry is not “resistance” or loss sensitivity | §4.2 defines rho_opp as a task-relative opposing component, combining magnitude and direction. It explicitly distinguishes this from loss sensitivity and the fraction of pressure removed. Appendix A.2 retains the projection guard and pre-learning-rate/weight-decay coordinates. |
| Separate FFN input m from hidden h | Figure 5a and §4.4 use retained per-site counts: m changes 96.77→68.70%, h changes 99.94→99.88%. Counts are directly measured, not inferred from rounded operation contributions. The 80:20 pooling issue is stated. |
| Add dense reference, pressure-free bars and layer detail | Figure 5a includes the dense distribution reference; Figure 5b includes all four A4/A7 pressure-off/on endpoints at kappa = 0.5. Appendix Figure 10 shows five sites across six layers. Pressure-free density histograms are explicitly unavailable; no curves were manufactured. |
| Preserve nonlocal-response caution | §4.4 notes Q/K/V changes without direct four-site intervention, while leaving pressure, thresholds, normalization, cap behavior and representation adaptation as unresolved explanations. Cap activity is not asserted to cause the FFN-up change. |
| §2.5: one/four/seven site types per block | §3.1 defines {h}, {a,m,h,z}, {a,m,h,z,q,k,v} as repeated site types, not layer counts. Table 1 maps dense A0 and one-site ReLU A1-H to historical IDs. |
| Post-hoc thresholding and execution terminology | Main prose and new figures use dense reference, post-hoc magnitude thresholding/compact “post-hoc thresholds,” cross-size evaluation and batch-size-one uncached inference. Historical appendix plots retain their original labels; D.3 explicitly defines legacy “clipping” as zeroing small magnitudes, not amplitude clipping. Actual gradient clipping is still correctly called clipping. |
| Reach reference is not necessarily bounded utilization | §3.2 and Appendix B.2 qualify selected-site reach, natural zeros outside it, and the potentially greater-than-one ratio. Figures 1 and 6 use reach terminology. Historical U_arch remains only as an explicitly qualified appendix-table reference ratio. |
| Marker semantics must not erase naive L1 | New kernel Figure 7 uses open/filled only for no-pressure/OL1 four-/seven-site families. The gray group is explicitly dense, ReLU, naive L1 and OL1, with no fill-based pressure interpretation. |
| Robustness and missing controls | Setup, Introduction/Discussion and Appendix C.4 identify 54 configurations as single-seed observations using one validation split. Larger-model pressure-free controls, independent seeds and independent confirmation data remain absent and are not implied by the revision. |
| Reproducibility must not overclaim executable release | The reproducibility statement and Appendix D.6 say that supplementary measurements permit reanalysis but are not a standalone executable checkpoint/kernel release. Existing detailed protocol, source hashes, qualification and timing evidence are preserved. |

The rubric's requests for new seeds, an untouched confirmation set, stronger
compiled baselines and an executable anonymous artifact describe future
evidence needed for a stronger paper. They are not experiments performed by
this editing task. Their absence is now explicit rather than concealed by
wording. Reviewer scores, deadlines and predicted acceptance are not treated
as scientific results or inserted into the paper.

## Evidence and verification

The new derivations and five review PDFs come from
[`10_review_figures.py`](../../../../analyses/021-2026-09-10-training-results-figures/10_review_figures.py),
with their question, coverage, captions, results and caveats in
[O015](../../../../analyses/021-2026-09-10-training-results-figures/observations/O015-review-corrections.md).
The three previously generated kernel appendix PDFs are copied unchanged.
Original analysis figures, run artifacts, raw measurements and trained weights
are unchanged. Figure-copy hashes are in `../../figures/SOURCES.json`.

The focused checks cover integer mass conservation and site/layer aggregation,
pressure-free control identities, full-model operation denominators,
interaction arithmetic, quality costs and perplexity ratios, analytic reach,
all 30 plotted runtime coordinates, marker semantics, source provenance,
matched timing estimands and stratified associations. The test manifest,
build checks, page count, template hashes and PDF/font checks are recorded in
[verification.json](verification.json).

The normal build from `manuscript/draft` is:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The final build has no undefined citations/references or overfull boxes.
Two underfull horizontal boxes and one underfull vertical box remain; visual
inspection found no clipping or overlap. All new figures and affected pages
were rendered with Poppler. The official template, margins and font settings
are preserved.

The current initial-submission limit and separate AI-use disclosure were
checked against the official
[ICLR 2027 author guidelines](https://www.iclr.cc/Conferences/2027/AuthorGuidelines)
and [AI policy](https://www.iclr.cc/Conferences/2027/AIPolicyForAuthors).
The disclosure describes documented assistance and author responsibility;
it does not certify an unperformed final human review or submit the paper.

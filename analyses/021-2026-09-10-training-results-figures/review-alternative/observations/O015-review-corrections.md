# O015 — Archived review figure alternatives

Status: the manuscript revision in `f9351d7` was reverted at the author's
request. These figures and their evidence are retained separately; this
observation does not describe the current manuscript.

## Question and scope

Do activation zeros, counted zero products, and useful acceleration rank the
same interventions? The author requested application of
`manuscript/draft/feedback-review-task.md` to the current draft. This revision
uses retained measurements only: no training, checkpoint evaluation, timing,
or cloud execution was launched. Earlier analysis PDFs remain unchanged.

## Method and coverage

[10_review_figures.py](../10_review_figures.py) joins the canonical Analysis 018
endpoint counts, Run 031 signed histograms, and the existing 30-checkpoint K050
investigation. [review-corrections.json](../data/review-corrections.json) records
source SHA-256s, integer histogram counts and tails, per-layer zeros, operation
contributions, runtime records, and calculated contrasts.

- Histograms: dense reference and four-/seven-site OL1 at kappa = 0.5; six
  layers and sites m, h, q, k, v. Each evaluation covers all 338 complete
  validation blocks from 500 documents, excluding the 1,444-token tail.
  Q/K are post-RoPE. Counters come from FP16-autocast evaluation; pooling is
  by integer count, not mean percentage. The 16,000-bin [-8,8] histogram is
  rebinned by ten, preserving nonzero, exact-zero, underflow and overflow mass.
- Operation bars: four-site and seven-site at kappa = 0.5, both with and
  without OL1, using the common full-model product denominator. Pressure-free
  density histograms are unavailable in the retained set; the bars are not
  reconstructed from the pressure-on densities.
- Runtime: the unchanged 30 BF16 checkpoints, RTX5090, batch size one,
  uncached 2,048-token inference and full-vocabulary logits. Absolute latencies
  use 64 fixed inputs, seven passes and three processes. Kernel counters cover
  338 blocks; these coverage and precision differences remain explicit.
- Cross-size quality costs use canonical natural-log cross-entropy endpoints.
  They are calculated differences and exponentiated differences, not new runs.

## Figures, legends and captions

All five PDFs are preserved byte-for-byte from `f9351d7` in
[figures](../figures). Their draft copies were removed by the rollback.

1. [Quality–sparsity overview](../figures/01-quality-sparsity.pdf),
   overview: unchanged 26-checkpoint view from O001, with dense,
   post-hoc thresholding and selected-site reach terminology. Blue/orange
   identify four/seven sites; open/solid versus filled/dashed denote no
   pressure versus OL1 for those families. Dotted paths are post-hoc magnitude
   thresholding. The +0.13 loss callout is marginal pressure cost, not the
   complete recipe's +0.6208 dense-relative penalty.
2. [Site distributions and operation accounting](../figures/04-site-distributions.pdf),
   composite: **Site-specific zeros and operation-weighted consequences.**
   Panel (a) separately displays m, h and pooled Q/K/V for the dense reference
   (gray), four-site OL1 (blue) and seven-site OL1 (orange). Curves divide bin
   counts by all captured elements and bin width, so their integrals exclude
   zero and tail mass. The y-axis is symlog, linear below 0.01; FFN and attention
   display limits are [-2.5,3] and [-4,4]. Exact-zero percentages appear below.
   Panel (b) includes the two pressure-free controls and two OL1 checkpoints;
   six operation colors partition S_model, including the dense vocabulary head
   in the denominator only. Stacks are logical opportunities, not runtime gains.
3. [Cross-size recipes](../figures/05-cross-size.pdf):
   unchanged absolute-loss coordinates from O006, with reach-reference wording.
   The common quality axis and size-specific sparsity axes show independently
   trained four-/seven-site OL1 recipes and dense post-hoc thresholds. The
   kappa = 0.5 ordering recurs; the complete threshold response does not.
4. [Absolute latency and projection bypass](../figures/07-kernel-quality-latency.pdf),
   runtime: **Quality, absolute latency and exploitable zero structure.**
   Panel (a) plots all 30 canonical losses against projection-only K050 latency;
   a black cross shows native dense-reference execution. Blue diamonds and
   orange triangles identify four-/seven-site recipes, open without pressure
   and filled with OL1. Gray circles include dense, ReLU, naive L1 and OL1;
   gray fill does not encode pressure. Panel (b) plots actual projection MMA
   bypass against t0/tP with a descriptive OLS fit. Bypass includes padded
   instructions and replacement by scalar work, not solely eliminated arithmetic.
5. [Site-by-layer exact zeros](../figures/appendix-site-zero-heatmap.pdf),
   heatmap: five sites by six layers for the same three density checkpoints.
   The common color scale is 0–100%; whole-number labels are rounded and a
   displayed 100 does not establish an all-zero tensor.

The existing O011–O013 kernel appendix PDFs remain in the parent analysis
folder, in within-family, scalar-versus-instruction, and absolute-latency order.
Their manuscript adoption in the review revision was also reverted.
Their native-relative associations retain each checkpoint's own reference;
they do not rank absolute optimized recipe performance.

## Results

| Retained measurement | Four-site OL1 | Seven-site OL1 |
|---|---:|---:|
| FFN-up input m exact zeros | 96.76575% | 68.69617% |
| FFN hidden h exact zeros | 99.94038% | 99.87643% |
| Pooled Q/K/V exact zeros | 0.22032% | 95.59889% |
| S_model | 12.71345% | 27.48268% |
| Validation loss | 6.03798 | 5.82941 |
| Projection-only K050 latency | 0.451750 ms | 0.468365 ms |
| Full K050 latency | 0.459476 ms | 0.473366 ms |
| Native latency | 0.739444 ms | 0.844094 ms |

All rows compare kappa = 0.5. The h:m pooling weight of 80:20 obscures that
the FFN change concentrates at m. The highest-sparsity checkpoint is not the
lowest-latency checkpoint. Seven-site's larger native-relative ratio includes
its slower native denominator; it does not indicate faster optimized execution.

Projection-only execution is faster than full K050 for every cohort checkpoint.
Projection MMA bypass has descriptive R² = 0.946 against projection gain,
versus 0.413 for scalar projection zeros. For seven-site OL1 at kappa = 0.5,
native/full = (native/all-off) × (all-off/projection-only) × (projection-only/full)
is approximately 1.783 = 1.360 × 1.325 × 0.989. The first factor includes dense
kernels and layout changes; it is not a pure fusion effect.

The complete-recipe pressure interaction at kappa = 0.5 is +9.59796 pp and
−0.251793 loss. Seven-site OL1's dense-relative losses at 14M/70M/410M are
+0.6208336/+1.1161587/+0.5732352, with perplexity factors 1.86048/3.05310/1.77400.
The review's 70M +1.1161 used rounded endpoints; full-precision subtraction
rounds to +1.1162. These results do not show improving quality cost with size.

## Interpretation limits

There is one training seed per configuration and shared validation throughout.
Pressure targets and equal-tensor normalization change together. No untreated
confirmation set, additional seeds, intermediate thresholds from 0.1 to 0.5,
or larger-model pressure-free controls are introduced. Norm-budget behavior
is a possible explanation, not identified as the cause of the m change.
Runtime claims concern the measured shapes and hardware; stronger compiled
dense baselines and cached decoding remain untested. Supplementary measurements
support reanalysis but do not constitute a standalone executable kernel release.

## Reproduction and verification

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/review-alternative/10_review_figures.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/review-alternative/test_review_figures.py -q -p no:cacheprovider
```

The three new tests check site/layer count conservation, density mass, four
operation controls, interaction and quality-cost arithmetic, the analytic
reach simplification, all 30 plotted absolute latencies, correct marker fill,
the bypass correlation and the projection-only recommendation.

The archived PDFs were rendered and checked during the original revision.
The rollback preserves their bytes and adjusts only reproduction paths.
The original manuscript audit remains available in the history of `f9351d7`;
its manuscript adoption and page-count claims no longer apply.

# Compact detailed results and 14M post-hoc comparison

## Question and editorial scope

Which retained results add useful detail to the main paper's threshold/pressure,
quality/sparsity and execution comparisons without repeating its figures?
The author requested a much shorter results appendix, current T/P terminology,
compact tables, a restyled 14M-only post-hoc figure, and preservation of the
410M stress-test panel and its two discussion paragraphs. A follow-up requested
removal of large whitespace gaps, including the introduction/Section 2 transition.

## Selection and method

All 74 trained endpoints remain, in three disjoint tables:

- **14M-only ablations:** 18 endpoints. L1/OL1 at T1/P1 are paired by lambda;
  unpressured T4/P0 and T7/P0 are paired by kappa in a second block.
- **Matched 14M/70M recipes:** 44 endpoints in 22 rows, with loss and model-wide
  sparsity side by side. Multirow recipe labels and horizontal rules separate
  controls and the four multisite pressure recipes.
- **410M:** 12 endpoints, adjacent to the retained stress-test figure.

The redundant common-seven-site ceiling ratio is removed. All trained losses
now use the same ordinary final-checkpoint pass as the main figures. The old
tables mixed that convention with the logical diagnostic pass for 54 historical
rows. Twelve of those loss entries change at four decimals; the largest raw
difference is 0.000244677. No model, measurement, sparsity count or claim changes.
This convention audit belongs here rather than in the manuscript captions.

The old paired-effect tables can be reconstructed from the endpoints and repeat
the main paired-pressure figure. The appendix's heatmaps duplicate the main
site/layer figure. The density/tail archive and operation-accounting table/chart
are removed from the paper: they introduce a large historical cohort and
additional estimands without advancing the present pressure-placement argument;
the accounting appendix and current kernel diagnostics retain the definitions
and direct execution evidence. Their original files and raw evidence remain.
The complete 70M/410M all-checkpoint clipping displays are removed. The 410M
stress-test panel, including its Base/ReLU paths, is byte-identical to its source;
both requested discussion paragraphs are preserved verbatim.

## Figure, coverage and caption

[21-14m-posthoc-quality-sparsity.pdf](../figures/21-14m-posthoc-quality-sparsity.pdf)
retains all 300 existing 14M clipping settings for 30 checkpoints. Checkpoint
content hashes join each sweep to its trained recipe. No sweep is inferred for
the ten T4/Ph and T7/Ph models lacking these measurements. Both panels display
the same data: one resolves loss 5.0--6.25, while the other shows the complete
5.0--9.6 range. The latter omits no evaluated point. Recipe colors, trained line
styles, circular markers and T/P labels follow the main quality figure. The
ordinary L1 comparison is identified separately from OL1.

**Caption:** Post-hoc clipping on trained Pythia-14M models. Thirty trained
checkpoints and ten clipping targets per checkpoint (p = 0, 0.1, ..., 0.9).
Both panels show the same settings, with different loss ranges. Prominent
markers and recipe lines identify trained models; dotted paths with small
open markers follow each fixed model through increasing clipping targets.
Colors and recipe labels match the main figures, with the L1 comparison
included explicitly. Each path starts at its measured p = 0 point. Lines
connect evaluated settings, not fitted frontiers.

Coverage remains all 500 validation documents, 338 complete 2,048-token blocks
(692,224 input tokens), excluding the 1,444-token tail. Sparsity divides pooled
integer zero-product counts by the model denominator. Trained losses use the
ordinary final pass; post-hoc points retain their actual measured losses,
including p=0, and their original FP16 diagnostic convention.

The figure makes the high-loss ends of the clipping paths visible while
retaining a useful view near the trained models. It supplies quality/sparsity
detail, not new runtime evidence. One seed, the finite grid and unmeasured
h-only sweeps limit the comparison; no interpolated frontier is claimed.

## Sources and reproduction

- Builder: [33_compact_results_appendix.py](../33_compact_results_appendix.py).
- Export and hashes: [compact-results-appendix.json](../data/compact-results-appendix.json).
- Endpoint evidence: [all-model-quality-sparsity.json](../data/all-model-quality-sparsity.json),
  with original metrics and logical counts verified by the focused tests.
- Clipping evidence: [Run030 clipping-points.json](../../../runs/030-2026-09-08-all-models-posthoc-clipping/results/clipping-points.json).
- Generated tables: [compact-results](../tables/compact-results/).
- Manuscript: [results-appendix.tex](../../../manuscript/draft/results-appendix.tex).

Run `python analyses/024-2026-09-17-h-only-kernel-latency/33_compact_results_appendix.py`
from the repository root. The three TeX fragments and PDF are copied into the
manuscript; hashes verify each copy. No training, inference or cloud work ran.

## Layout and verification

The 20-page manuscript replaces the preceding 30-page build. Tables 6--7 share
page 16; the 14M post-hoc figure is Figure 9 on page 17; the 410M table and
unchanged artwork are Table 8/Figure 10 on page 18. The newer kernel appendix
continues immediately afterward, with its text and artwork unchanged.

Normal vertical spacing replaces flush-bottom stretching. Figure 1 uses 90%
line width so Related Work begins on page 2 after the introduction. The main
recipe table, quality figure and site/layer figure stay beside their text;
the 410M table/figure no longer force a floating-only page. Author prose and
all existing figure assets are preserved. Only these layout directives change
in the three already author-edited main-text files.

Eight focused tests pass: exact 74-endpoint coverage and matched settings,
serialized table values against original metrics/counts, all 300 clipping IDs,
checkpoint hashes, actual p=0 measurements, and source/manuscript-copy hashes.
The final LaTeX build has no undefined references/citations, overfull boxes or
underfull vertical boxes. Three pre-existing underfull horizontal boxes remain
(title and kernel prose/table). All pages were reviewed in contact sheets;
the changed figures, table page and introduction transition were inspected at
higher resolution. The installed PDF matches the reviewed build's SHA256.

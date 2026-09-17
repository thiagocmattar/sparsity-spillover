# 004 — Clean 14M overview with A4 and A7 h-only OL1

## Question and scope

How do A4 and A7 with pressure restricted to h sit in the existing 14M
quality–sparsity overview? The user requested this figure in Run 032 with
no annotations. The display preserves the overview curves and adds the two
h-only curves. It is a post-hoc visualization of retained results; no new
experiment, manuscript update, or finding promotion is made.

## Sources, method, and coverage

- The original 46 points are copied without coordinate changes from
  `analyses/021-2026-09-10-training-results-figures/data/14m-quality-sparsity.json`.
  This contains 26 trained endpoints and 20 post-hoc clipping points from
  Analysis 018 and Run 030. Its source hashes are checked on regeneration.
- A4 h-only uses the five `run012_h_only` records in
  `analyses/009-2026-08-31-run012-vs-run015-a4-ol1-pressure-sites/figure_data.json`.
  That completed audit establishes realized pressure at h despite Run 012's
  original all-four-site declaration. Source code, verification, manifest,
  activation, and logical-counter hashes are checked. The A4 all-four-site
  curve in the overview uses the separate corrected Run 015 results.
- A7 h-only uses Run 032's complete `artifacts/verification.json` and each
  verified attempt's `diagnostics/logical_products.json`. It checks five
  conditions, six h captures, gate sites, pressure weight/budget, completed
  steps/tokens, initialization, data order, and validation coverage.

The two h-only curves share initialization and training-order hashes, use
712 updates and 1,493,172,224 training input tokens per condition, and evaluate
the final checkpoint. Each has kappa = 0, 0.01, 0.05, 0.1, 0.5, lambda = 1,
b = 1, and one seed. Validation covers all 500 MiniPile validation documents,
338 complete 2,048-token blocks (692,224 input tokens), excluding the
1,444-token tail. There are no error bars because these are single-seed runs.

Every x coordinate is recomputed as 100 times the pooled integer zero-product
count divided by the model-product count. Every y coordinate is the ordinary
final-checkpoint validation loss, not the eager diagnostic-pass loss.
The complete 56-point snapshot and SHA-256 source inventory are in
`../data/14m-quality-sparsity-h-only.json`.

## Caption and legend

**Pythia-14M quality and model-wide logical sparsity, including h-only OL1.**
Blue curves use A4 gates at a,m,h,z; orange curves use A7, adding symmetric
q_post,k_post,v gates. Hollow diamonds/triangles with solid lines show no OL1;
filled diamonds/triangles with dashed lines show OL1 on all four/seven sites.
Blue squares and orange circles with dash-dot lines show OL1 only at h.
Points are connected in increasing kappa order. Gray/green marks show the
baseline, ReLU, ReLU with OL1, and their dotted post-hoc clipping sweeps as
identified by the legend. Lower validation loss is better; the x axis is a
logical-product opportunity, not measured runtime speedup. The original
axis limits are retained. Eight clipping points (thresholds 0.6–0.9 for each
of baseline and ReLU) are above the visible y range but remain in the data.
Point labels, callouts, and analytic ceiling guides are omitted for clarity.

## Observed pattern and limits

At kappa = 0.5, A4 h-only reaches 10.2274% model-wide logical sparsity with
loss 5.722666, while A7 h-only reaches 16.6636% with loss 5.732049. These
endpoints differ by 6.4362 percentage points and 0.009383 loss. At kappa = 0,
the h-only endpoints nearly coincide. The lines interpolate no measured
intermediate thresholds; they only connect the five sampled conditions.

This is one model scale, one seed, and a fixed training budget. The contrast
does not isolate a particular Q/K/V mechanism or establish runtime savings.
The Run 012 pressure-label correction is essential to identifying its curve;
the original declaration must not be read as realized all-site pressure.

## Reproduction and verification

Source script: `../24_plot_h_only_quality_sparsity.py`.
Output: `../figures/01-14m-quality-sparsity-h-only.pdf`.

The plotting script completed with its source-hash, pooled-count, coverage,
threshold, and matched-identity assertions passing. An independent figure
check confirmed exact preservation of the reference's 46 records, 56 plotted
points, zero in-plot text annotations, and label/legend bounds inside the
canvas. The single-page PDF was rendered at 1,800 pixels and visually checked
for legibility, clipping, and legend placement. Both font subsets are embedded.
Temporary PNGs are previews only; the retained publication figure is PDF.

## 17 September loss-provenance correction

The sentence above describing every y coordinate as ordinary final validation
is too broad. The four inherited A4/A7 overview curves use final-checkpoint
eager logical-pass loss, while the added h-only curves use ordinary final loss.
The maximum between-pass difference across the 30 A4/A7 endpoints is
0.000105806356 nats. The original figure and its data remain unchanged.
Analysis 023 archives their exact values, names the loss pass per endpoint,
and supplies both uniform-pass tables for manuscript preparation.

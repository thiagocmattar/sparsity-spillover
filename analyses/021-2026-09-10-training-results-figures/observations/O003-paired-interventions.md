# O003 - Matched effects of adding OL1

**Frozen by the author after the final decluttering pass on 11 September 2026.**

## Question and status

At a fixed threshold, what changes when OL1 pressure is added, and how does
that differ between the four-site and seven-site recipes?

The author requested this focused redesign on 11 September 2026, initially
restricted to Analysis 021, then approved adoption in the manuscript.
This is a replot of retained measurements, not a new experiment or promoted finding.

## Method, sources and coverage

- Source: [Analysis 018 figure_data.json](../../018-2026-09-08-results-materials/figure_data.json),
  using its trained endpoints and reconciling against its existing contrasts.
  Its SHA-256 is recorded in the [local reduction](../data/14m-paired-interventions.json).
- Twenty Pythia-14M trained endpoints form ten pairs: A4-OL1 minus A4 and
  A7-OL1 minus A7 at kappa = 0, 0.01, 0.05, 0.1 and 0.5. A4 uses Run 011;
  corrected A4-OL1 uses Run 015; A7 and A7-OL1 use Runs 013 and 014.
  Historical Run 012's h-only pressure is excluded.
- Within each pair, initial-parameter hash, data-order schedule hash, model/data
  seeds, training token budget, optimizer settings and validation cache match.
  Model/data-order seeds are 1234; training uses 712 steps and 1,493,172,224
  input tokens. OL1 uses lambda = b = 1. The reduction retains these identities.
- Both loss and sparsity come from the same eager full-validation pass:
  all 500 MiniPile validation documents, 338 complete 2,048-token sequences,
  692,224 input tokens and 691,886 next-token predictions. The 1,444-token
  tail is excluded. Terminal training-log loss is not substituted for this loss.
- Four-site pressure targets a,m,h,z; seven-site pressure adds post-RoPE q,k
  and v. Pressure is the equal-tensor mean absolute activation objective.
  Gate settings are held fixed within each pressure-on/off pair.

Each loss difference is pressure-on minus pressure-off. Each sparsity difference
is 100 times the difference in pooled zero-product numerators, divided by the
common model-product denominator (6,363,055,915,008). The numerator pools all
six block operation families; the denominator includes the dense final output
projection and excludes future-masked attention pairs. It is a percentage-point
change in model-wide sparsity, not a relative percentage change or runtime gain.

## Figure and caption

[Figure PDF](../figures/03-14m-paired-interventions.pdf).
Two stacked panels share categorical threshold positions. Blue diamonds
denote four-site recipes and orange triangles seven-site recipes, matching the
quality-sparsity overview. Thin lines connect the measured endpoints. The top
panel spans -0.05 to +0.42 loss; the bottom spans -1 to +13 pp, retaining all
ten contrasts. Only the two complete highest-threshold effects are annotated.
The frozen layout has a light grid, repeated threshold ticks, and one bottom
x-axis title. The definition "Delta = (+OL1) - (no pressure), at matched
kappa" appears as a small subtitle above the panels. The shared 4-site /
7-site legend sits below the bottom x-axis title. Each endpoint callout places
the loss value below the sparsity gain; the narrower callouts allow wider
plotting areas without changing the overall figure dimensions.
Panel titles name the effect of adding OL1 on validation loss and model-wide
sparsity. Y-axis labels are shortened to delta validation loss and delta
S_model (pp); lower-/higher-is-better labels are removed. The figure is
5.5 by 3.5 inches. The contrasts and source data are unchanged; the reference
remains no pressure at the same threshold.

**The marginal effect of adding OL1 depends strongly on threshold and pressure
target set.** Each point is a matched comparison at the same trained threshold
kappa: Delta = (+OL1) - (no pressure), using A4-OL1 minus A4 for the four-site
recipe and A7-OL1 minus A7 for the seven-site recipe. **(a)** Change in validation
loss; negative values indicate improved quality. **(b)** Change in model-wide
sparsity S_model, in percentage points. **At kappa = 0.5, four-site pressure
adds 2.50 pp sparsity for +0.38 loss, whereas seven-site pressure adds 12.10 pp
for only +0.13 loss.** Lines connect matched threshold settings and do not
represent training trajectories. All comparisons use matched initialization,
data order, and training budget.

## Results

Values below are rounded for reading; plotted coordinates and the local
reduction preserve full precision.

| Kappa | 4-site loss change | 4-site sparsity change (pp) | 7-site loss change | 7-site sparsity change (pp) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -0.012240 | +0.598024 | +0.011788 | -0.163422 |
| 0.01 | -0.008179 | +1.173005 | +0.016981 | +0.105744 |
| 0.05 | +0.055645 | +2.250552 | +0.024925 | +0.736108 |
| 0.1 | +0.128580 | +2.441276 | +0.000819 | +1.371742 |
| 0.5 | +0.378304 | +2.497912 | +0.126512 | +12.095871 |

At four sites, pressure improves loss and sparsity at kappa = 0 and 0.01.
Beyond those thresholds, sparsity gains approach 2.50 pp while loss cost rises.
At seven sites, loss changes remain close to zero through kappa = 0.1;
the sparsity gain jumps to 12.10 pp at kappa = 0.5, with +0.13 loss.
At that matched highest threshold, seven-site pressure adds substantially more
sparsity at lower loss cost than four-site pressure (+2.50 pp / +0.38 loss).
Seven-site pressure is not uniformly beneficial: at kappa = 0, both metrics worsen.

## Interpretation limits

- One seed; thresholds are interventions, not independent replicates. No
  confidence bands, significance or seed-robustness claims are supported.
- The four-site and seven-site families differ in gate placement, pressure
  targets, architectural reach and the objective's equal-tensor normalization.
  Differences between their pressure responses characterize the complete recipes;
  they do not isolate a causal contribution from q,k,v alone.
- This figure does not compare OL1 with naive L1, directly plot A7 minus A4,
  or identify a loss benefit of the projection or norm cap independently.
- Lines are ordered settings, not a fitted frontier, an interpolated model,
  or successive changes to the same trained checkpoint.
- Model-wide sparsity is a logical zero-operand opportunity, not measured speedup.

## Reproduction and verification

Source script: [`03_paired_interventions.py`](../03_paired_interventions.py).
It writes the PDF and a JSON reduction with endpoint identities, integer
counters, source hash and exact contrasts. It checks counter pooling and
matched identities, and reconciles the calculated differences to Analysis 018.

Three focused tests in [`test_paired_interventions.py`](../test_paired_interventions.py)
pass: approved numerical values and coverage; pairing independent of source
order; rejection of mismatched initialization or a missing threshold.
The 5.5-by-3.5-inch PDF was rendered with Poppler and visually checked.
All fonts are embedded, all text is inside the page, and the ten pairs appear
at the requested categorical positions in both panels. Existing analysis
figures and source measurements remain unchanged by the replot.

Final checks confirm the shared two-entry legend below the panels, both
two-line endpoint callouts, one x-axis title, both sets of categorical tick
labels, and all ten exact contrasts in both panels.
Frozen PDF SHA-256:
`68848db125449b16befa526377d4fda9e57a0166983552ba8792d7c2c93cb46f`.

## Manuscript adoption (11 September 2026)

At the author's request, the frozen figure and caption above replace the old
29-contrast main-text display in
[`training-results.tex`](../../../manuscript/draft/training-results.tex).
The manuscript copy has the same SHA-256 as the analysis original, recorded
in `manuscript/draft/figures/SOURCES.json`. It appears as Figure 3 on page 6.
Earlier artwork and all 29 appendix contrast rows are retained.

Section 4.2 is renamed "Paired effect of adding pressure" and now defines
the same-threshold pressure-on/off comparison, describes all five thresholds,
and emphasizes the complete target-set and equal-tensor normalization caveat.
At kappa = 0.5, the unrounded seven-site/four-site ratios are 4.84239359 for
the sparsity increment and 0.33441794 for the loss increment, supporting
"nearly five times" and "roughly one third" in the approved prose. These
ratios compare the two observed increments, not absolute endpoint performance.

The one-sentence local L1N/OL1 result is at the end of Section 4.1, linked to
Appendix D, Table 6. Analysis 018's retained local contrasts give loss changes
-0.0080803, -0.0060852, -0.0024516 and +0.0189099 at lambda = 0.05, 0.1,
0.5 and 1. The geometry diagnostic's wording and data remain intact. Its
paragraph is kept together below Figure 3, followed by the approved transition
to activation distributions. The opening specifies a pressure-free reference
at each threshold: the reference checkpoint changes with kappa, while the
comparison rule remains the same.

The rebuilt draft has 26 pages, resolved cross-references and no overfull
boxes. Three underfull vertical-box warnings remain. Figure placement and the
affected pages were rendered with Poppler and visually checked. The frozen
analysis PDF, source script, reduced data and all other manuscript section
sources are unchanged by this adoption.

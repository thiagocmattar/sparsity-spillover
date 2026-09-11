# O003 - Matched effects of adding OL1

## Question and status

At a fixed threshold, what changes when OL1 pressure is added, and how does
that differ between the four-site and seven-site recipes?

The author requested this focused redesign on 11 September 2026, restricted
to Analysis 021. No manuscript source, figure copy or compiled draft is changed.
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
The subsequent readability pass adds a light grid, repeats the threshold
ticks and axis label on the top panel, and labels the zero references
"lower is better" and "higher is better". Height increases from 3.25 to
3.5 inches to fit the repeated axis. The final wording pass adds the prominent
definition "Delta = (+OL1) - (no pressure), at matched kappa" above both
panels. Titles read "Effect of adding OL1 on validation loss" and "Effect of
adding OL1 on model-wide sparsity"; y-axes likewise specify changes from
adding OL1. The figure is now 5.5 by 3.75 inches, preserving the data-region
height while accommodating the definition. The contrasts and source data
are unchanged; the reference remains no pressure at the same threshold.

**The incremental effect of pressure depends on threshold and target set.**
Matched changes from adding OL1 to four-site (blue diamonds) and seven-site
(orange triangles) Pythia-14M recipes at each threshold kappa. Every point is
pressure-on minus pressure-off at the same threshold: A4-OL1 minus A4 or
A7-OL1 minus A7, with lambda = b = 1. (a) Validation-loss change; negative
values indicate improvement. (b) Model-wide sparsity change in percentage
points; positive values indicate more sparsity. Dotted lines mark zero change.
Annotations report both changes at kappa = 0.5. Four-site recipes target
a,m,h,z; seven-site recipes additionally target post-RoPE q,k and v. The ten
pairs share initialization, data order and training budget, and use full
338-sequence validation. Thresholds are equally spaced categories; connecting
lines organize separately trained endpoints. These are one-seed contrasts,
without estimates of variation across seeds.

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
The 5.5-by-3.75-inch PDF was rendered with Poppler and visually checked.
All fonts are embedded, all text is inside the page, and the ten pairs appear
at the requested categorical positions in both panels. Existing analysis
figures and the entire manuscript draft remain unchanged.

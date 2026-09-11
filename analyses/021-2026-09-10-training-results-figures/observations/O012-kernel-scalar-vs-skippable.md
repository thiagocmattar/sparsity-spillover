# O012 - Scalar sparsity versus kernel-skippable work

## Question

How do scalar zeros and actual instruction bypass relate to the matched gains
from projection and attention skipping?

## Method, coverage and source

[09_kernel_appendix_figures.py](../09_kernel_appendix_figures.py) displays four
existing relationships from the investigation's [30 rows](../investigation/data/checkpoints.json)
and [descriptive fits](../investigation/data/analysis.json). Fractions are
multiplied by 100 for plotting. Scalar fractions divide pooled zero-product
counts by projection-only or attention-only product counts, not by the full
model denominator. Projection bypass and attention skip fractions pool their
respective eligible instruction counts. No statistic is inferred from S_model.

Projection gain is the all-skips-off candidate latency divided by the
projection-on/attention-dense candidate latency. Attention gain is the latter
divided by full K050 latency. Both are full-model ratios, with the other
implementation choices retained. Canonical scalar counts use FP16 over all
338 validation blocks; kernel counters use BF16 over those blocks. Timing uses
64 inputs, seven passes and three processes per implementation, BF16,
batch-one 2,048-token full-vocabulary inference on RTX5090. No new model runs.

## Figure and proposed caption

[Appendix Figure 2](../figures/appendix/02-kernel-scalar-vs-skippable.pdf).

**Scalar zeros and skipped instructions describe different runtime opportunities.**
Each panel shows the same 30 checkpoints; gray circles, blue diamonds and orange
triangles identify baseline/local, four-site and seven-site recipes. The top
row compares scalar projection zeros and projection MMA bypass against the
gain from projection skipping. The bottom row compares scalar attention zeros
and attention MMA skips against attention-skipping gain. Dashed lines are
descriptive OLS fits with intercepts; dotted lines mark no net benefit at 1×.
Y-ranges match within each row but differ between rows. Projection MMA bypass
includes scalar substitution and padded rows, so it is not arithmetic eliminated.
Attention instruction counts include masked/padded work. Its skip predictor
takes only two values, with one checkpoint at the higher value.

## Result and caveats

Projection scalar sparsity has R² = 0.413, versus 0.946 for MMA bypass.
Attention has R² = 0.126 and 0.139 respectively, but every attention gain is
below 1×. The instruction-level attention fit is effectively a single-checkpoint
contrast, not broad evidence for a smooth response. Projection benefit is not
universal either: the low-sparsity region includes gains below one.

The in-figure clarification uses "matrix-multiply instructions skipped"
instead of "matrix-multiply work avoided" to preserve the counter's actual
meaning. [METHODS](../investigation/METHODS.md) distinguishes direct instrumentation,
operand-derived reconstruction, SIMT substitution, padding and missing counters.

## Verification

All 120 scatter coordinates and four OLS R² values match the retained fields
and independent Pearson checks. Both y-range pairs and the four 1× references
are verified. The five investigation tests pass; the single-page PDF was
rendered and checked for layout, embedded fonts and page bounds. The manuscript
and existing five-page diagnostic PDF are unchanged.

# Canonical R_model and full-model acceleration: single-panel linear fit

## Question and requested presentation

On 2026-09-07 the user requested a standalone version of Figure03's left
panel: full-model acceleration against canonical R_model, no variant-specific
legend, an expanded native linear y-axis, and a fitted linear model with R2.
The intended message is that a specialized sparse kernel makes R_model
informative about achieved speedup. This is a descriptive replot of completed
measurements, not a new experiment, kernel change, or cloud launch.

## Sources, coverage, and reduction

The source is `../results/summary-002.json`, already audited by
`../121_reduce_hybrid.py`; its hash is checked against
`../results/figure-provenance-002.json` before plotting. The frozen K050 policy
`../final-policy-002.json` is also hash-verified. The complete method and
numerical exceptions are retained in
[observation03](03-hybrid-rmodel-speedup-and-sparsity.md).

All 35 existing Pythia-14M variants are retained, including the five historical
A4+OL1@h variants and the A0 graph regression. Every point is the same native
graph / K050 graph paired geometric-mean ratio used in Figure03(a), equally
weighted across three fresh processes, each with 64 fixed validation input
identities and seven paired passes. The 105 processes all completed on one
RTX 5090, BF16, batch 1, sequence length 2,048, uncached causal inference with
all 50,304 vocabulary logits. Timing seed is 2504; checkpoint seed is 1234,
step 712. All numerical gates cover 500 validation documents / 338 complete
blocks per process, 692,224 input tokens, 691,886 prediction tokens, and the
reported excluded 1,444-token tail. All new variants pass the predeclared
gate; c33/c34 have small accepted nonzero errors, not universal exact identity.

Canonical R_model is recomputed from the stored pooled FP16 logical integer
numerator and model denominator, including the dense LM head. It is not the
BF16 operand-zero fraction or measured instruction savings. The x-axis
displays 100 R_model in percent.

The fit is unweighted ordinary least squares with a freely estimated intercept
on the **35 variant-level ratios**, not 105 pseudo-independent process points
or the individual timing samples. No weighting by timing uncertainty, family
balancing, point removal, forced unit intercept, log transform, or extrapolated
data is used. The line is drawn only between observed minimum and maximum x.
R2 is 1 - SSE/SST, where SST is centered about the mean observed speedup.
The six focused tests verify synthetic regression arithmetic, percentage versus
fraction units, an independent NumPy polyfit/correlation check on all 35
points, and rejection of incomplete, duplicated, unqualified or count-drifted
cohorts. The original measurements and Figure03 are not overwritten.

## Fitted result

With R_model expressed as a fraction in [0,1], the fitted model is:

```text
predicted speedup = 0.9412867254 + 3.6934308890 * R_model
R2 = 0.7796037979; n = 35
```

Equivalently, the slope is 0.03693430889 speedup-ratio units per percentage
point of canonical R_model. Residual SSE is 0.2958135174, SST is 1.3421897230,
and in-sample RMSE is 0.0919337521 speedup-ratio units. The fitted positive
association accounts for approximately 78.0% of the observed between-variant
speedup variance in this cohort. The intercept is estimated rather than
forced to one because dense-kernel and specialized-kernel overheads differ.

## Figure caption and visual encoding

**Figure05. Canonical logical sparsity is positively associated with achieved
full-model acceleration under the specialized K050 kernel.** Blue circles
show all 35 Pythia-14M variants, without color or shape distinctions between
families. Each point is native graph latency divided by K050 graph latency;
vertical bars span the three process geometric means and are not confidence
intervals. The orange line is an unweighted variant-level OLS fit with its
equation and R2 inset; the equation uses fractional R_model while the x-axis
displays percentages. The dashed gray line marks equal speed. The y-axis is
an explicitly labeled expanded linear view from 0.9x to 2.0x, omitting the
unused near-zero region while retaining every observation, error bar and the
fitted line. Hardware, precision, workload, baseline and timing replication
are identified on the figure. No variant legend or regression confidence band
is used.

Source script: `../124_acceleration_regression.py`.
Output: `../figures/05-hybrid-acceleration-linear-fit.pdf`.
Machine-readable fit, all points, source/script/figure hashes, coverage and
plotting versions: `../results/acceleration-regression-001.json`.

## Interpretation, caveats, and manuscript boundary

A supported phrasing of the intended message is: **For these Pythia-14M
variants, canonical R_model tracks full-model speedup under a specialized
sparse implementation (OLS R2 = 0.780).** This is empirical support for the
metric's relevance once sparse execution is implemented, not an identity
between logical sparsity and elapsed time or a validated universal predictor.

The fit is in-sample and post-hoc on an adaptively studied checkpoint cohort,
not held-out predictive validation. Variants share one training seed but differ
in topology, gates, learned weights, sparsity pattern and predictive quality;
uniform markers do not remove these confounders. Family sizes are unequal.
The highest-R_model endpoint is retained despite falling below the fitted
line. Full-model gains include fusion and other implementation effects, so
this scatter cannot attribute the complete gain to skipped products. The
matched sparse-path and fusion controls in observations03/04 remain the
attribution evidence, including negative attention-only results.

No manuscript TeX, scientific definition or finding registry is changed.
The central research index and the run README's initial status sentence
predate the completed K050 closeout; the terminal artifacts and appended
closeout record take precedence. This new observation does not silently
rewrite those historical statements. The PDF skill's render-and-inspect
workflow was used: the initial title/subtitle overlap was corrected, and the
final complete vector figure was rendered at 160 dpi and visually checked.

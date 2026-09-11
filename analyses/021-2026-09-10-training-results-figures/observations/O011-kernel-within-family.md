# O011 - Within-family sparsity-speedup association

## Question

Does the S_model-speedup relationship persist inside each intervention family,
rather than arising only from separation between family means?

## Method, coverage and source

[09_kernel_appendix_figures.py](../09_kernel_appendix_figures.py) reads the
unchanged investigation [checkpoints](../investigation/data/checkpoints.json)
and [fits](../investigation/data/analysis.json). Ten checkpoints belong to each
group: A0 plus nine one-site conditions, four-site recipes, and seven-site
recipes. Each panel displays its own unweighted descriptive OLS fit with an
intercept. The centered statistic subtracts each family's mean from both
S_model and speedup before pooling the 30 residual pairs and fitting OLS.

S_model uses canonical FP16 counts pooled over 338 validation blocks from all
500 MiniPile validation documents (1,444-token excluded tail). Full K050 speedup
uses the paired native reference: BF16, batch one, 2,048-token uncached
inference with full-vocabulary logits on RTX5090, 64 inputs, seven passes,
three processes. [Investigation O001](../investigation/observations/O001-kernel-investigation.md)
retains the raw-timing provenance. No new measurements or manuscript changes.

## Figure and proposed caption

[Appendix Figure 1](../figures/appendix/01-kernel-within-family.pdf).

**The sparsity-speedup relationship persists within intervention families.**
Each panel shows ten Pythia-14M checkpoints, with gray circles for baseline/local,
blue diamonds for four-site and orange triangles for seven-site recipes.
Dashed lines are separate descriptive OLS fits with intercepts; annotations
report Pearson r and R². Both axis ranges vary across panels to expose
within-family variation. The centered statistic pools residuals after
subtracting each family's mean from both variables. These comparisons do not
isolate a causal effect of sparsity or control for threshold, pressure, quality
or native execution cost.

## Result and caveats

Within-family r/R² values are 0.736/0.542, 0.885/0.784 and 0.873/0.762.
After within-family centering, R² is 0.660. The global association is not solely
family-mean separation; this is narrower than ruling out all confounding.

## Verification

All 30 plotted coordinates match the retained rows, the three R² values were
independently checked against squared Pearson correlation, and the centered
statistic is taken from the verified investigation. The five investigation
tests pass. The one-page PDF is rendered and visually checked, with embedded
fonts and text within the page. Source hashes and cohort IDs are recorded in
[kernel-appendix-figures.json](../data/kernel-appendix-figures.json).

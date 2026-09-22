# O017 - Pythia-14M layer sparsity: manuscript-style review figure

## Question and method

Restyle the existing site-by-layer appendix heatmaps using current manuscript
T/P nomenclature and the term activation sparsity. This is
the author's 19 September 2026 request for a preview before manuscript inclusion.
No new model evaluation or scientific comparison was added.

[plot_layer_sparsity.py](../plot_layer_sparsity.py) reads the unchanged
[O016 evidence](../data/evidence.json), verifies the hashes of the 12 source
activation-statistics files, and matches all 504 cells to their retained integer
numerators and denominators. Each value is 100 times the zero-activation count
divided by the total activation count at that site and layer.

## Coverage and legend

- Pythia-14M only; one seed; final models; 500 validation documents,
  338 complete 2,048-token sequences, 692,224 input tokens, and 1,444 excluded
  tail tokens.
- Two PDF pages: threshold kappa=0.05 and kappa=0.5, respectively.
- Each page has six panels: T4/P0, T4/Ph, T4/Pall in the first row;
  T7/P0, T7/Ph, T7/Pall in the second. Pall targets all thresholded sites.
- Seven activation sites by six layers, with layers numbered 1 through 6.
  Q and K are measured after RoPE.
- Following the author's preview feedback, panel titles are black and every
  panel uses the same `viridis` scale from 0 to 100%. Cell numerals are white
  on q/k/v rows and black on a/m/h/z rows, as requested after preview. Text
  has no background or outline.
  The displayed term is activation sparsity; the underlying zero criterion
  and original integer counts are unchanged.

## Figure and proposed caption

Analysis PDF: [appendix-14m-layer-sparsity.pdf](../figures/appendix-14m-layer-sparsity.pdf).
The byte-identical review copy is
[manuscript/draft/figures/appendix-14m-layer-sparsity.pdf](../../../../manuscript/draft/figures/appendix-14m-layer-sparsity.pdf).

**Activation sparsity by site and layer in Pythia-14M.** The six threshold/pressure
recipes at kappa=0.05 (page 1) and 0.5 (page 2). Each cell gives the percentage
of zero activations, pooling integer counts over all 338 complete validation
sequences for one site and layer. Q/K are post-RoPE. P0 denotes no pressure,
Ph denotes pressure only at h, and Pall denotes pressure at all thresholded
sites. All panels share the same 0-100% color scale. Cell values are rounded
to whole percentages; a displayed 0 or 100 does not establish a completely
dense or all-zero tensor.

## Result, caveats, and verification

All values and the two-page comparison are preserved. At kappa=0.5, the T7/Pall
maps show substantially greater Q/K/V sparsity than the other pressure scopes.
These are activation fractions, not model-wide sparsity or measured speedups.
The original one-seed, precision, and independent diagnostic-pass caveats in
[O016](O016-pressure-placement.md) still apply.

At kappa=0.05, pooled h/z sparsity increases from 81.34/77.64% under T4/P0
to 97.91/88.31% under T4/Ph. Under T7, the corresponding change is from
81.30/78.02% to 97.95/90.12%; T7/Pall instead gives 81.05/72.07%.
These values support the manuscript's nonlocal-response paragraph. The more
frequently active OL1 cap under T7/Pall is a possible explanation, not an
identified cause; O016 owns the pressure-budget evidence.

Both pages were rendered and inspected for label clipping, legibility, and
contrast. On 19 September, the author requested the polished nonlocal-response
paragraph and inclusion of both heatmap pages in the main results section.
`training-results.tex` now includes the figure under label
`fig:pressure-layer-sparsity`; the original appendix inclusion remains unchanged.
The exact plotted values, source digest,
and output digest are in [layer-sparsity-review.json](../data/layer-sparsity-review.json).

## 22 September: focused spillover interpretation in the main text

The user requested only page 1 (kappa=0.05) in the main results figure.
The PDF and appendix remain unchanged. The main caption now identifies the
T4/T7 rows and P0/Ph/Pall columns, rather than two threshold values.

The revised subsection follows three comparisons: h-only pressure increases
z sparsity in every layer under both T4 and T7; T4/Pall nearly eliminates z
in layers 2--4; T7/Pall increases pooled q/k/v sparsity while reducing h/z
sparsity relative to Ph in every layer. Exact T4/Pall z fractions in those
layers round to 99.825%, 99.846%, and 99.742%, so the text says near-total,
not completely zero. Reduced attention communication is proposed as a possible
contributor to quality loss, not a demonstrated causal mechanism. The quoted
ordinary validation losses (5.490 and 5.196) are from Analysis033 frontier.json.

The main text again cites `fig:ol1-geometry`, using
[Analysis024's geometry evidence](../../../024-2026-09-17-h-only-kernel-latency/observations/039-ol1-appendix.md).
At kappa=0.05, median removed-component/task-norm percentages are
66.841, 1.122, 0.169, 0.155 for T7/Pall, T4/Pall, T7/Ph, T4/Ph.
Corresponding median pre-cap norm ratios are 45.997, 0.275, 0.083, 0.102.
Projection frequency is approximately 100% for all four, so the subsection
compares magnitude, not a claimed frequency difference. These are adaptive
task-direction diagnostics, not raw task-gradient measurements.

The combined interpretation supports site-dependent susceptibility and a
possible shared-budget mechanism. Broadening the target set also changes
objective composition and normalization; no matched T7/P4 control isolates
q/k/v. The text therefore does not attribute the geometry solely to those
sites or identify projection as the cause of endpoint quality differences.

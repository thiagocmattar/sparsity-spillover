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

Both pages were rendered and inspected for label clipping, legibility, and
contrast. The original appendix PDF and manuscript TeX inclusion are unchanged;
the new figure awaits author review. The exact plotted values, source digest,
and output digest are in [layer-sparsity-review.json](../data/layer-sparsity-review.json).

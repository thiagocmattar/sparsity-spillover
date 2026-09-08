# Exact zeros and small nonzero activations: Figure 05-v2

## Question

Where do A0, A4-OL1 and A7-OL1 differ in exact zeros versus small nonzero
activations? This is an alternative presentation of O005, retained for user
review alongside the original Figure 05.

## Method

Pool integer counts over layers and validation blocks before division.
The top row is `100 * exact_zero_count / total`. The lower row is
`100 * (threshold_hits[0.01] - exact_zero_count) / total`, equivalently the
sum of the retained `(0,.001]` and `(.001,.01]` bins before division.
Exact zeros are excluded from the lower row. The fixed measurement cutoff
0.01 is distinct from the trained gate threshold kappa.

## Coverage

Seven Pythia-14M checkpoints: A0 and A4-OL1/A7-OL1 at kappa=0,.05,.5.
There are five common sites per checkpoint: h, m, q_post, k_post and v.
Full validation uses all 500 documents packed into 338 complete 2048-token
blocks, with the 1,444-token tail excluded. These are one-seed results;
thresholds and sites are not independent replicates. The same A0 baseline
appears in all three columns, producing 90 plotted markers from 35 site
measurements, each summarized by two fractions.

## Figure and caption

[Alternative publication PDF](../figures/05-v2-activation-mass-grid.pdf)

Exact zeros and small activations across sites (Pythia-14M). Columns denote
trained gate thresholds kappa=0,.05,.5. The top row shows the fraction of
activations equal to zero; the lower row shows the fraction satisfying
0<|x|<=0.01. Every panel uses a linear 0-100% scale. Color and marker identify
A0, A4-OL1 and A7-OL1; small vertical offsets separate recipes within each
site. The same A0 reference is repeated across columns. Sites h and m are
FFN hidden and input activations, q/k are post-RoPE queries/keys, and v is
the value operand. Fractions pool integer counts over layers and all complete
validation blocks. No interpolation between sites or thresholds is drawn.

## Result

At kappa=.5, A4-OL1/A7-OL1 have 99.940%/99.876% exact zeros at h.
A4-OL1 has 47.640%, 56.117% and 64.307% small nonzero mass at q, k and v;
A7-OL1 has 93.545%, 94.541% and 98.713% exact zeros at those sites.
The display separates magnitude shrinkage from exact zeros without a
logarithmic percentage axis.

## Caveats

Fractions near zero overlap visually with zero on the linear scale; exact
values remain in the table. The two smaller nonzero bins are intentionally
combined, and the >.01 tail is their complement with the exact-zero fraction.
All four bins and RMS remain in the supporting table. The absolute .01
cutoff does not normalize activation scale between sites or recipes.
A0 lacks a/z near-zero diagnostics, so those sites are omitted from this
matched display. No new measurements were performed. Complete recipes
change both gate and pressure sets; these marginals do not isolate a causal
spillover mechanism or imply runtime speedup.

## Source script and evidence

`plots.py:activations_v2`, included in `01_build.py`.
Sources: [O005](O005-activation-case-study.md),
[activation case table](../tables/activation-case-mass-bins.md), and
[figure_data.json](../figure_data.json), which retains source identities,
pooled integer counts, all four bins and RMS values.

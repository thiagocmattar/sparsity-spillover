# Figure 6: 14M quality, sparsity, and full-model latency

PDF: [06-14m-quality-sparsity-latency.pdf](../figures/06-14m-quality-sparsity-latency.pdf).
Source: [14_plot_14m_quality_latency.py](../14_plot_14m_quality_latency.py).
Exact point identities, coordinates, styling, limits, and hashes:
[14m-quality-sparsity-latency.json](../data/14m-quality-sparsity-latency.json).

## Question and requested revision

How do the same 14M pressure recipes compare in quality-sparsity and
sparsity-latency views? This is a new figure following the user's revised
guidelines, preserving Figure 1 and the manuscript. Both panels use the same
22 trained checkpoints: Base model, GeLU -> ReLU, and the four 4/7-Threshold
OL1(h)/OL1(all) families at κ=0,.01,.05,.1,.5. Pressure-free multisite recipes
and local pressure sweeps are excluded. Uniform small circular markers replace
the previous topology-specific shapes. H-pressure uses teal/purple and dashed
lines; all-pressure uses blue/orange and solid lines. There is one shared
six-entry legend, no pressure-free or nondominance legend, and no nondominated outlines.

## Method and coverage

Reuse the verified [common checkpoint table](../data/paper-checkpoints.json).
Panel (a) uses ordinary-final absolute validation loss and canonical count-pooled
FP16 model-wide sparsity. Panel (b) uses the same checkpoint keys and sparsities,
with full-model K050 latency from the retained BF16 timing records. The builder
asserts matching trained membership, ordered κ grids, and that all trained points
fit the visible limits. Analytic 4/7-Threshold ceilings are 12.833152%/29.952377%.

Panel (a) also directly annotates the two measured control post-hoc paths,
retaining all 20 records in provenance. Eight high-loss records lie above its
5.12-6.19 nats/token view; all are available in
[Appendix A1](../figures/A1-complete-quality-sparsity.pdf). Their measured eager
losses are unadjusted. No matched kernel timings exist for the clipped points,
so those paths are not plotted in panel (b). There is no new evaluation or timing.

## Publication caption

**14M quality-sparsity trade-offs and full-model latency.** (a) Absolute validation
loss versus model-wide sparsity. (b) Full-model K050 latency versus the same
sparsity measure for the identical 22 trained checkpoints. The Base model and
GeLU -> ReLU controls use gray and olive circles; the four threshold/pressure
recipes use the colors and line styles in the shared legend. Each pressure
recipe uses OL1 with pressure on h or all thresholded sites, respectively.
Lines connect separately trained κ settings in increasing order
(0,.01,.05,.1,.5), not training steps or attainable interpolated models.
Annotated dotted paths in (a) apply post-hoc magnitude clipping at a,m,h,z
to the fixed control checkpoints. Their high-loss continuation is shown in
Appendix A1; clipping has not been kernel-benchmarked. Vertical guides denote
the analytic 4-Threshold and 7-Threshold architecture/workload ceilings,
not measured speedup. Validation uses all 338 complete blocks from the 500
MiniPile validation documents, excluding the 1,444-token tail. Latency is the
geometric mean of 1,344 retained host timings on RTX 5090, BF16, batch one,
2,048-token full-sequence inference with all 50,304 logits. The 7-Thresh-h-Pressure
timings are from Run033; the other timings are from Run029 on a different
physical GPU/host. One final checkpoint is used per setting; no training-seed
uncertainty is represented.

## Result and associated proposed manuscript writing

Related manuscript sections:
[quality-sparsity results](../../../manuscript/draft/training-results.tex),
`sec:quality-sparsity-results`, and
[kernel execution](../../../manuscript/draft/kernel-autoresearch.tex),
`sec:kernel-autoresearch`. No manuscript file is edited.

> The pressure recipes expose distinct quality-sparsity operating points,
> while the same models' latency depends on the execution structure of their
> zeros. At κ=.5, 7-Threshold all-site pressure reaches 27.483% model-wide
> sparsity, compared with 16.664% for h-only pressure, but their observed
> full-model latencies are both approximately 0.474 ms. At the same threshold,
> 4-Threshold h-only pressure runs at 0.462 ms with 10.227% sparsity and loss
> 5.72267. The panels therefore support a joint quality/execution comparison
> rather than a runtime ranking inferred from aggregate sparsity alone.

## Caveats and verification

Clipping curves describe evaluated dose paths, with no nondominance claim or
extra Pareto markers. Logical sparsity and runtime are different estimands.
The ceilings describe selected-site analytic reach, and natural numerical zeros
may also occur outside that reach. Small cross-session latency differences do
not establish reliable winners or equivalence. The figure is one-seed descriptive
evidence. The original Figure 1's hash is preserved; the new PDF was rendered
and visually reviewed and all fonts are embedded. The existing seven scientific
checks still pass, and the new builder validates the matched 22-point cohort.

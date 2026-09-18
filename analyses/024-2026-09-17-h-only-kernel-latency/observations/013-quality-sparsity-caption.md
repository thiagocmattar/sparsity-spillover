# Figure 1: Threshold and pressure choices produce different quality-sparsity trade-offs

PDF: [01-quality-sparsity-tradeoffs.pdf](../figures/01-quality-sparsity-tradeoffs.pdf).
Manuscript placement: [Introduction](../../../manuscript/draft/introduction.tex),
`fig:quality-sparsity-overview`, referenced again in
[Section 4.1](../../../manuscript/draft/training-results.tex), `sec:quality-sparsity-results`.

## Question, method, and coverage

Which measured quality-sparsity operating points are available within each model size?
The declared comparison comprises 32 final 14M checkpoints and 22 final 70M checkpoints:
dense, ReLU, and the five-threshold 4-site/7-site OL1(h)/OL1(all) grids, plus the
two pressure-free multisite grids available at 14M. The eight historical local
pressure settings are outside this primary cohort. One checkpoint per setting;
initial parameter and training-order hashes match within size. Loss is the
ordinary reloaded final-checkpoint validation loss throughout, rather than mixing
it with losses recorded during logical diagnostics. All evaluations cover the
338 complete blocks from 500 MiniPile validation documents; the 1,444-token tail
is excluded. Model-wide sparsity pools the FP16 logical integer counters before
dividing by full-model product counts, including the untargeted vocabulary head.

The four control clipping paths contain 40 evaluated records. Their retained
FP16 eager losses are plotted without adjustment. At zero clipping, their maximum
absolute difference from ordinary-final loss is 0.000012548 nats/token.
Outlines use strict measured nondominance: maximize sparsity and minimize loss,
within size, including the control clipping records. No fitted envelope is used.

## Publication caption

**Threshold and pressure choices produce different quality-sparsity trade-offs.**
Panels show 14M and 70M final checkpoints. Loss is normalized to the corresponding
dense reference, ΔL = L − L_dense,m (5.2085825000 and 4.0997673379 nats/token).
Blue diamonds and orange triangles denote 4-site and 7-site thresholding;
gray stars and green circles denote dense and ReLU controls. Open markers with
solid lines indicate no pressure; filled markers with short and long dashes
indicate OL1(h) and OL1(all), respectively. Multisite lines connect separately
trained settings in increasing κ; they are not training trajectories or evidence
of attainable interpolated models. Labels identify κ=.05 and .5. Dotted control
paths apply post-hoc magnitude clipping at a,m,h,z, calibrated per layer/site
on ten training blocks at targets 0,.1,...,.9, with weights held fixed. Black
outlines mark nondominated measured points in this comparison. The shared Y
range includes all 54 trained endpoints; 14 high-loss control clipping records
(seven per size) lie outside it and are shown in Appendix Figure A1. Vertical
guides mark computational coverage of the 4-site and 7-site interventions,
not measured zero rates or runtime speedup ceilings. Full validation is used;
each trained setting has one final checkpoint and no independent-seed uncertainty.

## Result and proposed manuscript writing

> The evaluated operating points depend on both threshold and pressure scope.
> At 14M, 7-site OL1(h) with κ=.05 reaches 10.126% model-wide sparsity at
> loss 5.19496, close to the dense reference's 5.20858. At κ=.5, broad 7-site
> pressure reaches 27.483% sparsity but at loss 5.82939. The 70M recipes also
> exhibit distinct quality-sparsity regimes, with the broad-pressure comparison
> reversing at high threshold (Figure 3). These single-seed operating points
> motivate choosing a quality budget rather than ranking recipes by sparsity alone.

## Caveats and provenance

Nondominance is descriptive and sensitive to small numerical differences;
zero-dose eager/ordinary loss differences are not scientific improvements.
Computational coverage is 12.833%/29.952% at 14M and 37.063%/49.424% at 70M
for 4/7 sites. Untargeted numerical zeros can occur outside those sites, so the
guides are not hard empirical sparsity bounds. There are no pressure-free
multisite 70M checkpoints in the retained cohort.

Source: [13_rebuild_paper_figures.py](../13_rebuild_paper_figures.py),
[paper_quality_figures.py](../paper_quality_figures.py),
[checkpoint identities and source hashes](../data/paper-checkpoints.json), and
[exact selected/omitted IDs and axis limits](../data/paper-derived.json).
Associated writing above is proposed text; this task does not modify manuscript TeX.

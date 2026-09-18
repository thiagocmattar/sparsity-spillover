# Figure 5: More computational sparsity does not necessarily yield a better quality-latency trade-off

PDF: [05-quality-latency.pdf](../figures/05-quality-latency.pdf).
Manuscript placement: [Section 4.5](../../../manuscript/draft/kernel-autoresearch.tex),
`sec:kernel-autoresearch` and proposed replacement of `fig:kernel-autoresearch`'s
deployment comparison. The instruction explanation moves to Appendix A4.

## Question, method, and coverage

Which qualified measured models give useful quality-latency operating points?
The runtime cohort is the same 32/22-checkpoint primary cohort as Figure 1.
Every row is joined to ordinary-final loss and canonical sparsity through its
checkpoint content identity. Full qualification covers all 338 validation blocks;
timing uses 64 declared block indices, seven passes in three fresh processes.
Latency is the geometric mean of all 1,344 raw host times per checkpoint.
Exact process means, GPU identities, sessions, and kernel configurations are
retained separately in the common table. No intervals or training uncertainty
are inferred. Within each row the Y limits are identical and linear; model
sizes use separate scales.

## Publication caption

**More computational sparsity does not necessarily yield a better quality-latency
trade-off.** Rows show 14M and 70M; columns plot full-model latency against
model-wide sparsity and dense-relative validation loss. All 32 14M and 22 70M
checkpoints in the declared runtime cohort passed numerical qualification on
complete validation. Measurements use an RTX 5090, BF16, batch one, uncached
2,048-token full-sequence inference, and full 50,304-vocabulary logits. The
implementations are final K050 at 14M and the qualified k050-70m-v2 shape port
at 70M, including the full configured model execution rather than a projection-only
ablation. Each latency is the geometric mean of 1,344 host timings (64 inputs ×
seven passes × three processes). Loss and logical sparsity use retained FP16
evaluations, not the BF16 timing pass. Horizontal lines show each size's optimized
dense-reference latency. Black outlines on the right mark observed nondominated
loss/latency configurations, without interpolation or a fitted frontier. Lines
join separately trained κ settings within recipes; .05/.5 labels appear on the
left and selected configurations are named on the right. Most 14M measurements
come from Run029; 7-site OL1(h) comes from Run033 on a different physical GPU/host.
All 70M timings come from Run035. Small cross-session differences do not establish
a reliable winner. The 70M full-model result does not isolate the profitability
of attention skipping.

## Result and proposed manuscript writing

> Quality changes which runtime operating points are attractive. At 14M and
> κ=.5, 4-site OL1(h) has loss 5.72267 and latency 0.46225 ms, versus 5.82939
> and 0.47337 ms for 7-site OL1(all), despite much less logical sparsity.
> At 70M, expanding 7-site pressure at κ=.5 increases sparsity and lowers loss
> (Figure 3), but latency rises from 1.61837 to 1.74842 ms. Thus the quality-sparsity
> improvement does not translate into a simultaneous latency improvement. The
> observed frontier is a finite comparison of qualified checkpoints, conditional
> on these implementations, hardware sessions, and one training seed.

## Caveats and provenance

Nondominance uses exact displayed estimands: no other measured row has both
lower/equal loss and latency with one strict improvement. It does not imply
statistical significance. The 14M and 70M kernels have different shape-specific
configurations and unequal optimization effort; size comparisons are descriptive.
The old manuscript's pending 7-site h-only timing claim and 35-point cohort must
be updated if this proposed figure is adopted. The primary cohort here has 32
14M points because the eight local pressure settings are excluded and five new
7-site h-only endpoints are included. No training, timing, or cloud work was run.

Source: [paper_execution_figures.py](../paper_execution_figures.py),
[13_rebuild_paper_figures.py](../13_rebuild_paper_figures.py),
[common checkpoint/timing table](../data/paper-checkpoints.json), and
[frontier membership, limits, and reference values](../data/paper-derived.json).
Proposed manuscript text is recorded here without modifying TeX.

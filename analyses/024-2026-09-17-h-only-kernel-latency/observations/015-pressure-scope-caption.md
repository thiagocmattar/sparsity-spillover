# Figure 3: Global (Pall) vs. local (Ph) pressure paired analysis

PDF: [03-pressure-scope-threshold.pdf](../figures/03-pressure-scope-threshold.pdf).
Adopted unchanged on 18 September 2026 in [Section 4.2](../../../manuscript/draft/training-results.tex),
`sec:paired-results`, as manuscript Figure 4 (`fig:paired-intervention-effects`).
The subsection now gives a two-paragraph analysis: h-only pressure lowers loss
in all sixteen pairs through kappa=.1 and lowers latency in eighteen of twenty
pairs. The two small 14M latency reversals at kappa=.5 are explicit. The
previous pressure-budget diagnostics move to Appendix D.6. See the
[adoption record and numerical checks](../../../manuscript/draft/reviews/2026-09-18-paired-pressure/README.md).
The caption and method below retain the full analysis detail; the manuscript
uses a shorter caption with the same subtraction direction and timing limits.

## Question, method, and coverage

How does extending OL1 pressure from h to all selected threshold sites change
loss, model-wide sparsity, and full-model latency at matched model size,
threshold topology, and κ?
Twenty exact pairs use 40 unique checkpoints: two model sizes (14M and 70M),
two threshold scopes (four and seven sites), and five κ values. Every contrast
subtracts OL1(h) from OL1(all). Subtract ordinary-final loss and count-pooled
logical sparsity, and subtract retained checkpoint geometric mean latencies
(converted from milliseconds to microseconds). All pairs share initial-parameter
and training-schedule hashes, with λ=b=1 for OL1.

The user extended the figure to six panels: 14M in the top row and 70M in the
bottom row; loss, sparsity, and latency changes from left to right. The two rows
share loss and sparsity limits. Latency uses separate scales by size so the
smaller 14M differences remain legible. κ is categorical, with all five values
explicit. Validation covers 338 complete 2,048-token blocks from 500
documents, excluding the 1,444-token tail. The ordinary-final losses and
canonical FP16 counters are reused without re-evaluation.

Timings use RTX 5090, BF16, batch 1, sequence length 2,048, and the full 50,304-way
logit head. Each checkpoint has 64 common inputs × 7 passes × 3 processes
(1,344 timings). The plotted estimand is the difference between checkpoint
geometric means, not the mean of paired per-input timing differences.

The final styling follows Figure 6, widened to 12.8 inches for three columns
at the existing 5.8-inch height: DejaVu Sans, 14-point
overall title, 11.5-point panel titles, 11-point axes/legend and 10-point ticks.
Panel titles name paired loss/sparsity/latency differences; X axes read "Threshold κ".
Panels (b)/(e) use "Sparsity change ΔS_model (pp)"; (c)/(f) show "Latency change Δt (µs)".
The redundant OL1 legend
subtitle is removed; the two T/P contrast labels retain the subtraction direction.

## Publication caption

**Global (Pall) vs. local (Ph) pressure paired analysis.**
Validation-loss change (left), model-wide sparsity change (middle), and full-model
latency change (right), computed
as OL1(all) minus OL1(h) at 14M (top) and 70M (bottom), holding threshold
placement and κ fixed. Blue and orange curves show four- and seven-site
threshold topologies, respectively. The legend states each matched pressure
contrast explicitly; colours and uniform circular markers follow Figure 6.
Here global means pressure at all selected threshold sites, and local means h-only.
Negative loss changes are better; positive sparsity changes mean more logical
zero-operand products, in percentage points. Latency changes subtract checkpoint
geometric means in microseconds; negative values mean global pressure is faster.
Loss and sparsity share scales across rows; latency has a separate scale per
model size. Timings use RTX 5090, BF16, batch 1, full 2,048-token sequences and
the full logit head, with 1,344 timings per checkpoint. The 14M T7 contrast spans
Run029/Run033 GPU sessions, whereas T4 uses Run029 and all 70M timings use Run035.
The 70M kernel is the qualified K050-derived port. The horizontal line denotes
zero change. Lines join separately trained settings, not training trajectories or
independent replicates. Each setting has one seed and one final checkpoint,
with matched initialization, data order, and training budget. No independent-seed
uncertainty is represented, and small timing differences do not establish a
speedup or equivalence, particularly across sessions.

## Result

All-site pressure has higher loss in all ten 14M pairs and nine of ten 70M
pairs. The exception is the seven-site 70M setting at κ=0.5: loss changes by
-0.1213 nats/token and logical sparsity by +8.369 percentage points. Under
seven-site thresholding at κ=0.05, all-site pressure instead raises loss by
0.2679 (14M) and 0.2966 (70M), while sparsity changes by -0.263 and -0.371
points, respectively. At κ=0.5 the seven-site 14M contrast gives +10.819
points of sparsity for +0.0973 loss. These are matched descriptive differences
for the retained settings, not estimates over independent seeds.

For T7, all-site minus h-only latency at κ=0.05 is +59.643 µs (14M) and
+750.821 µs (70M). At κ=0.5 the differences are -0.522 µs and +130.050 µs.
For T4 at κ=0.5 they are -2.776 µs and +6.476 µs. All ten 70M differences are
positive. The near-zero 14M endpoints are descriptive, not evidence of a robust
latency improvement.

## Caveats and provenance

Pall denotes pressure on all selected threshold sites: a,m,h,z for T4, and
additionally q,k,v for T7. Changing pressure scope also changes the equal-tensor
normalization of its objective. These data do not isolate direct Q/K/V pressure
or attribute effects to normalization versus target selection. Five κ values
are five settings, not five seed replications. Logical sparsity is not runtime
speedup. This contrast requires no pressure-free multisite controls at 70M.

The two 14M T7 endpoints of each contrast use different physical GPU/host
sessions (Run029 versus Run033); checkpoint matching does not remove that
timing confound. T4 uses one session at 14M, and both topologies use Run035 at
70M. Within each pair, workload and the 64 timing input indices agree. The 70M
shape-specific port is not the unchanged 14M K050 binary. See the retained
[timing method and session caveat](006-paired-pressure-effects.md) and
[70M port qualification](004-70m-final-latency-topology.md).

Source: [paper_effect_figures.py](../paper_effect_figures.py),
[13_rebuild_paper_figures.py](../13_rebuild_paper_figures.py),
[checkpoint table](../data/paper-checkpoints.json), and
[20 unrounded contrast pairs](../data/paper-derived.json).

Regenerate only this figure with `13_rebuild_paper_figures.py --only-pressure-scope`.
The seven focused evidence tests pass, including direct integer-count contrast
checks, retained-latency subtraction, matched timing workload/indices, explicit
session provenance, and agreement with the difference of the two retained 14M
pressure-versus-none effects for all three metrics. The final styled PDF was rendered at 2,000 pixels
and visually inspected; all fonts are embedded. All 20 contrast records are
unchanged from the preceding version. Other figure PDFs and the manuscript
were unchanged by that plotting revision; the later manuscript adoption is
recorded above.

# Figure 3: Global (Pall) vs. local (Ph) pressure paired analysis

PDF: [03-pressure-scope-threshold.pdf](../figures/03-pressure-scope-threshold.pdf).
Proposed manuscript placement: [Section 4.2](../../../manuscript/draft/training-results.tex),
`sec:paired-results` (pressure-scope portion of `fig:paired-intervention-effects`).

## Question, method, and coverage

How does extending OL1 pressure from h to all selected threshold sites change
loss and model-wide sparsity at matched model size, threshold topology, and κ?
Twenty exact pairs use 40 unique checkpoints: two model sizes (14M and 70M),
two threshold scopes (four and seven sites), and five κ values. Every contrast
subtracts OL1(h) from OL1(all). Subtract ordinary-final loss and count-pooled
logical sparsity. All pairs share initial-parameter and training-schedule hashes,
with λ=b=1 for OL1.

The user selected four panels: 14M in the top row and 70M in the bottom row;
loss change in the left column and sparsity change in the right column.
The two rows share limits within each metric. κ is categorical, with all five
values explicit. Validation covers 338 complete 2,048-token blocks from 500
documents, excluding the 1,444-token tail. The ordinary-final losses and
canonical FP16 counters are reused without re-evaluation.

The final styling matches Figure 6: 8.6-inch width, DejaVu Sans, 14-point
overall title, 11.5-point panel titles, 11-point axes/legend and 10-point ticks.
Panel titles name paired loss/sparsity differences; X axes read "Threshold κ".
Panels (b)/(d) use "Sparsity change ΔS_model (pp)". The redundant OL1 legend
subtitle is removed; the two T/P contrast labels retain the subtraction direction.

## Publication caption

**Global (Pall) vs. local (Ph) pressure paired analysis.**
Validation-loss change (left) and model-wide sparsity change (right), computed
as OL1(all) minus OL1(h) at 14M (top) and 70M (bottom), holding threshold
placement and κ fixed. Blue and orange curves show four- and seven-site
threshold topologies, respectively. The legend states each matched pressure
contrast explicitly; colours and uniform circular markers follow Figure 6.
Here global means pressure at all selected threshold sites, and local means h-only.
Negative loss changes are better; positive sparsity changes mean more logical
zero-operand products, in percentage points. The horizontal line denotes zero
change. Lines join separately trained settings, not training trajectories or
independent replicates. Each setting has one seed and one final checkpoint,
with matched initialization, data order, and training budget. No independent-seed
uncertainty is represented.

## Result

All-site pressure has higher loss in all ten 14M pairs and nine of ten 70M
pairs. The exception is the seven-site 70M setting at κ=0.5: loss changes by
-0.1213 nats/token and logical sparsity by +8.369 percentage points. Under
seven-site thresholding at κ=0.05, all-site pressure instead raises loss by
0.2679 (14M) and 0.2966 (70M), while sparsity changes by -0.263 and -0.371
points, respectively. At κ=0.5 the seven-site 14M contrast gives +10.819
points of sparsity for +0.0973 loss. These are matched descriptive differences
for the retained settings, not estimates over independent seeds.

## Caveats and provenance

Pall denotes pressure on all selected threshold sites: a,m,h,z for T4, and
additionally q,k,v for T7. Changing pressure scope also changes the equal-tensor
normalization of its objective. These data do not isolate direct Q/K/V pressure
or attribute effects to normalization versus target selection. Five κ values
are five settings, not five seed replications. Logical sparsity is not runtime
speedup. This contrast requires no pressure-free multisite controls at 70M.

Source: [paper_effect_figures.py](../paper_effect_figures.py),
[13_rebuild_paper_figures.py](../13_rebuild_paper_figures.py),
[checkpoint table](../data/paper-checkpoints.json), and
[20 unrounded contrast pairs](../data/paper-derived.json).

Regenerate only this figure with `13_rebuild_paper_figures.py --only-pressure-scope`.
The seven focused evidence tests pass, including direct integer-count contrast
checks and agreement with the difference of the two retained 14M
pressure-versus-none effects. The final styled PDF was rendered at 1,800 pixels
and visually inspected; all fonts are embedded. All 20 contrast records are
unchanged from the preceding version. Other figure PDFs and the manuscript
are unchanged by this revision.

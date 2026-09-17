# O001 - Complete 70M trained quality-sparsity overview

## Question

Where do all completed 70M trained models lie in the model-wide
logical-sparsity versus validation-loss plane, including the new h-only OL1 grid?

## Sources, method and coverage

Run018 contributes 12 endpoints: GeLU, ReLU, and five thresholds each for
A4/A7 with pressure on all active gate sites. Run034 contributes ten endpoints:
the same A4/A7 thresholds with pressure only on h. The source verifications,
per-attempt manifests, terminal metrics and logical diagnostics are hashed in
`data/70m-quality-sparsity.json` (68 source files, 22 plotted points).

Both runs share the canonical initial parameter hash and realized training
schedule hash, seed 1234, 712 updates and 1,493,172,224 training input tokens.
Each final validation and logical pass covers 500 documents, 338 complete
2,048-token blocks and 692,224 input tokens, excluding the 1,444-token tail.

The x coordinate is 100 times the pooled integer block zero-product count
divided by the full-model product count, including the dense LM head. Counts
are not averaged as layer/batch percentages. The y coordinate is ordinary
final-checkpoint validation loss from `metrics.validation.final.loss` for
every point. The separate eager logical-pass loss is retained only for audit.
The maximum absolute difference between these two loss passes is
0.000244676714 nats across this cohort. No mixed-pass values enter the figure.

## Figure caption and legend

**Pythia-70M: model-wide sparsity versus final validation loss for all 22
completed training conditions.** Baseline GeLU and ReLU are isolated gray
diamonds and green squares. Blue circles denote A4 and orange triangles A7.
Filled markers and solid lines denote OL1 on all active sites; open markers
and dashed lines denote OL1 on h only. Lines connect kappa = 0, 0.01, 0.05,
0.1, 0.5 in threshold order, not a fitted frontier. Lower loss is better.
All points are visible. One initialization seed and one training-data pass;
model-wide sparsity measures exact-zero logical-product opportunity, not
realized latency or speedup. Post-hoc clipping points are not included.

## Result and limitations

The GeLU and ReLU controls have validation losses 4.099767 and 4.222745,
at 0.000454% and 10.065758% model-wide sparsity. Across the full plotted
cohort, the largest observed model-wide sparsity is 40.601872% for A7 with
all-site OL1 at kappa=0.5, with loss 5.215976.

At kappa=0.5, A4/h has loss 5.305709 and sparsity 29.235859%, versus
5.389543 and 35.596219% with all-site pressure. A7/h has loss 5.337238 and
sparsity 32.232517%, versus 5.215976 and 40.601872% with all-site pressure.
These endpoint descriptions do not establish a pressure pathway, universal
ordering, seed-robust effect, or runtime improvement. The absent pressure-free
70M A4/A7 arms cannot be inferred from 14M controls. Changing pressure scope
also changes the equal-layer-tensor pressure average. Different GPU hosts and
the H100 fallback for one condition are recorded, not treated as replicates.

## Reproduction

Source script: `../01_plot.py`.
Figure: `../figures/01-70m-quality-sparsity.pdf`.
Reduction and source identities: `../data/70m-quality-sparsity.json`.
This observation is descriptive and has not been promoted to a finding or TeX.

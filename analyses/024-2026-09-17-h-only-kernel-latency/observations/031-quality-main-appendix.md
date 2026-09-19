# Quality overview: 14M/70M main text and complete 410M appendix

**Subsequent figure revision:** [observation 032](032-quality-ol1-layout.md)
records the current OL1-only pressure view, independent loss scales and
three-row legend. Details below describe the original main/appendix split;
the source measurements and complete 410M appendix figure remain unchanged.

## Question and authorized scope

Focus the main quality overview on 14M and 70M, directly after Table 1,
followed by the existing base-model speedup figure. Retain the full 410M
trade-off panel in the appendix, next to the existing complete post-hoc
trajectories and endpoint data. Keep 410M's role explicit in the main text
as a fixed-token sparsity stress test, not a quality-scaling experiment.

## Figures and provenance

- [14-14m-70m-quality-sparsity.pdf](../figures/14-14m-70m-quality-sparsity.pdf):
  manuscript Figure 3, page 5, immediately after Table 1.
- [A5-410m-quality-sparsity.pdf](../figures/A5-410m-quality-sparsity.pdf):
  manuscript Figure 15, page 26, with the complete 410M loss range.
- Source script: [23_plot_quality_main_appendix.py](../23_plot_quality_main_appendix.py).
- Exact panel membership, records, plot limits and hashes:
  [quality-main-appendix.json](../data/quality-main-appendix.json).
- Unchanged measurement source:
  [all-model-quality-sparsity.json](../data/all-model-quality-sparsity.json)
  and [observation 028](028-all-model-quality-sparsity.md).
- Manuscript placement and verification:
  [review record](../../../manuscript/draft/reviews/2026-09-18-quality-scope-placement/README.md).

The original three-panel Figure 11 and every other existing analysis PDF
are retained unchanged. The existing historical 410M post-hoc figure is
unchanged; it is now manuscript Figure 16 on page 27.

## Method and coverage

This is an exact partition of the original 74 trained records and 60
Base/ReLU clipping records. Main panels retain all 40/22 trained conditions
and 20/20 clipping settings at 14M/70M. The appendix panel retains all twelve
trained 410M conditions and twenty control-clipping settings. Coordinates,
recipe identities, final-loss conventions and integer logical counts are
unchanged; no checkpoint was evaluated again.

Trained losses use ordinary reloaded final-checkpoint validation, while
clipping retains its original FP16 evaluation pass. Model-wide sparsity
pools integer logical counts with the full-model denominator, including the
dense logit head. Coverage remains all 338 complete 2,048-token validation
blocks from 500 documents (692,224 input tokens; 1,444-token tail excluded).
Recipes retain their original colors, line styles and circular markers.

The main panels keep the original shared loss range [4.0, 6.2]: every
trained point is visible, while eight 14M and five 70M clipping settings lie
above the view. Their complete paths remain in the existing appendix plots.
The 410M panel expands the loss range to [4.2, 9.35], so all trained and
control-clipping points are visible, including seven clipping settings
outside the old three-panel view. The separate historical 410M figure still
retains all 120 clipping evaluations across all twelve trained checkpoints.

## Main figure caption and associated writing

**Quality-sparsity trade-offs at 14M and 70M.** All 40/22 trained conditions,
with a shared loss scale. Colors distinguish executed recipes; lines connect
separate threshold settings (kappa = 0, .01, .05, .1, .5), or pressure
weights (lambda = .05, .1, .5, 1) for T1/P1. Its L1 and OL1 variants are
shown separately; all other pressured recipes use OL1. Dotted paths apply
post-hoc clipping at a, m, h, z to the fixed base and ReLU controls. Vertical
guides mark the T4/T7 analytic sparsity ceilings. All trained points are
visible; 8/5 high-loss clipping evaluations continue above the view
(complete paths in the appendix). Trained losses use ordinary
final-checkpoint evaluation; sparsity uses pooled FP16 logical counts over
all 338 complete validation blocks.

The main operating-regimes paragraph retains the high-threshold result:
T7/Pall reaches 27.48%/40.60% model-wide sparsity at 14M/70M, or
91.8%/82.2% of the respective ceilings, at loss increases of 0.62/1.12.
The speedup PDF is now manuscript Figure 4 on page 6. Its caption and
agentic-development discussion move after this quality overview unchanged.

## Appendix figure caption and associated writing

**Complete 410M quality-sparsity stress-test panel.** All twelve trained
conditions: Base, ReLU, and T4/Pall and T7/Pall at kappa = 0, .01, .05,
.1, .5. Dotted paths retain all ten post-hoc targets for each fixed
Base/ReLU control, including their full loss range. Vertical guides mark
the four- and seven-site sparsity ceilings. Trained losses use ordinary
final-checkpoint validation; clipping retains its original FP16 pass.
Sparsity pools logical counts over all 338 complete validation blocks.
The following figure additionally retains the historical clipping sweeps
of every trained recipe.

The appendix retains the 410M evidence: T7/Pall at kappa = .5 reaches
80.62% sparsity (92.4% of its ceiling), with loss 5.121 versus the base's
4.547. Its larger ceiling contributes to its greater sparsity reach. The
base loss exceeds 70M's 4.100 under the same 712 optimizer steps and
1.493B-token budget. Different tokens per parameter and the lower 410M
learning-rate schedule are plausible contributors, not isolated causes.
The training-trajectory and endpoint-table references remain available.

## Interpretation and caveats

The main experiment section defines 14M as the controlled intervention
study, 70M as a targeted replication of the regimes, and 410M as a
fixed-token stress test of attainable sparsity. Table 1 retains its 410M
ceiling column. This reorganization establishes neither a quality-scaling
claim nor a measured 410M runtime gain. Results remain single-seed and
observational across model sizes.

## Verification and reproduction

The four existing `test_all_model_quality.py` tests pass. An independent
record comparison verifies the exact disjoint partition of all 74 trained
and 60 control-clipping records. Source and copied artifact hashes match;
all previous analysis PDFs remain unchanged. Both standalone PDFs and the
rebuilt 33-page manuscript were rendered and visually checked. Table/figure
order and references resolve; there are no overfull boxes. No experiment,
kernel benchmark or cloud work was run.

Rebuild the two analysis figures and their export with:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/024-2026-09-17-h-only-kernel-latency/23_plot_quality_main_appendix.py
```

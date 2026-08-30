# O002 - Full-pass frontier with post-hoc control clipping

## Question

Where do the complete-validation A1 and A4 trained endpoints from Analysis 004
lie relative to uniform TEAL-style clipping curves attached to that analysis's
GeLU and ReLU control checkpoints, when displayed validation loss is capped at
6?

## Method and coverage

`03_plot_with_full_pass_frontier.py` joins two completed, verified sources:

- `../../004-2026-08-30-full-pass-quality-logical-frontier/comparison.json`
  supplies all 15 Analysis 004 endpoints: the GeLU and ReLU controls, five A1-H
  naive-L1 doses, four A1-H OL1 doses, and four trained A4 threshold doses.
- `../teal_frontier.json` supplies the complete target-sparsity sweep
  `p in {0.0, 0.1, ..., 0.9}` for uniform TEAL-style magnitude clipping of
  each retained control checkpoint.

Every underlying point covers all 500 MiniPile validation documents and all
338 complete 2,048-token blocks (692,224 input tokens); the 1,444-token tail is
excluded. The figure is a one-seed, Pythia-14M checkpoint comparison.

The post-hoc intervention clips matrix-input activations only during evaluation;
it does not retrain either control. The plot uses the Analysis 004 control
measurements as the `p=0` curve anchors so each post-hoc curve is visibly
attached to the exact endpoint already present in the trained comparison. The
independent Analysis 005 zero-threshold reproductions differ from those anchors
by `+0.00001156` loss and `+0.000000073` percentage points of `R_model` for
GeLU, and `-0.00001255` loss and `+0.000003530` percentage points of `R_model`
for ReLU.

The vertical display range is explicitly capped at validation loss 6. Post-hoc
points above the cap are omitted rather than moved to the boundary: GeLU
targets `p >= 0.5` and ReLU targets `p >= 0.6`. Their exact measurements remain
in `figure_data_02.json` and the complete sweep remains in `teal_frontier.json`
and Figure 1.

## Caption and legend

**Figure 2. Trained full-pass endpoints and post-hoc clipping on controls.**
All Analysis 004 A1-H naive-L1, A1-H OL1, and trained A4 threshold endpoints
are shown in the same measured `R_model`--final-validation-loss space as the
visible portions of the Analysis 005 GeLU- and ReLU-control curves. Magenta
crosses and black downward triangles are uniform TEAL-style magnitude clipping
applied only at evaluation to the retained controls; labels give the common
target sparsity `p`. The gray plus and black upward triangle are the canonical
GeLU and ReLU control anchors. The y-axis is capped at loss 6, and above-cap
post-hoc targets are explicitly omitted. Lines connect dose or target order
only. `R_model` is exact-zero logical-product opportunity, not measured
speedup; one seed.

## Result

The combined view preserves all 15 trained endpoints and shows how each
post-hoc control curve leaves its corresponding control in the same coordinate
system. Below the display cap, GeLU clipping contributes targets `p=0.1--0.4`
and reaches `R_model=5.125544%` at loss `5.605384`; ReLU clipping contributes
targets `p=0.1--0.5` and reaches `R_model=6.994451%` at loss `5.976274`.

This plot supports visual comparison of three intervention families, not a
claim of mechanistic equivalence: A1 and A4 are trained interventions, whereas
TEAL is post-hoc evaluation-only clipping on already-trained controls. The
cap prevents the high-loss tail of the complete TEAL sweep from compressing the
quality-relevant part of the trained frontier.

## Caveats

- The curves compare different intervention semantics and activation
  topologies; the connecting lines are ordering guides, not interpolated
  achievable frontiers.
- The y cap is a presentation rule, not data filtering in the source analysis.
- All measurements are from one seed, so small apparent differences lack
  replicate uncertainty.
- `R_model` is a logical opportunity fraction and must not be interpreted as
  runtime speedup.

## Provenance

- Source script: `../03_plot_with_full_pass_frontier.py`
- Reduced figure data: `../figure_data_02.json`
- Figure: `../figures/02-full-pass-frontier-with-posthoc-controls.pdf`
- Trained endpoint source: `../../004-2026-08-30-full-pass-quality-logical-frontier/comparison.json`
- Post-hoc source: `../teal_frontier.json`

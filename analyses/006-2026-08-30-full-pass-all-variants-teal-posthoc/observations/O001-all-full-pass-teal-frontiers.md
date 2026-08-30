# O001 - All full-pass checkpoints with post-hoc TEAL clipping

## Question

Does uniform TEAL-style post-hoc magnitude clipping extend the measured
quality--`R_model` frontier when it is applied not only to the GeLU and ReLU
controls, but also to every verified full-pass A1-H and A4-Z checkpoint?

## Method and coverage

Analysis 006 reuses the frozen Analysis 005 control sweeps and evaluates the
remaining 13 retained step-712 checkpoints at target sparsities
`p in {0.0, 0.1, ..., 0.9}`. The intervention is post-hoc and evaluation-only:
for each checkpoint, the first 10 complete source-order training blocks
calibrate a separate absolute-value threshold for every `site.layer` tensor at
`a`, `m`, `h`, and `z`; during validation, values with magnitude at or below
that threshold become exact zero. On A4-Z checkpoints this symmetric clip is
applied after the learned one-sided gate. It does not modify weights or training
state and is not TEAL's optional block-wise greedy allocation.

Every one of the 150 combined points covers all 500 validation documents, all
338 complete 2,048-token blocks, 692,224 input tokens, and 693,668 source
tokens. The 1,444-token tail is excluded. Counts are pooled as integers before
division. All 130 new points have finite activation statistics and reconciled
logical-product counts; all 13 zero-threshold losses reproduce their canonical
source endpoint exactly within the `5e-4` guard.

## Figure caption

**Pythia-14M trained A1/A4 endpoints with post-hoc TEAL on every checkpoint.**
The single panel contains all original trained endpoints, the visible portion
of every complete target-sparsity trajectory, the GeLU and ReLU post-hoc
control trajectories, and the pooled TEAL-augmented nondominated envelope.
Post-hoc clipping of the controls is explicitly evaluation-only. Thin lines
connect target order; trained endpoint lines connect dose order and are not
interpolated causal claims. The displayed validation-loss axis is capped at 6:
54 points above the cap are omitted only from the view, not clamped or removed
from the JSON and Markdown table. `R_model` is exact-zero logical-product
opportunity, not measured speedup. One seed.

## Result

The pooled descriptive envelope begins with A1-H naive L1 at `lambda=1`:
loss 5.1023 at `R_model=3.9493%` for `p=0`, reaching loss 5.2456 at
`R_model=6.4902%` for `p=0.3`. A4-Z one-sided thresholding at `kappa=0.1`
extends the envelope to loss 5.4196 at `R_model=8.9537%` (`p=0.5`) and loss
5.4466 at `R_model=9.3424%` (`p=0.6`). The `kappa=0.5` A4-Z checkpoint reaches
loss 5.7058 at `R_model=10.5631%` for `p=0.7` while remaining under the
displayed loss cap.

The trained exact-zero mass in several A4-Z checkpoints already exceeds lower
requested targets, so their calibrated added threshold stays at zero and
multiple target values coincide. For example, `kappa=0.5` is unchanged through
`p=0.5` at loss 5.6597 and `R_model=10.2154%`. These plateaus are retained as
distinct target rows in the table. Neither control lies on the pooled envelope
once the trained checkpoints are included, but both complete control
trajectories remain visible for the requested baseline comparison.

## Caveats

- This is one seed and one 14M-parameter model scale.
- A1-H and A4-Z differ in activation topology, pressure method, and trained gate
  operator; the pooled envelope is descriptive, not an isolated causal
  comparison between methods.
- Post-hoc clipping compounds learned sparsity interventions and measures
  logical opportunity only. No sparse kernel or runtime speedup is evaluated.
- Natural zero ties can exceed a requested target. Coincident target points are
  expected in that case.
- The loss-6 display cap hides 54 high-loss points; the complete values remain
  in `teal_all_variants.json`, `figure_data.json`, and `tables.md`.
- The reused controls retain the frozen Analysis 005 protocol hash while the
  130 new rows retain the Analysis 006 protocol hash; evidence origin is stored
  per row so the two protocols are not conflated.

## Provenance

- Source evaluator: `../01_evaluate.py`
- Figure/table reduction: `../02_plot.py`
- Combined evidence: `../teal_all_variants.json`
- Reduced evidence: `../figure_data.json`
- Complete table: `../tables.md`
- Figure: `../figures/01-all-full-pass-teal-frontiers.pdf`
- Reused control source: `../../005-2026-08-30-run004-controls-teal-posthoc/teal_controls.json`
- Source checkpoints and endpoint identities: `../../004-2026-08-30-full-pass-quality-logical-frontier/comparison.json`

This observation is not promoted to `research/findings/` and does not change
the manuscript.

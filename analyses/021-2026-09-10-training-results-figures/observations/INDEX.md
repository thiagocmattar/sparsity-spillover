# Observations

- [Investigation O001 - K050 structured sparsity and native-reference cost](../investigation/observations/O001-kernel-investigation.md):
  all 30 checkpoints and ten matched target-set pairs; distinguishes logical
  zeros, MMA bypass/SIMT substitution, attention skips and absolute versus
  native-relative runtime. Includes raw timing reconstruction and source trace.

- [O001 - 14M quality-sparsity overview](O001-14m-quality-sparsity.md): short
  labels, both clipping controls, architectural reach guides, and direct
  annotations of the local and broader pressure regimes.
- [O002 - OL1 conflict and target-set-dependent saturation](O002-ol1-geometry.md):
  all 7,120 multisite 14M optimizer steps; both groups have 99.9% conflict,
  with per-threshold opposing-component medians and IQRs in percent of the task
  norm, and cap activity 0.5%/97.0% at four/seven sites. Median removed components
  span 0.25-1.21%/19.78-84.78%, despite less negative seven-site median cosines.
- [O003 - Matched effects of adding OL1](O003-paired-interventions.md): ten
  four-/seven-site pressure additions at matched thresholds; stacked loss and
  model-wide sparsity differences. At kappa = 0.5, +2.50 pp / +0.38 loss versus
  +12.10 pp / +0.13 loss. Frozen after the final decluttering pass;
  adopted as manuscript Figure 3 with Section 4.2 focused on adding pressure.
- [O004 - Adding Q/K/V thresholding without pressure](O004-qkv-thresholding.md):
  five matched A7-minus-A4 threshold comparisons, shown as raw endpoints in a
  manuscript table. At kappa = 0.5, adding Q/K/V thresholds gives +5.17 pp
  model-wide sparsity for +0.043 validation loss.
- [O005 - Activation structure and operation accounting](O005-distributions-and-operations.md):
  the 14M kappa = 0.5 reversal in one composite figure. A4-OL1 has more FFN
  zeros but less model-wide sparsity; QK/PV contribute 0.06 versus 16.77 pp.
  Adopted as manuscript Figure 5; full threshold distributions and cross-size
  operation accounting are retained in Appendix D.1/D.2.
- [O006 - Quality costs and ceilings across sizes](O006-scale-transfer.md):
  one row of 14M/70M/410M panels with absolute validation loss,
  four-/seven-site ceiling guides, and common A7 utilization at kappa = 0.5.
  A separate Markdown table retains loss costs relative to A0; no table is
  embedded in the figure.
  Adopted as manuscript Figure 6, supporting Section 4.5's high-threshold
  complete-recipe comparison and architectural-ceiling interpretation.
- [O007 - A0 training loss and pre-clipping norms](O007-a0-optimization.md):
  two horizontal panels, all 712 training steps at each size, and consistent
  nine-step smoothing over faint raw trajectories. The gradient panel uses
  logged unscaled global L2 norms before clipping. Adopted as Figure 8 in
  Appendix C.4, after the realized-protocol table; no convergence claim.
- [O008 - K050 speedup and matched kernel ablations](O008-kernel-realization.md):
  all 30 K050 checkpoints, OLS R² = 0.817, and the same-cohort fusion,
  projection-skipping and attention-skipping GMs. Relative matched changes
  are +5.7% and -1.3%. Analysis-only; the manuscript is unchanged.

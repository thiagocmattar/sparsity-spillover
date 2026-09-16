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
- [O008 - K050 native-relative speedup and projection bypass](O008-kernel-realization.md):
  the same 30 checkpoints in both panels, with descriptive OLS R² = 0.817
  and 0.946. The projection gain uses matched raw candidate latencies, and
  bypass counts retain the SIMT/padding qualification. Analysis-only.
- [O009 - Figure 07 v2: model-wide sparsity versus projection-skipping gain](O009-kernel-realization-v2.md):
  a separate version with S_model on both x-axes. Panel (b) connects the
  four multisite threshold sweeps and two muted one-site pressure-weight
  sweeps, with one κ = 0.5 label per color and no displayed regression.
  Original Figure 07 is preserved.
- [O010 - Kernel manuscript adoption](O010-kernel-manuscript-adoption.md):
  frozen v2 becomes Figure 7 in Section 4.6. The section and Discussion focus
  on sparsity location and execution structure; the appendix retains search
  history, counter definitions, all 30 latency ablations and diagnostics.
- [O011 - Within-family kernel association](O011-kernel-within-family.md):
  three ten-checkpoint fits and the centered R² = 0.660, with independent axes.
- [O012 - Scalar zeros versus skippable work](O012-kernel-scalar-vs-skippable.md):
  a 2×2 projection/attention diagnostic; instruction bypass and scalar zeros
  have different relationships with matched skipping gains.
- [O013 - Native and optimized absolute latency](O013-kernel-absolute-latency.md):
  matched four-/seven-site curves in a 2×3 implementation-by-pressure layout,
  with y-ranges shared within columns. These three new PDFs are analysis-only.
- [O014 - Kernel v3 with a common A0 reference](O014-kernel-common-a0-reference.md):
  all 30 full-model points use one native A0 latency; R² = 0.510. Four-/seven-site
  OL1 at kappa = 0.5 reach 1.427x/1.385x. Panel (b) retains its matched ablation
  reference and layout; v2 and the manuscript are preserved.

- [O015 - Review corrections](O015-review-corrections.md): separate m/h statistics, dense distributions and pressure-free bars; absolute quality-latency comparison and projection bypass; qualified cross-size and pressure claims; nine-page main text and an AI-use statement. Five new PDFs and the three existing kernel appendix figures are adopted with source hashes.

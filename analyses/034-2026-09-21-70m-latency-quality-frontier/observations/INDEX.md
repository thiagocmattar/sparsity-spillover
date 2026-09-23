# Observations

- [001: Two-panel 70M comparison](001-two-panel.md): optimized latency versus
  sparsity and validation loss, highlighting T2/Ph and T7/Pall against native Base.
- [002: Combined 14M and 70M loss-latency panel](002-combined-scales.md): all
  66 focused checkpoints on logarithmic axes, with kernel and PyTorch Base
  references for both model sizes.
- [003: Combined scale comparison and transfer argument](003-manuscript-integration.md).
- [004: Lean appendix with native-Base deltas](004-lean-appendix.md): paired
  results with optimized latency and loss/latency changes; obsolete panels removed.
- [005: Kernel implementation and validation appendix](005-kernel-validation-appendix.md):
  compressed mechanisms, unchanged validation protocol, and retained execution controls.
- [006: Newer kernel overlay](006-recent-kernel-overlay.md): preserve all 68
  original points; add 18 Run051/052 measurements, with failed numerics marked
  and component-only experimental kernels excluded.

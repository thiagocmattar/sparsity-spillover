# Retained bottlenecks before Run052

Question: where can unchanged T2/Ph kappa=.05/.1 activations plausibly support
an additional sparse execution gain at 70M?

Method and coverage: `01_audit.py` reads the retained Run050 component screen,
Run051 full-grid table and two T2/Ph profile summaries, plus Analysis034's
14M figure data. [Machine-readable audit](../results/prior-audit.json) contains
every source path and SHA256. Checkpoints and inputs are unchanged. Historical
latencies come from different sessions; they do not constitute a new matched
cross-size result. Profiling covers four inputs, separately from latency runs.

The historical positive neighbouring-kappa drop is 16.697 microseconds at 14M
and 12.774 microseconds at 70M. Thus the retained 70M result does not meet the
requested larger absolute drop, even though it beats native PyTorch Base.
The signed deficit is 3.923 microseconds, without matched-session inference.

Five h layers use sparse gathered tensor cores; h layer0 and all z layers
remain dense. h index construction totals 18.66/18.55 microseconds for
kappa=.05/.1 in instrumented profiles; h gathered consumers total 58.11/47.84.
Dense z totals 56.34/56.48. These durations suggest examining metadata
construction and z consumers, but cannot be added to predict model latency.

The old global-packing alternatives already lost against efficient dense z.
The next implementation therefore tests CTA-local compaction plus tensor cores
and direct warp scans, keeping global-packing C as a control. Removing the
global buffer repeats scans across output tiles and may cost more overall.
No speedup for either new variation is asserted before GPU qualification.

Interpretation limits: high scalar sparsity is not sufficient for profitable
tile/row scheduling. Requested loads and padded products are logical counters,
not measured memory traffic. F003's finding remains: the existing result does
not demonstrate effective exploitation of broader h/z sparsity. The experiment
must earn h and z attribution separately with numerical checks and matched
skip controls. No new paper claim is promoted here; no figure is generated.

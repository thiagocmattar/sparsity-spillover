# 70M proposal frozen — 17 September 2026

User instruction: "freeze it, we'll do it later." The 70M proposal is deferred
until the user explicitly resumes it. No 70M run folder, implementation,
calibration, Pod, or training launch is authorized by this freeze.

Retained scope: A4-OL1(h) and A7-OL1(h), each at kappa 0.05 and 0.5,
with the prior matched 70M protocol used for the estimate.

- [70M-ETC.md](70M-ETC.md): four H200s, 115–160 minutes elapsed;
  estimates and evidence were committed in `f299fbf`.
- [70M-COST.md](70M-COST.md): approximately USD35–50 incremental usage,
  USD65–70 planning allowance; quote committed in `e08e8a2`.

These historical estimates are retained unchanged. A future launch requires
fresh price/availability and exact h-only calibration, followed by the
repository's design and launch confirmations. Current work switches to kernel
latency benchmarking of the completed 14M checkpoints.

## Subsequent instruction

Later on 17 September, after the kernel benchmark completed, the user resumed
planning for both 70M and 410M and requested time/cost for two versus all five
kappas. See [70M-410M-ETC-COST.md](70M-410M-ETC-COST.md). This supersedes the
planning pause only; no implementation, calibration or launch has been approved.

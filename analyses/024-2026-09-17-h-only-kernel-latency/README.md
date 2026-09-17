# H-only pressure: final K050 latency extension

This analysis owns the requested extension of the model-wide-sparsity versus
full-model-speedup and absolute-latency figures. Only the five new Run032
A7+OL1(h) final checkpoints are measured in Run033. All 35 historical Run029
checkpoints, including A4+OL1(h), are reused from their retained raw pairs.
The 70M training proposal remains frozen.

**Complete and verified:** all 15 new processes qualified over all 338 validation
blocks. All 180 returned files passed byte-count/SHA-256 verification before
the Pod was deleted. `01_reduce.py` requires all 15 complete new processes;
`02_plot.py` additionally requires every plotted endpoint to qualify.

## Results and figures

- [Full-model speedup](figures/01-14m-k050-sparsity-speedup.pdf), with
  [observation and caption](observations/001-sparsity-speedup.md).
- [Native and K050 latency](figures/02-14m-k050-sparsity-latency.pdf), with
  [observation and caption](observations/002-absolute-latency.md).
- [Complete 40-checkpoint table](TABLE.md) and [full-precision data](data/results.json).

The five new A7+OL1(h) speedups are **1.2060, 1.3929, 1.4706, 1.5649 and
1.7839x**, in ascending kappa order. At kappa=0.5, h-only pressure has
16.663640% S_model and 0.473888 ms K050 latency. The retained all-site A7
endpoint has 27.482684% S_model, 0.473366 ms and 1.7832x speedup. These two
observed runtime endpoints are close despite different logical sparsity;
this is a descriptive comparison across two GPU/host sessions, not a formal
equivalence test or a causal decomposition of the pressure intervention.

The publication PDFs were rendered and visually checked; fonts are embedded.
Three focused aggregation tests passed. The historical 35-checkpoint raw-pair
reduction reproduces the saved speedups and source hashes. Input preparation
and prelaunch verification passed all 246 bootstrap/contract tests in Run033.
No manuscript TeX or historical figure was overwritten.

The display includes all six A4/A7 pressure families at kappa
{0, 0.01, 0.05, 0.1, 0.5}, A0 and nine single-site controls. The clean figures
use no point annotations or fitted trend lines. Small speedup whiskers show
the range of three process means, not confidence intervals.

## Protocol and sources

- GPU: RTX5090; BF16, batch 1, sequence length 2048, full 50,304 logits.
- Full numerical validation: 338 blocks from 500 documents; 1,444-token tail
  excluded. Canonical FP16 sparsity pools integer logical-product counts.
- Speedup: geometric mean of all 1,344 paired native/K050 host-latency ratios.
- Absolute latency: geometric mean of the 1,344 raw host times. This agrees
  with the previous latency investigation's estimand; the original Run029
  summary's `native_ms`/`candidate_ms` instead aggregate process medians.
- Run029 and Run033 use different physical GPUs/hosts. Paired same-model
  native references preserve the speedup definition; cross-session effects
  remain a limitation, particularly for absolute latency.
- New skip-control ablations and a new A0 reference were not requested.
  Historical projection/attention-ablation panels stay in Analysis021.

`01_reduce.py` rechecks old source hashes and reconciles the raw-pair speedups
with Run029's saved reduction. It records every source hash and all exact
counts, qualification results, individual process means, runtime identities,
and timing block indices in `data/results.json`. `TABLE.md` exposes every
endpoint. Run033 owns source/input identity checks and verified retrieval.

Reproduce with `.venv/Scripts/python.exe analyses/024-2026-09-17-h-only-kernel-latency/01_reduce.py`
then `02_plot.py` from the same folder. Publication outputs are PDF only.

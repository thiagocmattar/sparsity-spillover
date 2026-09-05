# Observation 001 — Larger Pythia shapes do not rescue the exact-ELL path

## Question

Does increasing the selected Pythia workload from 70M to 410M amortize the
Sakana-derived exact-ELL path, and does the relation between `R_model` and
measured full-sequence prefill speedup persist?

## Method and coverage

This reduction compares Run 023 (70M) and Run 024 (410M) for A0, A1-H,
A4-OL1 at `kappa={0,0.5}`, and A7-OL1 at `kappa={0,0.5}`. Both runs use the
same physical H100 NVL, software versions, upstream commit, full-sequence
prefill protocol, batch sizes one and 32, and seven paired timing blocks. Each
run passed complete 338-block validation, signed exact packing, shape,
numerical-equivalence, and official upstream positive-control checks. Model
size, checkpoint loss, activation distribution, layer count, and linear shapes
differ; the comparison is therefore descriptive.

## Results

| Condition | 70M B1 | 410M B1 | 70M B32 | 410M B32 |
|---|---:|---:|---:|---:|
| A0 | 0.183 | 0.049 | 0.277 | 0.155 |
| A1-H | 0.660 | 0.328 | 0.618 | 0.478 |
| A4-OL1, κ=0 | 0.583 | 0.074 | 0.277 | 0.111 |
| A4-OL1, κ=0.5 | 0.983 | 0.262 | 0.707 | 0.499 |
| A7-OL1, κ=0 | 0.467 | 0.085 | 0.252 | 0.118 |
| A7-OL1, κ=0.5 | 0.797 | 0.281 | 0.405 | 0.432 |

All six batch-one results become worse at 410M. Five of six batch-32 results
also become worse; only high-threshold A7 improves slightly, and it remains a
`2.32x` slowdown. Neither scale has a full-model break-even result.

The low-level evidence agrees. Run 023 has 0/108 linear pack+kernel
break-even measurements, with best complete/kernel-only speedups of
`0.603/0.836x`. Run 024 has 0/432, with best values `0.481/0.710x`. Dense
adapter controls remain near native, so adapter dispatch cannot account for
the large gap. For A1-H, which has similar median W2 zero mass at the two
scales, native full-model latency grows `3.92x` while sparse latency grows
`7.90x`.

A7 attention remains separate from the full model. At `kappa=0`, its median
speedup is effectively unchanged (`0.200→0.198x`). At `kappa=0.5`, much higher
410M Q/V zero mass raises it from `0.334x` to `0.564x`, but none of 48 410M
layer compositions breaks even.

## Interpretation

The “model is simply too small to amortize fixed overhead” hypothesis is not
supported through 410M. The derived kernels execute correctly, but their
throughput scales less favorably than dense H100 BF16 GEMM at the tested
Pythia shapes; kernel-only results rule out packing as the sole cause. More
layers also repeat per-operation launch costs rather than amortizing one fixed
model-level cost.

The same GPU reproduces the official SparseLM0.5B speedup (`1.288x` during Run
023 and `1.274x` during Run 024), so parameter count alone is not the relevant
criterion. Architecture, matrix shape, batch, sparsity distribution, format,
and integration determine realized gain.

Within each predefined endpoint contrast, more logical opportunity generally
makes the sparse path less slow. Across topology and scale, however, `R_model`
does not identify the fastest endpoint and cannot be used as a universal
runtime conversion. The defensible paper use is a negative systems calibration
that preserves the distinction between logical opportunity and measured
speedup.

## Caveats

- Selected, one-seed endpoints are topology-confounded and differ in quality;
  this is not an inferential model-size effect.
- Timing covers uncached 2,048-token prefill on one H100, not decoding or other
  hardware.
- The derivative preserves every signed nonzero and does not prune; approximate
  formats may behave differently.
- A7 attention is an unfused standalone composition. The full model retains
  dense SDPA attention.
- Run 024's high-threshold A7 H100 V-only count differs by about 3.7 ppm from
  the pinned A100 union boundary, so its extended canonical `R_covered` is
  intentionally unavailable. Timings and component counts remain valid.

## Provenance

- Source script: `../01_compare.py`
- Machine-readable reductions: `../condition-comparison.csv`,
  `../matched-speedup-change.csv`, `../operation-comparison.csv`,
  `../attention-comparison.csv`, and `../comparison-summary.json`
- 70M source: `../../../runs/023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels/results/`
- 410M source: `../../../runs/024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels/results/`


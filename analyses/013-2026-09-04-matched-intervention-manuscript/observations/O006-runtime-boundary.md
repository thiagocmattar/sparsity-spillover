# O006 - Completed 70M runtime evidence

## Question and method

Does the completed Run 023 sparse implementation realize the measured
opportunity for its six 70M sentinels? Import its verified summary CSV and
verification JSON. This analysis performs no benchmark or checkpoint pass.

Full-model timing: one H100 NVL, BF16, uncached T=2048 prefill, batch 1/32,
seven randomized paired timing blocks, compilation and CUDA graphs disabled.
The signed sparse path replaces four downstream linears for A4/A7 and W2
for A0/ReLU; attention remains dense SDPA. Validation covers all 338 blocks
and reports the 1,444-token tail.

## Result display

The generated runtime.tex table includes all six sentinels and both batches.
There is no figure. All dense/sparse latency ratios are below one. Best:
A4+OL1@4, kappa=.5, batch 1, median .983496 with p10-p90 .982588-.985447.
The official upstream positive control achieves 1.2876x. The largest
BF16 dense/sparse validation-loss difference is .0006854 (< .001).
No tested linear primitive, even kernel-only, exceeds parity. Separate
unfused per-head attention compositions remain slower as well.

## Interpretation, caveats, and provenance

These results verify compatibility and a measured performance limitation of
one implementation. They do not establish performance of fused alternatives,
decoding, other devices/shapes, or explicitly structured training. The
70M A7 high-threshold checkpoint has 40.60% total opportunity, of which
30.18% is covered by the full-model sparse linear path. Timing-block
dispersion is separate from across-process, device, or seed uncertainty.

Primary observation:
[Run 023](../../../runs/023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels/observations/001-pythia70m-sentinel-sparse-kernel-feasibility.md).
Source scripts: Run 023 03_benchmark.py and 07_summarize.py; this analysis's
[evidence.py](../evidence.py) and [01_build.py](../01_build.py).
The completed result postdates adversarial-review-v3 and is newly incorporated
as a bounded negative runtime finding.

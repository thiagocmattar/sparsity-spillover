# Observation 001 — Pythia-410M sentinel sparse-kernel feasibility

## Question

Does moving the same selected intervention endpoints from Pythia-70M to
Pythia-410M make the Sakana-derived exact-ELL path large enough to amortize its
packing, dispatch, and kernel work? Can canonical logical-product opportunity
(`R_model`) be interpreted as realized full-sequence prefill gain?

## Method and coverage

Run 024 benchmarked A0, A1-H, A4-OL1 at `kappa={0,0.5}`, and A7-OL1 at
`kappa={0,0.5}` from verified Run-019 checkpoints. Each is the seed-1234
random-initialization pretraining realization after 712 optimizer boundaries
and 1,493,172,224 MiniPile input tokens; no released Pythia weights were
loaded. The checkpoints' weaker quality than the 70M endpoints is retained as
a limitation rather than filtered after seeing runtime results.

Run 024 and Run 023 used the same physical NVIDIA H100 NVL
(`GPU-ad566459-eb8b-a10b-4681-be5e0d9a33c9`), driver 570.133.20, CUDA 12.8,
PyTorch 2.11.0+cu128, and pinned upstream commit. The official unmodified
SparseLM0.5B positive control reached `1.2739x`, confirming a working upstream
speedup on its supported batch-64 workload. The derived path passed all seven
Pythia-410M shape checks.

Every loss covers all 338 complete 2,048-token blocks (692,224 tokens from all
500 documents) and reports the excluded 1,444-token tail. Source identity uses
the canonical eager-attention estimand at batch four. Native and sparse BF16
equivalence use batch one. Full-model timing uses randomized paired blocks at
batch one and the memory-qualified batch 32. A0/A1-H replace only `h -> W2`;
A4/A7 replace four linear paths. Full-model attention remains dense. A7's
Q-only/V-only attention path is a separate, unfused composition.

## Results

| Condition | Source loss | `R_model` | Linear `R_covered` | B1 speedup | B32 speedup |
|---|---:|---:|---:|---:|---:|
| A0 | 4.54744 | 0.000110 | 0.000110 | 0.049 | 0.155 |
| A1-H | 4.65127 | 0.210497 | 0.210475 | **0.328** | 0.478 |
| A4-OL1, κ=0 | 5.69206 | 0.383387 | 0.383387 | 0.074 | 0.111 |
| A4-OL1, κ=0.5 | 5.19110 | 0.715914 | 0.715830 | 0.262 | **0.499** |
| A7-OL1, κ=0 | 5.42629 | 0.411215 | 0.411214 | 0.085 | 0.118 |
| A7-OL1, κ=0.5 | 5.12063 | 0.806155 | 0.695516 | 0.281 | 0.432 |

No full-model result breaks even; the best batch-one result is `0.3275x` and
the best batch-32 result is `0.4989x`. The identity-wrapped dense path stays
near native dense (`0.950–1.008x` at batch one and `0.975–1.001x` at batch
32), so ordinary adapter dispatch is not the main slowdown. The sparse
operators themselves dominate it.

Within each predefined endpoint contrast, greater opportunity still moves in
the expected direction: A0→A1-H, A4 `kappa=0→0.5`, and A7
`kappa=0→0.5` all become less slow at both batches. Across topologies,
however, more `R_model` does not imply
a faster endpoint: A1-H is fastest at batch one despite substantially lower
`R_model` than the high-threshold A4/A7 conditions.

| Kernel evidence | 70M | 410M |
|---|---:|---:|
| Linear primitives | 108 | 432 |
| Linear pack+kernel break-even | 0 | 0 |
| Best pack+kernel speedup | 0.603 | 0.481 |
| Best kernel-only speedup | 0.836 | 0.710 |
| A7 attention compositions | 12 | 48 |
| Attention break-even | 0 | 0 |
| High-threshold attention median | 0.334 | 0.564 |
| High-threshold attention maximum | 0.361 | 0.604 |

The 410M linear path is not merely paying a fixed model-level overhead: none
of 432 primitives breaks even even with packing excluded. At the comparably
sparse A1-H endpoint, median per-layer W2 zero mass is 0.864 at 70M and 0.877
at 410M, but the kernel-only speedup falls from about `0.0526x` to `0.0420x`.
The native full model grows from 2.323 ms to 9.103 ms while the sparse path
grows from 3.523 ms to 27.840 ms. Larger dense shapes use the H100 more
effectively, while this exact-ELL implementation's work scales less favorably.

The attention result is more nuanced. With essentially no Q/V zeros, the
A7 `kappa=0` composition is unchanged (`0.200x` at 70M, `0.198x` at 410M).
At `kappa=0.5`, the 410M checkpoint has much higher median Q/V zero mass
(0.993/0.999) than the 70M checkpoint (0.762/0.909), and attention improves to
`0.564x`. It still does not break even in any layer, so the predeclared stop
rule does not authorize building a custom fused causal attention kernel.

For A7 `kappa=0.5`, the current H100 eager V-only count is 60,013,381 products
above the pinned Run-019 A100 eager P/V union, about 3.7 ppm of the component.
The exact H100 count and timing are retained, but an H100 component cannot be
claimed as a subset of that A100 union. Extended canonical `R_covered` is
therefore null; no count was capped and canonical `R_model` is unchanged.

## Interpretation

Pythia-410M is technically compatible with the derived kernels, including all
four linear sites and separate A7 attention. The performance hypothesis is
refuted for this implementation and workload: increasing parameter count and
matrix widths from 70M to 410M does not reach break-even. All six batch-one
sentinels become slower relative to their matched 70M results; at batch 32,
only high-threshold A7 improves slightly (`0.405→0.432x`) and remains far below
one.

This narrows the explanation from “fixed overhead on a model that is too
small.” The limiting factors include sparse-kernel throughput at these exact
Pythia shapes and per-operation launches, while dense BF16 GEMM benefits from
larger matrices. Packing is not the sole cause because kernel-only timing also
fails everywhere. Parameter count alone is not a sufficient workload-size
criterion: the upstream 0.5B/batch-64 positive control accelerates on the same
GPU, while the Pythia-410M B1/B32 exact-preservation path does not.

For the paper, this is a systems calibration and limitation, not a failed
activation-sparsity result. It supports the manuscript's deliberate separation
of logical opportunity from measured runtime: `R_model` orders available
work-elision opportunity within matched interventions, but does not specify
whether a concrete sparse kernel realizes it. A universal `R_model → speedup`
mapping is not supported by these data.

## Timing and cost

The clean six-condition attempt reached its final post-timing check in 57m53s.
Condition-file boundaries give the following operational planning estimates on
this H100: setup plus A0 6m13s, A1-H 3m06s, A4 `kappa=0` 10m15s, A4
`kappa=0.5` 8m59s, A7 `kappa=0` 15m34s, and A7 `kappa=0.5` 13m44s. At the
observed `$2.59/GPU-hour` plus about `$0.0139/hour` transient storage, a clean
prepared-Pod pass is approximately `$2.51`.

The verified science segments total 71m36s because the final condition was
rerun after a post-timing serialization invariant failed. The full Pod
lifecycle also included setup, failed infrastructure attempts, source-backend
diagnostics, and idle review time; rate-times-duration is approximately
`$8.4`. The billing API showed `$5.8007` at closeout but was visibly missing
later hourly buckets, so that number is provisional until posting completes.

## Caveats

- The model-size comparison is descriptive: the checkpoints have different
  activation distributions and losses, and size was not independently
  randomized.
- One seed, one physical H100, and seven timing blocks do not establish
  cross-device performance.
- This is uncached 2,048-token prefill, not autoregressive decoding.
- The derivative preserves every signed BF16 nonzero and does no pruning or
  capacity truncation; conclusions need not transfer to approximate sparse
  formats.
- `torch.compile` and CUDA graphs are off. Attention is a separate unfused
  composition and is not part of the full-model speedup.

## Provenance

- Run config and harness: `../config.yaml`, `../03_benchmark.py`
- Cross-device accounting correction:
  `../launch-control/attempt-008-community-h100-nvl-cross-device-attention-coverage/`
- Condition table: `../results/sentinel-summary.csv`
- Primitive table: `../results/linear-primitive-summary.csv`
- Attention table: `../results/attention-summary.csv`
- Passed verification and raw hashes: `../results/raw/verification.json`,
  `../results/raw/SHA256SUMS.txt`
- Cross-scale reduction:
  `../../../analyses/014-2026-09-04-pythia70m-vs-410m-sparse-kernel-sentinels/`
- Source summarizer: `../07_summarize.py`

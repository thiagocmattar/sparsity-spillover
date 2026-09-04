# Observation 001 — Pythia-70M sentinel sparse-kernel feasibility

## Question

Can the pinned Sakana Sparse-er/Faster-LLMs implementation, with the minimum
declared Pythia compatibility changes, execute the retained Pythia-70M ladder
faithfully? If so, does canonical logical-product opportunity (`R_model`) map
to measured full-sequence prefill speedup once actual kernel coverage and
integration costs are exposed?

## Method and coverage

Run 023 used the six approved Run-018 sentinels: A0, A1-H, A4-OL1 at
`kappa={0,0.5}`, and A7-OL1 at `kappa={0,0.5}`. Every source checkpoint is the
seed-1234 random-initialization pretraining realization after 712 optimizer
boundaries and 1,493,172,224 MiniPile input tokens; no released Pythia weights
were loaded.

The benchmark ran on one NVIDIA H100 NVL with CUDA 12.8, compute capability
9.0, PyTorch 2.11.0+cu128, and upstream commit
`661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5`. The official, unmodified
SparseLM0.5B positive control reached `1.2876x` over Torch. The separately
labeled Sakana-derived Pythia path then passed signed round-trip and numerical
checks for all seven required matrix shapes.

Each loss covers all 338 complete 2,048-token validation blocks (692,224 input
tokens from all 500 documents), with the 1,444-token tail excluded and
reported. Source FP16-autocast validation used batch four; native and sparse
BF16 equivalence used batch 32. Source loss matched Run 018 exactly for every
condition, and the largest sparse-versus-dense BF16 loss difference was
`0.0006854`, below the `0.001` gate.

Full-model timing used randomized paired blocks at batch sizes one and 32.
Speedup below is native-dense latency divided by sparse-path latency, so `>1`
means faster. A0/A1-H replace only `h -> W2`; A4/A7 replace four linear paths.
The full model deliberately leaves attention dense. A7's Q-only and V-only
attention kernels were tested separately as six per-layer, batch-one,
full-sequence compositions for each condition. There is no figure for this
observation; the tables are the result display.

## Results

| Condition | Source loss | `R_model` | `R_covered,linear` | `R_covered,+attention` | B1 speedup | B32 speedup |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 4.09977 | 0.000005 | 0.000005 | — | 0.183 | 0.277 |
| A1-H | 4.22274 | 0.100658 | 0.100658 | — | 0.660 | 0.618 |
| A4-OL1, κ=0 | 4.80529 | 0.256725 | 0.256683 | — | 0.583 | 0.277 |
| A4-OL1, κ=0.5 | 5.38954 | 0.355962 | 0.355850 | — | **0.983** | **0.707** |
| A7-OL1, κ=0 | 4.94120 | 0.235624 | 0.235624 | 0.235624 | 0.467 | 0.252 |
| A7-OL1, κ=0.5 | 5.21598 | 0.406019 | 0.301780 | 0.399741 | 0.797 | 0.405 |

The best full-model result, A4-OL1 at κ=0.5 and batch one, remains a slowdown:
its seven-block paired p10–p90 span is `0.9826–0.9854x`. Its batch-32 result is
`0.7066x`. Thus none of the measured full-model spans cross break-even.

| Condition | Linear primitives | Break-even count | Median pack+kernel speedup | Best pack+kernel | Best kernel-only |
|---|---:|---:|---:|---:|---:|
| A0 | 6 | 0 | 0.011 | 0.011 | 0.011 |
| A1-H | 6 | 0 | 0.049 | 0.067 | 0.073 |
| A4-OL1, κ=0 | 24 | 0 | 0.053 | 0.348 | 0.419 |
| A4-OL1, κ=0.5 | 24 | 0 | 0.382 | 0.603 | **0.836** |
| A7-OL1, κ=0 | 24 | 0 | 0.053 | 0.126 | 0.135 |
| A7-OL1, κ=0.5 | 24 | 0 | 0.142 | 0.516 | 0.704 |

No one of the 108 linear primitives broke even, even before packing: the best
kernel-only result was `0.836x`. The best complete primitive was `0.603x` on an
all-zero A4 `z -> Wo` input. This rules out packing cost as the sole cause; the
current kernel itself is not competitive with H100 dense BF16 GEMM at these
Pythia-70M shapes.

| A7 condition | Attention layers | Break-even count | Composition speedup range | Median |
|---|---:|---:|---:|---:|
| κ=0 | 6 | 0 | 0.199–0.200 | 0.200 |
| κ=0.5 | 6 | 0 | 0.259–0.361 | 0.334 |

For A7 κ=0.5, linear kernels cover 74.3% of canonical `R_model`; adding the
separate Q-only/V-only lower bound covers 98.5%. Nevertheless, the attention
composition is 2.8–3.9 times slower than its already-unfused dense comparator.
The configured non-break-even stop rule therefore applies, and no custom fused
causal kernel is authorized by this run.

Across the six selected sentinels, the descriptive Pearson association between
`R_model` and speedup is `0.807` at batch one but only `0.241` at batch 32;
using `R_covered,linear` gives `0.823` and `0.291`. These six points are selected
and topology-confounded, so those values are not an inferential regression.
More defensibly, every matched endpoint comparison moves in the expected
direction: A0→A1-H, A4 κ=0→0.5, and A7 κ=0→0.5 all become less slow as logical
opportunity increases. None becomes faster than dense.

## Interpretation

The technical compatibility question is answered positively for Pythia-70M:
signed activation packing, all four linear sites, complete validation, and
separate Q/V attention compositions run correctly on the pinned derivative.
The performance hypothesis is not supported. `R_model` is useful here as a
within-topology opportunity ordering, but it does not determine realized
speedup. Matrix shape, batch regime, kernel efficiency, dispatch, and
integration overhead remain decisive.

The official positive control matters: the environment can reproduce the
upstream gain on its supported large SparseLM workload. The Pythia result is
therefore evidence about this derivative and workload regime, rather than a
generic claim that the upstream method never accelerates inference.

For the paper, this is best treated as a systems calibration or limitation,
not as evidence that `R_model` equals runtime savings. It supports retaining
the manuscript's distinction between logical opportunity and measured gain.
The current data do not justify promoting the six interior 70M thresholds as a
speedup sweep. Measuring them could still estimate a negative/approach-to-zero
curve, but that is a new scientific choice after the approved stop rule.

## Caveats

- One seed, one H100 NVL, and seven paired timing blocks do not establish
  cross-device performance.
- This is uncached 2,048-token prefill, not autoregressive decoding.
- `torch.compile` and CUDA graphs are off; the implementation is a minimal
  auditable derivative, not a fully optimized serving stack.
- The standalone attention baseline is deliberately unfused and per-head.
  Full-model attention remains dense SDPA Flash, so standalone attention
  speedups must not be attached to full-model `R_covered,linear` results.
- Peak reserved memory was 52.21 GB. The exact batch-32 protocol therefore does
  not have safe headroom on a 48 GB GPU without changing memory behavior.
- The original verifier rejected a legitimate all-zero RMS. Recovery 007
  changed only that invariant and independently reverified the unchanged data.

## Provenance

- Benchmark source: `../03_benchmark.py`
- Executed compatibility patches:
  `../launch-control/attempt-003-community-h100-nvl-inference-context/benchmark-inference-context.patch`
  and
  `../launch-control/attempt-005-community-h100-nvl-eager-logical-context/eager-logical-context.patch`
- Verification correction:
  `../launch-control/attempt-007-verification-recovery/verification-all-zero-rms.patch`
- Condition table: `../results/sentinel-summary.csv`
- Primitive table: `../results/linear-primitive-summary.csv`
- Attention table: `../results/attention-summary.csv`
- Descriptive association: `../results/association-summary.json`
- Passed verification: `../results/raw/verification.json`
- Raw artifact hashes: `../results/raw/SHA256SUMS.txt`
- Source summarizer: `../07_summarize.py`
- Local-only transfer archives and sidecars:
  `../artifacts/attempts/005-community-h100-nvl-20260904-1750/incoming/`

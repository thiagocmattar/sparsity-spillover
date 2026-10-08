# Vocabulary-head kernel: diagnosis and quality

The kernel is numerically exact on this workload, but slower than dense cuBLAS. Skipping helps within the custom implementation; its baseline execution cost outweighs that saving.

## What is sparse and what executes densely

All four frozen 14M models have the original T2 corner gates at h, z and the head input. The head gate acts on the FP32 output of the learned affine final LayerNorm, using 8-token by 16-feature tiles. An original corner <= kappa rejects the entire tile. Surviving signed interiors, checkpoint weights and all 50,304 vocabulary scores are preserved.

- **Dense cuBLAS:** gated activations, dense block projections, dense vocabulary projection.
- **Custom no-skip:** the same activations and blocks; custom head preparation and projection with bypass disabled.
- **Custom sparse:** the same activations and blocks; only the vocabulary consumer skips empty reduction groups.

Ph and Ph+head name the retained checkpoints' training pressure histories. Both histories have the head gate. They do not name different inference kernels. No new training or pressure is applied. Earlier BF16 block consumers are not part of this FP32-parameter/FP16-autocast comparison.

```text
native transformer blocks (h/z gated; projections execute densely)
  -> learned affine final LayerNorm (FP32)
  -> original head 8x16 corner gate (FP32)
  -> FP16 input conversion; custom paths also retain native FP32 corner support
  -> vocabulary projection [2048,128] @ [128,50304]
       dense cuBLAS / same custom no-skip / custom sparse
  -> [2048,50304] FP16 logits (all entries written)
```

Loss, softmax, token selection and input staging are outside the latency measurements. Static weight conversion and custom fragment packing are outside timing and recorded separately.

## Quality and pooled head activation statistics

Native, dense, custom no-skip and custom sparse losses agree exactly within each process. All 12 processes have maximum logit error zero over all 338 validation blocks. These B1 RTX 5090 native losses are their own reference; Run066 B32 and local-laptop losses are provenance.

| kappa | Pressure history | Native = all-path loss | Exact zeros | Empty 8x16 gate tiles | Empty 16x16 consumer tiles / skipped MMA | RMS |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Ph | 5.529083497 | 25.324% | 25.324% | 14.380% | 1.530314 |
| 0 | Ph+head | 5.497690684 | 27.424% | 27.424% | 15.918% | 1.462316 |
| 0.1 | Ph | 5.547929021 | 17.336% | 17.336% | 7.075% | 1.557104 |
| 0.1 | Ph+head | 5.506956520 | 19.796% | 19.796% | 9.067% | 1.492585 |

Counts are pooled as integers over all 338 blocks before division. RMS uses the pooled sum of squares. Exact/near-zero counts and L1/L2 accumulators are retained in comparison.json and each head-diagnostics.json.

The M16/N256/K16 consumer combines two independent M8/K16 gate tiles. A reduction group is skipped only when **both** are empty. The 17.34-27.42% gate emptiness therefore becomes only 7.07-15.92% actual consumer/MMA bypass. This is a measured occupancy mismatch. MMA counts are source-level work counts, not measured DRAM traffic.

## Projection-only diagnostic

Times below are synchronized host microseconds. Projection-only excludes recurring input preparation and is a diagnostic, separate from the primary head path in REPORT.md.

| kappa | Pressure history | Dense cuBLAS | Custom no-skip | Custom sparse |
|---:|---|---:|---:|---:|
| 0 | Ph | 186.10 | 312.08 | 259.05 |
| 0 | Ph+head | 187.16 | 312.44 | 254.63 |
| 0.1 | Ph | 186.08 | 312.55 | 272.87 |
| 0.1 | Ph+head | 186.95 | 312.64 | 269.43 |

## Where the net latency change comes from

The measured head path includes the original gate plus recurring conversion/certification and projection. The same-custom-kernel control isolates the benefit of bypassing within that package. The cuBLAS comparison additionally reflects different custom layout and execution costs.

| kappa | Pressure history | Custom no-skip penalty vs dense | Head saving from skipping | Head saved (us) | Sparse forward slowdown | Range across 3 processes |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Ph | 63.27% | 15.47% | 51.21 | 7.56% | 7.55-7.58% |
| 0 | Ph+head | 62.70% | 16.79% | 55.67 | 7.19% | 7.17-7.21% |
| 0.1 | Ph | 63.21% | 11.43% | 37.87 | 8.86% | 8.83-8.90% |
| 0.1 | Ph+head | 62.68% | 12.33% | 40.94 | 8.53% | 8.51-8.55% |

The custom no-skip head is about 63% slower than cuBLAS. Skipping recovers 11.43-16.79% of its own head-path latency, or about 38-56 us. That saving is smaller than the baseline custom penalty, leaving the sparse head 35.37-44.55% slower than dense and complete forward 7.19-8.86% slower. All three processes reproduce the forward slowdown.

Every execution writes 2048 x 50304 FP16 logits (206,045,184 bytes). Sparsity changes reduction work, not that output contract. The source-level bypass, occupancy and paired timings establish the saving and baseline penalty; they do not identify an exact instruction, cache or DRAM bottleneck. Gate and cast times were not independently measured, and the three timing boundaries must not be subtracted as an additive decomposition.

## Independent CUDA-event check

The separate event-timed replays reproduce the slowdown. These are GPU elapsed times and are not substituted for the primary host measurements.

| kappa | Pressure history | Dense head (us) | Sparse head (us) | Dense forward (us) | Sparse forward (us) | Forward slowdown |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Ph | 198.44 | 274.46 | 994.84 | 1070.77 | 7.63% |
| 0 | Ph+head | 199.61 | 270.72 | 995.48 | 1067.59 | 7.24% |
| 0.1 | Ph | 198.70 | 288.45 | 995.30 | 1084.25 | 8.94% |
| 0.1 | Ph+head | 199.77 | 285.33 | 995.98 | 1081.78 | 8.61% |

## Coverage, aggregation and provenance

One physical RTX 5090; 3 fresh processes per checkpoint, 64 fixed inputs x 7 randomized paired passes per process and boundary. Runtime seed 2801, timing seed 2504. Full validation: 500 documents, 338 complete 2048-token blocks, 692,224 inputs, 691,886 predictions; 1,444 tail tokens excluded. The 12 full processes cover 12,168 full-graph logit comparisons and 4,608 standalone head-graph comparisons, all exact. The cloud component suite adds 48 checks.

Latency values are geometric means of the paired samples, aggregated equally over the three processes. Ratios are calculated from the paired records. Process ranges show repeatability on this GPU, not a population confidence interval. CUDA-event timings use separate replays and remain in comparison.json alongside the primary synchronized-host times.

- GPU UUID: `b9478201-c4f2-571b-4cae-c7e0d606a11f`.
- Verified deployment commit: `72ff70aba176da41a5af022d112f255d34c4222e`.
- [Primary table](REPORT.md), [raw reduction](comparison.json), [PDF](../figures/01-sparse-vocabulary-head.pdf).
- [Scientific transfer receipt](../transfer/receipt-001.json) verifies 163 files; [delivery receipt](../ops/delivery-receipt-001.json) verifies 8 supplemental files.
- [Resource closeout](../ops/teardown-001.json) confirms the owned Pod was deleted after verification. Existing foreign resources were unchanged.
- [20_analyze.py](../20_analyze.py) reduces raw paired records; [22_diagnostic_report.py](../22_diagnostic_report.py) generates these tables and interpretation.

This result refutes practical acceleration by this implementation for B1/T2048 at 14M. It does not refute sparse-head kernels generally, establish cached-decoding performance, or justify changing trained gates. No new training, manuscript change or finding promotion is included.

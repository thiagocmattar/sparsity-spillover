# 001 — Four h/z threshold endpoints and final K050 latency

**Question.** How do the four approved h/z thresholds behave with h-only OL1, lambda=1?

**Method and coverage.** Four random-initialized Pythia70M models share initialization, seed, MiniPile order and 712-step budget. Gates act only at h and z; pressure acts only at post-gate h. No post-hoc clipping. Training validation covers all 338 complete blocks from 500 documents (1,444 tail tokens excluded). The frozen Run035 k050-70m-v2 kernel, with the HZ allowlist bridge, uses RTX5090, BF16, B=1, T=2048, full logits, three fresh processes and 64 inputs × 7 passes each. Every process receives complete numerical qualification against native eager logits. The speedup denominator is native CUDA-graph inference of the same checkpoint.

**Result.** Exact-zero fractions pool integer counts. Latencies and paired speedups below are geometric means over 1,344 pairs per condition.

| kappa | Training val loss | h zero % | z zero % | R_model % | Native ms | K050 ms | Speedup | Qualified |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 4.128669 | 97.001 | 85.658 | 14.630 | 1.752241 | 2.772560 | 0.6320x | True |
| 0.01 | 4.149913 | 97.282 | 86.793 | 14.699 | 1.750878 | 2.728571 | 0.6417x | True |
| 0.05 | 4.192501 | 98.323 | 93.320 | 15.030 | 1.748872 | 2.451620 | 0.7134x | True |
| 0.1 | 4.182317 | 98.919 | 96.847 | 15.212 | 1.749263 | 2.302983 | 0.7596x | True |

**Numerical qualification.** 12/12 processes passed complete 338-block qualification. Maximum kernel logit absolute error: 0.5; relative L2: 0.0003220707; absolute pooled loss difference: 1.721969e-07. The fixed logit bound is abs(error) <= 0.25 + 0.02 * abs(reference), with relative L2 <= 0.02 and absolute pooled loss difference <= 0.001.

**Runtime result.** The frozen specialized kernel is slower than native graph inference for all four conditions on this workload.

**Interpretation limits.** One seed, one model size and one data pass. This joint intervention does not isolate the effects of either gate or pressure. Logical-product opportunity is distinct from measured speedup. BF16 qualification losses are retained separately from training validation. Timing of an unqualified condition, if any, is not evidence of a valid accelerated implementation. No manuscript or research finding is promoted.

**Source and provenance.** `../17_report.py`, `../artifacts/summary.json`, `../artifacts/verification.json`, `../latency/artifacts/verification.json`, and the raw artifacts listed with SHA-256 identities in the summary. No figure was required; the table above is the complete four-condition result.

# 001 — One h/z threshold endpoint and final K050 latency

**Question.** How does the approved h/z threshold of 0.5 behave with h-only OL1, lambda=1?

**Method and coverage.** One random-initialized Pythia70M model matches Run043 initialization, seed, MiniPile order and 712-step budget. Gates act only at h and z; pressure acts only at post-gate h. No post-hoc clipping. Training validation covers all 338 complete blocks from 500 documents (1,444 tail tokens excluded). The frozen Run035 k050-70m-v2 kernel, with the HZ allowlist bridge, uses RTX5090, BF16, B=1, T=2048, full logits, three fresh processes and 64 inputs × 7 passes each. Every process receives complete numerical qualification against native eager logits. The speedup denominator is native CUDA-graph inference of the same checkpoint.

**Result.** Exact-zero fractions pool integer counts. Latencies and paired speedups below are geometric means over 1,344 pairs per condition.

| kappa | Training val loss | h zero % | z zero % | R_model % | Native ms | K050 ms | Speedup | Qualified |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.5 | 4.838034 | 99.908 | 99.847 | 15.427 | 1.691950 | 1.690154 | 1.0011x | True |

**Numerical qualification.** 3/3 processes passed complete 338-block qualification. Maximum kernel logit absolute error: 0.5; relative L2: 7.76402958e-05; absolute pooled loss difference: 3.38747999e-08. The fixed logit bound is abs(error) <= 0.25 + 0.02 * abs(reference), with relative L2 <= 0.02 and absolute pooled loss difference <= 0.001.

**Runtime result.** The numerically qualified specialized kernel has a pooled native-relative ratio of 1.001063x. The three process ratios are 1.003766x, 0.997529x, 1.001903x. They straddle unity: this endpoint is effectively at parity under this protocol; no clear runtime advantage is demonstrated.

**Interpretation limits.** One seed, one model size and one data pass. This joint intervention does not isolate the effects of either gate or pressure. Logical-product opportunity is distinct from measured speedup. BF16 qualification losses are retained separately from training validation. Timing of an unqualified condition, if any, is not evidence of a valid accelerated implementation. No manuscript or research finding is promoted.

**Source and provenance.** `../17_report.py`, `../artifacts/summary.json`, `../artifacts/verification.json`, `../latency/artifacts/verification.json`, and the raw artifacts listed with SHA-256 identities in the summary. No figure was required; the table above is the complete single-condition result.

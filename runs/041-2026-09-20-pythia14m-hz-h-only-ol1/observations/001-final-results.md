# 001 — Four h/z threshold endpoints and final K050 latency

**Question.** How do the four approved h/z thresholds behave with h-only OL1, lambda=1?

**Method and coverage.** Four random-initialized Pythia14M models share initialization, seed, MiniPile order and 712-step budget. Gates act only at h and z; pressure acts only at post-gate h. No post-hoc clipping. Training validation covers all 338 complete blocks from 500 documents (1,444 tail tokens excluded). K050 uses RTX5090, BF16, B=1, T=2048, full logits, three fresh processes and 64 inputs × 7 passes each. Every process receives complete numerical qualification against native eager logits. The speedup denominator is native CUDA-graph inference of the same checkpoint.

**Result.** Exact-zero fractions pool integer counts. Latencies and paired speedups below are geometric means over 1,344 pairs per condition.

| kappa | Training val loss | h zero % | z zero % | R_model % | K050 ms | Speedup | Qualified |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 5.128602 | 93.143 | 60.485 | 4.631 | 0.612348 | 1.1670x | True |
| 0.01 | 5.130649 | 93.956 | 65.277 | 4.717 | 0.608618 | 1.1755x | True |
| 0.05 | 5.134379 | 96.264 | 82.825 | 5.004 | 0.590124 | 1.2125x | True |
| 0.1 | 5.151098 | 97.783 | 92.730 | 5.175 | 0.573427 | 1.2464x | True |

**Numerical qualification.** All 12 processes passed complete 338-block qualification. The maximum recorded K050 logit absolute error, relative L2 error and pooled loss difference versus native eager inference were all zero.

**Interpretation limits.** One seed, one model size and one data pass. This joint intervention does not isolate the effects of either gate or pressure. Logical-product opportunity is distinct from measured speedup. BF16 qualification losses are retained separately from training validation. Timing of an unqualified condition, if any, is not evidence of a valid accelerated implementation. No manuscript or research finding is promoted.

**Source and provenance.** `../17_report.py`, `../artifacts/summary.json`, `../artifacts/verification.json`, `../latency/artifacts/verification.json`, and the raw artifacts listed with SHA-256 identities in the summary. No figure was required; the table above is the complete four-condition result.

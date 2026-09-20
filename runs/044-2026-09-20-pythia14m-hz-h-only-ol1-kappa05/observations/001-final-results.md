# 001 — Kappa=0.5 h/z endpoint and final K050 latency

**Question.** How does the missing kappa=0.5 h/z threshold behave with h-only OL1, lambda=1?

**Method and coverage.** One random-initialized Pythia14M model matches Run041 initialization, seed, MiniPile order and 712-step budget. Gates act only at h and z; pressure acts only at post-gate h. No post-hoc clipping. Training validation covers all 338 complete blocks from 500 documents (1,444 tail tokens excluded). K050 uses RTX5090, BF16, B=1, T=2048, full logits, three fresh processes and 64 inputs × 7 passes each. Every process receives complete numerical qualification against native eager logits. The speedup denominator is native CUDA-graph inference of the same checkpoint.

**Result.** Exact-zero fractions pool integer counts. Latencies and paired speedups below are geometric means over 1,344 pairs per condition.

| kappa | Training val loss | h zero % | z zero % | R_model % | K050 ms | Speedup | Qualified |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.5 | 5.536255 | 99.841 | 99.841 | 5.339 | 0.506323 | 1.4189x | True |

**Numerical qualification.** Three processes received complete 338-block qualification; pass/fail and maximum logit absolute error, relative L2 error and pooled loss difference are retained in summary.json. Qualification is not assumed.

**Interpretation limits.** One seed, one model size and one data pass. This joint intervention does not isolate the effects of either gate or pressure. Logical-product opportunity is distinct from measured speedup. BF16 qualification losses are retained separately from training validation. Timing of an unqualified condition, if any, is not evidence of a valid accelerated implementation. No manuscript or research finding is promoted.

**Source and provenance.** `../17_report.py`, `../artifacts/summary.json`, `../artifacts/verification.json`, `../latency/artifacts/verification.json`, and the raw artifacts listed with SHA-256 identities in the summary. No figure was required; the table above is the complete single-condition result.

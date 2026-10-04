# Scalar versus corner tile: full-model latency

RTX5090 BF16 B1/T2048 full50304 logits; uncached; equal staging/setup excluded; six fresh processes

| Gate | kappa | Native ms | No h/z skip ms | Sparse ms | Native/sparse | No-skip/sparse | Tokens/s | BF16 loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scalar | 0 | 0.725639 | 0.622483 | 0.635143 | 1.1425x | 0.9801x | 3224473 | 5.221073 |
| tile | 0 | 1.236364 | 0.631184 | 0.564432 | 2.1905x | 1.1183x | 3628425 | 5.723608 |
| scalar | 0.1 | 0.828028 | 0.624111 | 0.556401 | 1.4882x | 1.1217x | 3680796 | 5.238963 |
| tile | 0.1 | 1.234983 | 0.631640 | 0.516416 | 2.3915x | 1.2231x | 3965799 | 5.837609 |

| kappa | Tile/scalar sparse speedup | Three process ratios |
|---:|---:|---|
| 0 | 1.125277x | 1.124714, 1.126098, 1.125018 |
| 0.1 | 1.077430x | 1.077128, 1.077249, 1.077913 |

Speedup >1 means the named implementation is faster. Ratios pool same-input/pass pairs, not selected medians.
The no-skip control disables h/z tile bypass and short-row substitution together; all other K050 settings are matched.
one seed/GPU; original FP16 quality is reported separately from BF16 runtime qualification; skipping gain is not measured DRAM bandwidth or decoding speed

# Scalar versus corner tile: full-model latency

RTX5090 BF16 B1/T2048 full50304 logits; uncached; equal staging/setup excluded; six fresh processes

| Gate | kappa | Native ms | No h/z skip ms | Sparse ms | Native/sparse | No-skip/sparse | Tokens/s | BF16 loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scalar | 0 | 0.645658 | 0.601531 | 0.606922 | 1.0638x | 0.9911x | 3374404 | 5.128739 |
| tile | 0 | 0.746999 | 0.603671 | 0.540220 | 1.3828x | 1.1175x | 3791047 | 5.556248 |
| scalar | 0.1 | 0.646298 | 0.603103 | 0.565242 | 1.1434x | 1.0670x | 3623224 | 5.153716 |
| tile | 0.1 | 0.747714 | 0.606030 | 0.508340 | 1.4709x | 1.1922x | 4028797 | 5.598785 |

| kappa | Tile/scalar sparse speedup | Three process ratios |
|---:|---:|---|
| 0 | 1.123472x | 1.124849, 1.121217, 1.124353 |
| 0.1 | 1.111937x | 1.110059, 1.112559, 1.113195 |

Speedup >1 means the named implementation is faster. Ratios pool same-input/pass pairs, not selected medians.
The no-skip control disables h/z tile bypass and short-row substitution together; all other K050 settings are matched.
one seed/GPU; tile models have higher original loss; skipping gain is not measured DRAM bandwidth or decoding speed

# Run 027 observations

- [01: Speedup versus logical sparsity](01-speedup-vs-rmodel.md): complete 35-checkpoint characterization and paired zero-skipping control.
- [02: Fusion and eligible sparsity](02-fusion-and-eligible-sparsity.md): distinguish QKV fusion, joint projection fusion, and the actual h/z products eligible for skipping.

Status: complete. Six of 35 variants qualify; the largest qualified speedup
is 1.8076x, with a 1.2365x controlled skip-toggle benefit. All failures remain
visible. No finding is promoted and no manuscript change is made.

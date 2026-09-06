# Run028 observations

- [01: R_model versus full-model and isolated skip speedup](01-rmodel-speedup-and-skip-contribution.md): complete35-variant/105-process K036 study; positive but small high-sparsity skip contribution, not the full optimization goal.
- [02: Controls and zero granularity](02-controls-and-zero-granularity.md): previous-kernel numerical failures retained, attention skipping remains slower, and scalar zeros frequently do not yield empty hardware fragments.

- [03: K050 R_model, full-model speedup and sparse-path contribution](03-hybrid-rmodel-speedup-and-sparsity.md): complete second 35-variant/105-process study; up to 1.739x graph acceleration with a substantial high-sparsity contribution, including numerical and quality caveats.
- [04: K050 controls and zero granularity](04-hybrid-controls-and-zero-granularity.md): fusion isolated, all six qualified previous endpoints improved, attention contribution remains negative, and hybrid MMA/SIMT counters distinguish bypass from scalar zeros.

Observations 01/02 retain the frozen K036 study; 03/04 report the separately
frozen K050 study. All are evidence records, not manuscript finding promotion.

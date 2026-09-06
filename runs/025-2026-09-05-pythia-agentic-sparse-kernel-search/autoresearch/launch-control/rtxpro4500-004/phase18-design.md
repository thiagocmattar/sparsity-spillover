# Adaptive Phase 18: direct 14M P0-to-K001 confirmation

Phase 17's first complete pair showed that the final K013 robustness policy is
slower than P0 at 14M A4-OL1 kappa 0.5, despite both candidates passing complete
validation. That comparison tests the final deployment policy, but not the
early kernel optimization that moved from the Sakana-derived P0 kernel to K001.

Phase 18 therefore tests the original implementation-improvement claim
directly, without altering a checkpoint, canonical `R_model`, timing workload,
or validation gate. It uses the same two predeclared high-sparsity development
conditions as Phase 17, runs P0 and K001 adjacently with counterbalanced order,
and collects three fresh eager-only processes per implementation and condition
(12 processes). All 338 validation blocks remain mandatory.

This is adaptive development evidence. Its selection was prompted by a Phase
17 development-endpoint result and must not be described as held-out evidence.
The estimated incremental ETC is 8--12 minutes after one K001 compilation, and
the expected incremental GPU cost is $0.10--$0.15 at $0.72/hour.

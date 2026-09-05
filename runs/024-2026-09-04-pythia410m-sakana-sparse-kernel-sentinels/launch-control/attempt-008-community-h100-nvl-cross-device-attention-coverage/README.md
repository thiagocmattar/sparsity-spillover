# Attempt 008: exact cross-device attention accounting

This append-only operational correction addresses the post-timing failure in
Attempt 007. The source checkpoint, validation data and coverage, sparse
kernels, timing protocol, tolerance, batch decision, and canonical logical
counts are unchanged.

`cross-device-attention-coverage.patch` makes the extended attention coverage
fail informatively rather than fabricate a ratio when an H100 Q-only or V-only
component cannot be a subset of its pinned A100 canonical union. It preserves
the exact current-device counts, the exact signed excess, and a null extended
`R_covered`. Comparable conditions retain the original calculation.

`recover_final_condition.py` reruns only A7-OL1 `kappa=0.5` from the start with
its original condition index. It verifies byte hashes of the five already
serialized condition results before and after, then writes the complete cohort.
It does not splice measurements within a condition. `launch_recovery.sh`
performs the patch check, focused smoke, recovery, fail-closed verification,
hash inventory, and packaging.

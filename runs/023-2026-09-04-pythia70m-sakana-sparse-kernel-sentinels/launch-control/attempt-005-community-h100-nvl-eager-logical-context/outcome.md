# Outcome

All six approved scientific benchmarks completed. The benchmark body took
860.4076 seconds and wrote six condition JSON files plus `cohort.json`.

The original post-run verifier then exited one because it required positive
RMS for every occupancy record. A4-OL1 at `kappa=0.5`, `z.layer_3` was entirely
zero (354,418,688/354,418,688 elements), making its recorded RMS of zero valid.
No result was corrupted. Verification-only Recoveries 006 and 007 preserved
this attempt; Recovery 007 passed and packaged it without rerunning benchmarks.

# Attempt 006: verification-only recovery

Attempt 005 completed all six approved scientific benchmarks, then the original
post-run verifier rejected `A4-OL1`, `kappa=0.5`, `z.layer_3` because it required
strictly positive RMS. The pooled record contains 354,418,688 exact zeros out of
354,418,688 elements, so RMS 0 is the correct value.

This recovery does not rerun or alter a benchmark result. It copies the sealed
verifier, applies `verification-all-zero-rms.patch`, and verifies the existing
Attempt 005 directory. The amended invariant accepts a finite nonnegative RMS
and requires RMS to be zero if and only if the complete integer-pooled record is
all zero. The patched verifier, patch, and hashes are added to the result archive.

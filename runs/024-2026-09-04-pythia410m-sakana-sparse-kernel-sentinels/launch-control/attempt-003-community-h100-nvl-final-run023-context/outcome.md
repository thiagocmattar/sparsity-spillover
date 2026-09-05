# Attempt 003 outcome

The final Run-023 context patch worked: A0, A1-H, and A4-OL1 `kappa=0`
completed and serialized, including all primitive timings. Attempt 003 then
failed closed during the source-validation gate for A4-OL1 `kappa=0.5`, before
that condition's sparse or timing work.

The implementation compared its SDPA/Flash validation pass against Run 019's
canonical eager-attention endpoint loss. Run 019 retains both estimands. For
this condition, the current SDPA loss was `5.191222323468451`; the archived
SDPA loss is `5.19106511392537` (difference `0.000157209543081`, inside the
unchanged `0.0002` tolerance), while the archived eager loss is
`5.19096581858291` (difference `0.000256504885541`). A7-OL1 `kappa=0.5` has an
archived eager-to-SDPA difference of `0.000209560408394`, so the mismatched gate
would fail even on the archived values.

Attempt 004 adds the six archived SDPA losses, preserves the canonical eager
losses explicitly, and verifies that source reproduction compares SDPA to
SDPA. It does not widen a tolerance. No partial result is reused.

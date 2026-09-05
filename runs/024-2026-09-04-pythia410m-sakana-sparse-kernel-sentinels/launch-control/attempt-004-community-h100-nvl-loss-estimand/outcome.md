# Attempt 004 outcome

The six-condition cohort completed A0, A1-H, both A4-OL1 sentinels, and
A7-OL1 `kappa=0`, including their complete validations and timing records.
It then failed closed at the source-loss gate for A7-OL1 `kappa=0.5`, before
any sparse or timing work for that final condition.

The measured H100-NVL Flash-SDPA loss was `5.1206965474687385`. Run 019's
archived A100-SXM4 SDPA loss is `5.1209012291135165`, giving an absolute
difference of `0.00020468164477804862`: only `0.00000468164477804861` beyond
the unchanged `0.0002` gate. The same H100 value is within
`0.00000487876361648` of Run 019's canonical eager loss
`5.120691668705122`.

This exposes a backend/device estimand problem, not evidence of a checkpoint
hash mismatch. Attempt 005 is a source-only diagnostic that alternates forced
SDPA and eager validation on the unchanged checkpoint and H100. No partial
Attempt-004 timing record will be promoted into the final cohort.


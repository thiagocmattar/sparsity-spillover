# Attempt 007 outcome and attention-opportunity audit

The clean six-condition execution serialized the first five conditions and
completed every validation and timing stage for A7-OL1 `kappa=0.5`. It then
failed closed before serializing that final condition because the H100 eager
`v`-only count exceeded the canonical Run-019 A100 eager P/V-union count.
No incomplete A7 result was retained.

The focused audit reproduced exact integer counts on the unchanged checkpoint,
data, code, and H100 NVL:

- H100 eager Q-only: `13,565,483,506,517` products, which is
  `989,349,938,396` below the canonical Q/K union.
- H100 eager V-only: `16,375,415,456,929` products, which is
  `60,013,381` above the canonical P/V union (`16,375,355,443,548`).
- The causal denominator is identical. The V-only excess is approximately
  3.7 parts per million of that component count.

This is a cross-device thresholded-activation comparability issue: a current
H100 component count cannot be combined with a canonical A100 union while
preserving the integer set-inclusion invariant. Attempt 008 therefore retains
the H100 counts and runtime measurements exactly but reports the extended
canonical `R_covered` as unavailable when a component exceeds its canonical
union. Counts are not capped, and canonical `R_model` is not changed.


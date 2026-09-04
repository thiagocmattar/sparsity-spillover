# Attention-count forensics

Complete A7-OL1 kappa=0 coverage was recomputed on the H100 NVL under both
attention contexts.

- SDPA-Flash q-only/v-only counts were 81,253/128,521. Comparing the V-only
  integer with the archived eager P/V union (105,729) is invalid.
- The original six-operation eager counter reproduced every archived H200
  integer exactly, including `R_model=0.23562417494680196`.
- Eager q-only/v-only counts were 101,484/105,729. They are valid subsets of
  the canonical eager Q/K and P/V unions (209,387/105,729).
- Source coverage was all 338 blocks and the eager loss was
  `4.941205969223609`, exactly matching the archived logical-product artifact.

Therefore the logical lower-bound measurement must use Run 018's eager
protocol, while hardware occupancy and all runtime timings remain under the
declared SDPA-Flash runtime protocol. Counts from the two contexts must not be
combined.
